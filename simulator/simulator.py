"""This module provides the main simulator functionality."""

import os
import pickle
import time
from typing import Union

import pandas as pd
import numpy as np

from grecco_sim.simulator import results
from grecco_sim.util.type_defs import RunParameters
from grecco_sim.coordinators import coord_interface
from grecco_sim.sim_models import grid, grid_node
from grecco_sim.util import type_defs


class Simulator(object):
    """The class performing a simulation run."""

    def __init__(
        self,
        grid_nodes: list[grid_node.GridNode],
        coordinator: coord_interface.CoordinatorInterface,
        run_params: RunParameters,
        time_index: pd.DatetimeIndex,
        grid: Union[grid.Grid, None] = None,
    ):

        self.horizon = run_params.sim_horizon
        self.run_params = run_params

        self.grid_nodes = grid_nodes
        self.coordinator = coordinator
        self.time_index = time_index
        self.ts = pd.DataFrame(index=self.time_index, columns=["iterations"])
        self.grid = grid

        self.max_market_iterations = run_params.max_market_iterations
        self.inspection_times = run_params.inspection if run_params.inspection is not None else []
        self.execution_time = 0.0
        self.multiprocess = False

    def _write_inspection(self, k):
        """Write a pickled version of the relevant objects for market clearing.
        Use unpickle_debugging.py to inspect and make custom market clearing.
        """

        os.makedirs(self.run_params.output_file_dir, exist_ok=True)

        obj = (self.grid_nodes, self.coordinator, k)

        with open(
            self.run_params.output_file_dir / f"coordination_state_at_k_{k}.pkl", "wb"
        ) as pickle_file:
            pickle.dump(obj, pickle_file)

    def _market_clearing(self, k: int):
        """
        Make a market clearing in a certain time step.

        args:
        :param: k: time step

        """

        # Get initial schedule of connected nodes without any central signal
        initial_futures = {
            node.sys_id: node.get_current_future(
                k, self.coordinator.horizon, self.coordinator.get_initial_signal()
            )
            for node in self.grid_nodes
        }

        # Is 'inspection' conceptually the same as logging?
        if k in self.inspection_times:
            self._write_inspection(k)

        signals = self.coordinator.get_signals(initial_futures)

        market_iterations = 1
        while market_iterations <= self.max_market_iterations:

            # If coordinator has converged: we are done for this timestep.
            if self.coordinator.has_converged(k):
                break

            # Get the futures of all agents.
            futures = {
                node.sys_id: node.get_current_future(
                    k, self.coordinator.horizon, signals[node.sys_id]
                )
                for node in self.grid_nodes
            }

            # Get signals in current market iteration
            signals = self.coordinator.get_signals(futures)
            market_iterations += 1

        if k % 5 == 0:
            print(
                f"For {self.coordinator.coord_name} at k={k} needed {market_iterations} market iterations"
            )
        self.ts.loc[self.time_index[k], "iterations"] = market_iterations

        return initial_futures, signals

    def run_sim(self):
        """

        A simulation is executed by iterating through the simulation time from 0 to the horizon -1.
        In each step, the coordination mechanism is executed in order to reach the optimum schedule.

        Once this is reached the final signals are broadcast to the agents using the node.apply_control(...)
        function and the nodes are evolved to the next time step.

        :return: nothing
        """
        start_time = time.time()

        for k in np.arange(self.horizon):
            # if self.time_index[k].time() >= datetime.time(4, 30):
            # pass

            initial_futures, signals = self._market_clearing(k)

            # Broadcast binding signals back to the Households/GridNodes and iterate the local
            # systems to the next time step.
            realized_future: dict[str, float] = (
                {}
            )  # Grid power of households in the current time step
            for node in self.grid_nodes:
                node.apply_control(signals[node.sys_id])
                realized_future[node.sys_id] = node.get_grid_power_at(k)

            rew = self.coordinator.get_cost_realization(initial_futures, realized_future, signals)
            # TODO this line entails a high computation time for pandas indexing. -> rewrite using numpy
            self.ts.loc[self.time_index[k], list(rew.keys())] = list(rew.values())

            # TODO here we would need the grid simulation of that time step
            if self.grid is not None:
                # self.grid.make_sim_one_timestep(realized_future)
                pass

        self.execution_time = time.time() - start_time

    def get_sim_result(
        self, opt_pars: type_defs.OptParameters, grid_pars: type_defs.GridDescription
    ) -> results.SimulationResult:
        """Access simulation results."""
        if self.execution_time <= 0:
            raise TypeError("Perform simulation before accessing results!")

        return results.SimulationResult().from_simulation(
            run_params=self.run_params,
            opt_pars=opt_pars,
            grid_pars=grid_pars,
            raw_output={node.sys_id: node.get_output() for node in self.grid_nodes},
            assigned_grid_fees=self.ts,
            sizing={node.sys_id: node.model_input["params"] for node in self.grid_nodes},
            time_index=self.time_index,
            exec_time=self.execution_time,
        )
