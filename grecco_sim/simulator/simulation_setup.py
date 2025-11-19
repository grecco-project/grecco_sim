import pandas as pd

from grecco_sim.coordinators import coord_base
from grecco_sim import coordinators

from grecco_sim.simulator import sim_input
from grecco_sim.util.type_defs import RunParameters
from grecco_sim.util import logger, data_io, type_defs, result

from grecco_sim.simulator import simulator
from grecco_sim.sim_models import grid_node

from grecco_sim.analysis import plotter, simulation_eval

from grecco_sim.simulator.metrics import evaluate_kpis

logger.set_logger()


class SimulationSetup(object):
    """
    This class represents a simulation run.
    """

    def __init__(self, run_parameters: RunParameters):
        """
        Args
        :param run_parameters: Parameters specifying configuration of the simulation run

        """
        self.run_pars = run_parameters

        self.time_index = pd.date_range(
            start=self.run_pars.start_time,
            freq=self.run_pars.dt,
            periods=self.run_pars.sim_horizon,
        )
        self.sys_ids = []

    def _get_nodes(self, opt_pars: type_defs.OptParameters):
        """
        This function creates the simulated households.

        Here, the task is, to efficiently create different simulation scenarios.

        :param opt_pars:
        :return:
        """

        if self.run_pars.scenario["name"] == "sample_scenario":
            dl = sim_input.SampleInputDataLoader(self.time_index, self.run_pars.scenario)
        elif self.run_pars.scenario["name"] == "dummy_data":
            dl = sim_input.DummyInputDataLoader(self.time_index, self.run_pars.scenario)
        elif "opfingen" in self.run_pars.scenario["name"]:
            dl = sim_input.PyPsaGridInputLoader(
                self.time_index, self.run_pars.scenario, self.run_pars.dt
            )
        else:
            raise ValueError(f"Scenario name {self.run_pars.scenario['name']} not known.")

        self.sys_ids = dl.get_sys_ids()

        model_input = {
            sys_id: dict(
                params=dl.get_parameters(sys_id),
                ts=dl.get_input_data(sys_id, self.run_pars.scenario),
            )
            for sys_id in self.sys_ids
        }

        all_params = {"hp_p": 0, "pv_p": 0, "ev_p": 0, "ev_c": 0, "bat_p": 0, "bat_c": 0}
        all_units = {"hp": 0, "pv": 0, "ev": 0, "bat": 0}

        for sys_id in model_input.keys():
            params = model_input[sys_id]["params"]
            for unit, unit_params in params.items():
                if unit_params._system == "hp":
                    all_params["hp_p"] += unit_params.p_max
                    all_units["hp"] += 1
                elif unit_params._system == "pv":
                    all_units["pv"] += 1
                elif unit_params._system == "ev":
                    all_params["ev_p"] += unit_params.p_lim_ac
                    all_params["ev_c"] += unit_params.capacity
                    all_units["ev"] += 1
                elif unit_params._system == "bat":
                    all_params["bat_p"] += unit_params.p_inv
                    all_params["bat_c"] += unit_params.capacity
                    all_units["bat"] += 1

        print("Total system capacities in this simulation:")
        print(all_params)
        print(all_units)

        return [
            grid_node.GridNode(sys_id, self.run_pars, model_input[sys_id], opt_pars)
            for sys_id in self.sys_ids
        ]

    def _get_central_coordinator(
        self, controller_pars: type_defs.OptParameters, grid_pars: type_defs.GridDescription
    ) -> coord_base.CoordinatorInterface:
        """
        Each simulation needs a coordinator and local EMS.
        The latter is created in the grid nodes.

        The former is created here and selected from several options.

        In general different coordination mechanisms should be implemented.
        You can add them here to compare the performance and feasibility.

        :param controller_pars: parameters determining behavior of the coordination mechanism.

        :return: the coordinator
        """

        coordination_mechanism = self.run_pars.coordination_mechanism

        for cm in coordinators.AVAILABLE_COORDINATORS:
            if cm.name == coordination_mechanism:
                return cm.coordinator(controller_pars, self.sys_ids, grid_pars)
        raise ValueError(f"Coordination mechanism '{coordination_mechanism}' not understood")

    def run_sim(
        self,
        opt_pars: type_defs.OptParameters,
        grid_pars: type_defs.GridDescription,
    ):
        """
        This function executes a simulation parameterized by the controller_pars and the run_pars given in
        initialization of the object.

        At this stage, the parameterization of a simulation is not final.
        However, it is necessary to distinguish between parameters of the simulation models and the controller.
        As of now, only the (local) controller is parameterized through the parameters given in this function.

        The workflow of the function is as follows.

        1. First, the grid nodes (or households) are created.
        A grid node consists of the physical model of the household (including energy system such as PV, Load, and/or
        battery system) and the local EMS, i.e. the local controller.

        2. The coordinator is created.
        The coordinator is the simulated equivalent to the community manager. It queries the households for information
        and then broadcasts signals depending on some algorithm.
        Local EMS and coordinator must be selected in a way that the communication is compatible.

        3. A simulation manager is created and the simulation is run.
        See the simulator class for possibly more details. The interaction between simulation model and controller
        is handled therein.

        4. Analyze and plot results.
        In the end, the resulting simulated time series is obtained from the simulator.
        It should be saved at some point.
        Some analysis is performed.

        :param sim_tag: a unique name for the simulation.
        :param controller_pars: parameters passed on to the local controller of the system
        :param show: flag if plots should be shown.
        :return:
        """
        # Set up simulation ==================================================
        # Set up individual grid nodes
        grid_nodes = self._get_nodes(opt_pars)

        # Set up one central controller connected to all households
        coordinator = self._get_central_coordinator(opt_pars, grid_pars)

        # Make simulation ======================================================
        sim_manager = simulator.Simulator(grid_nodes, coordinator, self.run_pars, self.time_index)
        sim_manager.run_sim()

        sim_result = sim_manager.get_sim_result(opt_pars, grid_pars)

        # Write sim results to disc
        sim_result.export_to_files()

        # Evaluate simulation.
        network_kpis, agent_kpis = evaluate_kpis(sim_result)
        eval_results = simulation_eval.evaluate_sim(sim_result)
        grid_path = self.run_pars.scenario["grid_data_path"]
        network_with_loads = result.map_result_to_pypsa(grid_path, sim_result)
        # ToDo: Discuss with Rebecca if flex analysis is needed.

        # Translate results to grid.

        # Plot some results. Plotting function decides if plots must be shown based on run_pars
        plotter.make_plots(sim_result)
        plotter.make_agent_plots(eval_results["agent_res"], sim_result.run_pars, "all")

        return eval_results
