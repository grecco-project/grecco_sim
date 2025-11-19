"""Module for definitions of simulation result and ways to save it."""

import dataclasses
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from grecco_sim.util import type_defs
from grecco_sim.util import data_io


def filter_agent_ts(result_dict, time_index) -> pd.DataFrame:
    """Compile dataframe for an individual agent based on result dict.

    result_dict: Dictionary of individual agent results from simulation.
    time_index: Simulation time index is set as agent dataframe index.

    Returns:
        DataFrame: Agents individual results for each time step.
    """

    data = {}

    # Total load.
    for key in ["grid", "c_sup", "c_feed"]:
        data[key] = result_dict[key]

    # Baseload
    data["p_el_load"] = result_dict["load"]["p_load"]

    # PV
    if "pv" in result_dict:
        data["p_el_pv"] = result_dict["pv"]["p_ac"]

    # BSS
    if "bat" in result_dict:
        data["bat_p_ac"] = result_dict["bat"]["p_ac"]
        data["bat_p_net"] = result_dict["bat"]["p_net"]
        # ToDo: Insert reason: Why do we omitt soc[-1]?
        data["bat_soc"] = result_dict["bat"]["soc"][:-1]

    # HP
    if "hp" in result_dict:
        data["hp_temp"] = result_dict["hp"]["temp"][:-1]
        data["hp_losses"] = result_dict["hp"]["losses"][:-1]
        data["hp_p_in"] = result_dict["hp"]["p_in"]
        data["hp_q_hp"] = result_dict["hp"]["q_hp"]
        data["hp_u_ext"] = result_dict["hp"]["u_ext"]

    # EV
    if "ev" in result_dict:
        data["ev_soc"] = result_dict["ev"]["soc"][:-1]
        data["ev_p_ac"] = result_dict["ev"]["p_ac"]
        data["ev_p_net"] = result_dict["ev"]["p_net"]

    return pd.DataFrame(data=data, index=time_index)


class SimulationResult:

    def __init__(
        self,
        run_pars: type_defs.RunParameters,
        opt_pars: type_defs.OptParameters,
        grid_pars: type_defs.GridDescription,
        agent_dict: dict[str, pd.DataFrame],
        assigned_grid_fees: pd.DataFrame,
        sys_pars: dict[str, dict[str, type_defs.SysPars]],
        time_index: pd.DatetimeIndex,
        execution_time: float = -1,
    ):

        self.run_pars = run_pars
        self.opt_pars = opt_pars
        self.grid_pars = grid_pars

        self.time_index = time_index

        self.assigned_grid_fees = assigned_grid_fees
        self.sys_pars = sys_pars

        p_grid = {sys_id: agent_dict[sys_id]["grid"] for sys_id in agent_dict}
        self.ts_grid = pd.DataFrame(index=time_index, data=p_grid)

        # Apply prefilters and potential aggregation to agent data.
        self.agents_ts = dict()
        for sys_id, ts_data in agent_dict.items():
            self.agents_ts[sys_id] = filter_agent_ts(ts_data, time_index)

        self.execution_time = execution_time

    @property
    def dt_h(self) -> float:
        return self.run_pars.dt_h

    @property
    def tag(self) -> str:
        return self.run_pars.sim_tag

    @property
    def sys_ids(self) -> list[str]:
        return list(self.agents_ts.keys())

    @property
    def flex_ts(self) -> pd.DataFrame:
        data = {}
        for sys_id, agent_df in self.agents_ts.items():
            if "bat_p_ac" in agent_df:
                data[f"{sys_id}_bat_p_ac"] = agent_df["bat_p_ac"]
            if "hp_p_in" in agent_df:
                data[f"{sys_id}_hp_p_in"] = agent_df["hp_p_in"]
            if "ev_soc" in agent_df:
                data[f"{sys_id}_ev_p_ac"] = agent_df["ev_p_ac"]
                data[f"{sys_id}_ev_soc"] = agent_df["ev_soc"]

        return pd.DataFrame(data=data)

    @property
    def p_trafo(self) -> pd.Series:
        """Return time series of summed up power at transformer."""
        return self.ts_grid.sum(axis=1)

    def export_to_files(self, result_dir: Optional[Path] = None) -> None:
        """Write SimResult in given directory.

        result_dir/
        ├─ agents/                    # Per-agent time-series CSVs.
        │  └─ {sys_id}.csv            # Written once per agent.
        ├─ nodal_load_{tag}.csv       # Net nodal load (self.ts_grid).
        ├─ signals_costs_{tag}.csv    # Grid fees (self.assigned_grid_fees).
        ├─ flex_power_{tag}.csv       # Flexible power (self.flex_ts).
        ├─ system_parameters.json     # Static unit parameters (self.sys_pars).
        └─ parameters.json            # Run/opt/grid parameters (see below).

        If no dir is given, refert to self.run_pars.output_file_dir.

        """

        if result_dir is None:
            result_dir = self.run_pars.output_file_dir

        if not result_dir.exists():
            result_dir.mkdir()
            print(f"Creating {result_dir}.")

        if not (result_dir / "agents").exists():
            (result_dir / "agents").mkdir()
            print(f"Creating {(result_dir / 'agents')}.")

        p = result_dir / f"nodal_load_{self.tag}.csv"
        data_io.write_ts(self.ts_grid, p)

        p = result_dir / f"signals_costs_{self.tag}.csv"
        data_io.write_ts(self.assigned_grid_fees, p)

        p = result_dir / f"flex_power_{self.tag}.csv"
        data_io.write_ts(self.flex_ts, p)

        for sys_id, agent_df in self.agents_ts.items():
            p = result_dir / "agents" / f"{sys_id}.csv"
            data_io.write_ts(agent_df, p)

        p = result_dir / "system_parameters.json"
        data_io.custom_json_dump(p, self.sys_pars)

        sim_pars = dict()
        sim_pars["run_pars"] = dataclasses.asdict(self.run_pars)
        sim_pars["opt_pars"] = dataclasses.asdict(self.opt_pars)
        sim_pars["grid_pars"] = dataclasses.asdict(self.grid_pars)
        data_io.custom_json_dump(result_dir / "parameters.json", sim_pars)
