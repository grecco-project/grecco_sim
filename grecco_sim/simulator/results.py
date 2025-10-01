"""Module for definitions of simulation result and ways to save it."""

import dataclasses
import json
import os
import pathlib
import pandas as pd

from grecco_sim.util import type_defs
from grecco_sim.util import data_io


class SimulationResult(object):

    def __init__(self) -> None:
        # Initialize empty simulation result

        # Parameters of run
        self._run_params: type_defs.RunParameters | None = None
        self._opt_pars: type_defs.OptParameters | None = None
        self._grid_pars: type_defs.GridDescription | None = None

        # Parameterization
        self._sizing: dict[str, type_defs.SysPars] | None = None

        # grid time series
        self._assigned_grid_fees: pd.DataFrame | None = None

        self._exec_time: float | None = None

        # Calculated
        self._ts_grid: pd.DataFrame | None = None
        self._agents_ts: dict[str, pd.DataFrame] | None = None

    def _check_init(self):
        """Check if object has been initialized."""
        assert self._run_params is not None

    # =========================== Properties to access result data ==================
    @property
    def run_params(self) -> type_defs.RunParameters:
        self._check_init()
        return self._run_params

    @property
    def opt_pars(self) -> type_defs.OptParameters:
        self._check_init()
        return self._opt_pars

    @property
    def grid_pars(self) -> type_defs.GridDescription:
        self._check_init()
        return self._grid_pars

    @property
    def sizing(self) -> dict[str, type_defs.SysPars]:
        self._check_init()
        return self._sizing

    @property
    def assigned_grid_fees(self) -> pd.DataFrame:
        self._check_init()
        return self._assigned_grid_fees

    @property
    def ts_grid(self) -> pd.DataFrame:
        self._check_init()
        return self._ts_grid

    @property
    def agents_ts(self) -> dict[str, pd.DataFrame]:
        self._check_init()
        return self._agents_ts

    @property
    def sys_ids(self) -> list[str]:
        self._check_init()
        return list(self.agents_ts.keys())

    @property
    def flex_ts(self) -> pd.DataFrame:
        self._check_init()
        data = {}
        for sys_id, df_ag in self.agents_ts.items():
            if "bat_p_ac" in df_ag:
                data[f"{sys_id}_bat_p_ac"] = df_ag["bat_p_ac"]
            if "hp_p_in" in df_ag:
                data[f"{sys_id}_hp_p_in"] = df_ag["hp_p_in"]
            if "ev_soc" in df_ag:
                data[f"{sys_id}_ev_p_ac"] = df_ag["ev_p_ac"]
                data[f"{sys_id}_ev_soc"] = df_ag["ev_soc"]

        return pd.DataFrame(data=data)

    @property
    def exec_time(self) -> float:
        self._check_init()
        return self._exec_time

    @property
    def trafo_sum(self) -> pd.Series:
        """Return time series of summed up power at transformer."""
        self._check_init()
        return self._ts_grid.sum(axis=1)

    def from_simulation(
        self,
        run_params,
        opt_pars,
        grid_pars,
        raw_output,
        assigned_grid_fees,
        sizing,
        time_index,
        exec_time,
    ):
        """Initialize directly from simulator."""
        self._run_params = run_params
        self._opt_pars = opt_pars
        self._grid_pars = grid_pars

        self._assigned_grid_fees = assigned_grid_fees
        self._sizing = sizing

        self._ts_grid = pd.DataFrame(
            index=time_index,
            data={sys_id: raw_output[sys_id]["grid"] for sys_id in raw_output},
        )

        self._agents_ts = {
            sys_id: generate_agent_ts(raw_output[sys_id], time_index) for sys_id in raw_output
        }

        self._exec_time = exec_time

        return self

    def from_files(self, directory_name: pathlib.Path | str):

        if not isinstance(directory_name, pathlib.Path):
            directory_name = pathlib.Path(directory_name)

        with open(directory_name / "parameters.json") as all_params_file:
            _all_params = json.load(all_params_file)
            self._run_params = type_defs.RunParameters(**_all_params["run_pars"])
            self._opt_pars = type_defs.OptParameters(**_all_params["opt_pars"])
            self._grid_pars = type_defs.GridDescription(**_all_params["grid_pars"])

        self._sizing = data_io.load_sizing(directory_name / "system_parameters.json")

        self._assigned_grid_fees = data_io.read_ts(
            directory_name / f"signals_costs_{self.run_params.sim_tag}.csv"
        )

        self._ts_grid = data_io.read_ts(
            directory_name / f"nodal_load_{self.run_params.sim_tag}.csv"
        )

        self._agents_ts = {
            sys_id: data_io.read_ts(directory_name / "agents" / f"{sys_id}.csv")
            for sys_id in self._sizing
        }

        self._exec_time = -1.0

    def record(self):

        # Abbreviate
        out_dir = self.run_params.output_file_dir
        sim_tag = self.run_params.sim_tag

        # check that output directory extists
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir / "agents", exist_ok=True)

        # Write grid time series
        data_io.write_ts(self.ts_grid, out_dir / f"nodal_load_{sim_tag}.csv")

        # Write signal time series
        data_io.write_ts(self.assigned_grid_fees, out_dir / f"signals_costs_{sim_tag}.csv")

        # Write flexibility power profiles
        data_io.write_ts(self.flex_ts, out_dir / f"flex_power_{sim_tag}.csv")

        for sys_id, agent_df in self.agents_ts.items():
            data_io.write_ts(agent_df, out_dir / "agents" / f"{sys_id}.csv")

        # Dump parameterization
        data_io.dump_parameterization(out_dir / "system_parameters.json", self.sizing)
        all_pars = {
            "run_pars": dataclasses.asdict(self.run_params),
            "opt_pars": dataclasses.asdict(self.opt_pars),
            "grid_pars": dataclasses.asdict(self.grid_pars),
        }
        data_io.dump_parameterization(out_dir / "parameters.json", all_pars)


def generate_agent_ts(agent_output, time_index):
    """Take raw agents results and generate dataframe fro each agent."""

    individual_dfs = [
        pd.DataFrame(data={key: agent_output[key] for key in ["grid", "c_sup", "c_feed"]})
    ]

    individual_dfs += [pd.DataFrame(data={"p_el_load": agent_output["load"]["p_load"]})]

    if "pv" in agent_output:
        individual_dfs += [pd.DataFrame(data={"p_el_pv": agent_output["pv"]["p_ac"]})]

    if "bat" in agent_output:
        individual_dfs += [
            pd.DataFrame(data={f"bat_{key}": agent_output["bat"][key][:-1] for key in ["soc"]}),
            pd.DataFrame(
                data={f"bat_{key}": agent_output["bat"][key] for key in ["p_ac", "p_net"]}
            ),
        ]

    if "hp" in agent_output:
        individual_dfs += [
            pd.DataFrame(
                data={f"hp_{key}": agent_output["hp"][key][:-1] for key in ["temp", "losses"]}
            ),
            pd.DataFrame(
                data={f"hp_{key}": agent_output["hp"][key] for key in ["p_in", "q_hp", "u_ext"]}
            ),
        ]

    if "ev" in agent_output:
        individual_dfs += [
            pd.DataFrame(data={f"ev_{key}": agent_output["ev"][key][:-1] for key in ["soc"]}),
            pd.DataFrame(data={f"ev_{key}": agent_output["ev"][key] for key in ["p_ac", "p_net"]}),
        ]

    df = pd.concat(individual_dfs, axis=1)
    return df.set_index(time_index)
