import json
from pathlib import Path

import pandas as pd

from grecco_sim.simulator.results import SimulationResult
from grecco_sim.util import type_defs, data_io


def build_result_from_files(result_dir: Path | str) -> SimulationResult:
    result_dir = Path(result_dir)  # Ensure Pathclass

    if not (result_dir / "parameters.json").exists():
        raise FileNotFoundError("Expected parameters.json in result dir.")

    with open(result_dir / "parameters.json") as f:
        parameters = json.load(f)

    run_pars = type_defs.RunParameters(**parameters["run_pars"])
    opt_pars = type_defs.OptParameters(**parameters["opt_pars"])
    grid_pars = type_defs.GridDescription(**parameters["grid_pars"])

    p = result_dir / "system_parameters.json"
    sys_pars = data_io.load_system_parameters(p)

    p = result_dir / f"signals_costs_{run_pars.sim_tag}.csv"
    assigned_grid_fees = data_io.read_ts(p).values
    time_index = data_io.read_ts(p).index

    agent_ts_dict = dict()
    for sys_id in sys_pars:
        p = result_dir / "agents" / f"{sys_id}.csv"
        agent_ts = data_io.read_ts(p).values
        agent_ts_dict[sys_id] = agent_ts

    return SimulationResult(
        run_pars,
        opt_pars,
        grid_pars,
        agent_ts_dict,
        assigned_grid_fees,
        sys_pars,
        time_index)
