import json
from pathlib import Path

import pandas as pd

from grecco_sim.simulator.result import SimulationResult
from grecco_sim.util import type_defs, data_io
import pypsa


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

def map_result_to_pypsa(
        network_path: Path,
        sim_result: SimulationResult) -> pypsa.Network:

    """ Extract load data from SimulationResult and map it to pypsa.Network."""

    load_ts = sim_result.ts_grid
    get_bus_name = lambda x: x.split("_load")[0].split("bus_")[1]
    load_ts.rename(axis=1, mapper=get_bus_name)
    network = pypsa.Network(network_path, snapshots=load_ts.index)
    network.buses_t["p_set"] = load_ts
    network.lpf()
    network.pf(use_seed=True)

    pass

    def write_loads(self, state: dict[str, dict], t: int) -> None:
        """ Set grid state from simulation node state. """

        p_set_load = dict()
        p_set_gen = dict()
        p_set_bat = dict()

        for load in self.n.loads.index:
            bus = self.n.loads.loc[load, "bus"]
            data = state[build.sys_id(bus)]

            if "baseload" in load.lower():
                p_set_load[load] = data["baseload_p_model"]

            elif "heat" in load.lower():
                if not self.simulation_config.use_heatpumps:
                    continue
                else:
                    p_set_load[load] = data["hp_p_model"]

            else:
                raise NotImplementedError(f"Unknown load {load}.")

        if self.simulation_config.use_pv:
            for generator in self.n.generators.index:

                # Slack generators are not set as they are dependent variables.
                if self.n.generators.loc[generator, "control"] == "Slack":
                    continue

                bus = self.n.generators.loc[generator, "bus"]
                data = state[build.sys_id(bus)]

                if "pv" in generator.lower():
                    p_set_gen[generator] = data["pv_p_model"]
                else:
                    msg = f"Unknown generator {generator}."
                    raise NotImplementedError(msg)

        if self.simulation_config.use_batteries:
            storage_units = self.n.storage_units.query("type == 'h0_battery'")
            for storage in storage_units.index:
                bus = self.n.storage_units.loc[storage, "bus"]
                data = state[build.sys_id(bus)]
                p_set_bat[storage] = data["bat_p_model"]

        if self.simulation_config.use_ev:
            pass

        # PyPSA snapshots are not localized. PyPSA loads are in MW.
        time_index = [self.time_index[t].tz_localize(None)]
        p_set_load = pd.DataFrame(p_set_load , index=time_index) / 1000
        self.n.loads_t["p_set"].update(p_set_load)

        p_set_gen = pd.DataFrame(p_set_gen, index=time_index) / 1000
        self.n.generators_t["p_set"].update(p_set_gen)

        p_set_bat = pd.DataFrame(p_set_bat, index=time_index) / 1000
        self.n.storage_units_t["p_set"].update(p_set_bat)