import pathlib
import datetime
import pytz
import matplotlib.pyplot as plt
import os

from grecco_sim.simulator import simulation_setup
from grecco_sim.util import type_defs
from grecco_sim.analysis import simulation_eval

DUMMY_SCENARIO = {"name": "dummy_data", "n_agents": 4}
SAMPLE_SCENARIO = {"name": "sample_scenario", "focus": "pv_bat", "n_agents": 1}
HP_SCENARIO = {"name": "sample_scenario", "focus": "hp", "n_agents": 1}
OPFINGEN = {
    "name": "opfingen",
    "n_agents": 4,
    #     # Create symlink into data directory!
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    / "weather_data.csv",
    # "ev_capacity_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    # / "data"
    # / "Opfingen_Profiles_2023"
    # / "synpro_ev_data_pool 1.csv", #if file not available, capacities are set to 60kWh
    "hp": True,
    "ev": True,
    "bat": True,
}


def main(coord_type):

    run_parameters = simulation_setup.RunParameters(
        sim_horizon=24 * 4 * 23,
        # Summer: peak feed in
        start_time=datetime.datetime(2023, 4, 1, 6, 0, 0, tzinfo=pytz.utc),
        # start_time = datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc),
        max_market_iterations=4,
        coordination_mechanism=coord_type,
        # scenario=SAMPLE_SCENARIO,
        scenario=OPFINGEN,
        sim_tag=f"{coord_type}",
        # inspection=[36],
        use_prev_signals=False,
        plot=True,
        show=True,
        profile_run=True,
        output_file_dir=pathlib.Path("default")
        / (datetime.datetime.now().strftime("%y%m%d_%H%M") + "_" + coord_type),
        # output_file_dir=pathlib.Path("whole_year")
    )

    opt_pars = type_defs.OptParameters(
        rho=100.0,
        mu=5000.0,
        horizon=13,
        alpha=0.05,
        solver_name="osqp",
        fc_type="perfect",
    )

    grid_pars = type_defs.GridDescription(
        p_lim=75.0,  # Algorithm parameters
    )

    simulator = simulation_setup.SimulationSetup(run_parameters)
    kpis = simulator.run_sim(opt_pars, grid_pars)

    kpis["output_dir"] = run_parameters.output_file_dir

    return kpis


if __name__ == "__main__":
    # compare local self suff and none
    coord_type = "local_self_suff"
    kpis_self_suff = main(coord_type)
    coord_type = "none"
    kpis_none = main(coord_type)

    # Compare the two runs
    simulation_eval.compare_kpis(kpis_self_suff, kpis_none)

    self_suff_folder = kpis_self_suff["output_dir"]
    none_folder = kpis_none["output_dir"]

    eval, flex_bus = simulation_eval.evaluate_flex(self_suff_folder, none_folder)
    print("Buses with flexibility:")
    for bus in flex_bus:
        print(bus)
    for key, value in eval.items():
        if not value["self suff superior"] and not value["indefinite"]:
            print(f"{key} performes worse in self sufficiency than none")
        print(f"{key}: {value}")
