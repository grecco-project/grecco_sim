import pathlib
import datetime
import pytz

from grecco_sim.simulator import parallel_sim
from grecco_sim.util import type_defs

OPFINGEN_SCENARIO = {
    "name": "opfingen",
    "n_agents": 4,
    # Create symlink into data directory!
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    /
    # "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023pf_all_ts_reduced"
    "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023pf_all",
    # "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
    # "data_path": "/home/agross/data/grecco/0_Opfingen_scenario_pv_50_ev_20_hp_10_2023"
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    / "weather_data.csv",
}


def main(run_name: str):

    # coordination_mechanisms = ["second_order", "none", "local_self_suff", "admm"]
    # coordination_mechanisms = ["plain_grid_fee", "local_self_suff"]
    coordination_mechanisms = ["plain_grid_fee", "local_self_suff", "central_optimization", "second_order", "admm"]

    base_dir = (
        pathlib.Path(__file__).parent.parent.parent
        / "results"
        / "parallelized"
        / run_name
    )

    run_pars = []
    opt_pars = []
    grid_pars = []

    for cm in coordination_mechanisms:

        market_iterations = 5 if cm in ["admm", "second_order"] else 1
        # market_iterations = [1, 2, 3, 4, 5]
        
        for fc_type in ["perfect"]:
            sim_tag = f"{cm}_fc_{fc_type}"

            run_pars += [
                type_defs.RunParameters(
                    sim_horizon=96,
                    start_time = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.utc),
                    max_market_iterations=market_iterations,
                    coordination_mechanism=cm,
                    scenario=OPFINGEN_SCENARIO,
                    sim_tag=sim_tag,
                    plot=True,
                    show=False,
                    output_file_dir=base_dir / sim_tag,
                    profile_run=True
                )
            ]

            opt_pars += [
                type_defs.OptParameters(
                    rho=100.0,
                    mu=5000.0,
                    horizon=40,
                    alpha=1.0,
                    solver_name="osqp",
                    fc_type=fc_type
                )
            ]

            grid_pars += [
                type_defs.GridDescription(
                    p_lim=120.0,  # Algorithm parameters
                )
            ]

    parallel_sim.parallel_sim(base_dir, run_pars, opt_pars, grid_pars)


if __name__ == "__main__":
    main("res_example")
