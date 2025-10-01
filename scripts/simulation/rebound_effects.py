import os
import pathlib
import datetime
from dateutil import parser
import pytz
import pandas as pd
import matplotlib.pyplot as plt

from grecco_sim.analysis import sim_batch_comparison
from grecco_sim.simulator import parallel_sim, simulation_setup
from grecco_sim.util import config, type_defs

OPFINGEN_SCENARIO = {
    "name": "opfingen",
    # Create symlink into data directory!
    "grid_data_path": config.data_path()
    / "Opfingen_Profiles_2023"
    / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
    # "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023pf_all",
    # "data_path": "/home/agross/data/grecco/0_Opfingen_scenario_pv_50_ev_20_hp_10_2023"
    "weather_data_path": config.data_path()
    / "Opfingen_Profiles_2023"
    / "weather_data.csv",
    "hp": False,
    "pv_scale": 0.35
}

# This pypsa folder contains all time steps in which congestion and voltage issues are expected.
# Results come from indigo simulations
PATH_WITH_REDUCED_TS = (
    config.data_path()
    / "Opfingen_Profiles_2023"
    / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all_ts_reduced"
)

# Comment out if you want to analyse periods of highes congestion in trafo.  These cases are nevertheless interesting to analyse
SELECTED_CASES_AND_P_LIM = [
    # (datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.utc), 70),
    # (datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc), 65),
    # (datetime.datetime(2023, 3, 31, 0, 0, 0, tzinfo=pytz.utc), 70),
    (datetime.datetime(2023, 6, 14, 0, 0, 0, tzinfo=pytz.utc), 60),
    # (datetime.datetime(2023, 3, 27, 0, 0, 0, tzinfo=pytz.utc), 65),
    # (datetime.datetime(2023, 4, 5, 0, 0, 0, tzinfo=pytz.utc), 55),
    # (datetime.datetime(2023, 7, 23, 0, 0, 0, tzinfo=pytz.utc), 55),
    # (datetime.datetime(2023, 6, 14, 0, 0, 0, tzinfo=pytz.utc), 60),
    # (datetime.datetime(2023, 6, 15, 0, 0, 0, tzinfo=pytz.utc), 60),
]


def run_congestion_cases(run_name: str):
    """Run sims in parallel for the dates where congestion is likely."""
    # coordination_mechanisms = ["second_order", "none", "local_self_suff", "admm"]
    coordination_mechanisms = ["gradient_descent"]
    # coordination_mechanisms = ["local_self_suff", "none", "plain_grid_fee"]

    base_dir = config.result_dir() / "parallelized" / run_name
    run_par_sets = []
    
    run_pars = []
    opt_pars = []
    grid_pars = []

    for date, p_lim in SELECTED_CASES_AND_P_LIM:
        
        for max_iter in [3, 4, 5, 6, 7, 8, 9, 10]:

            _alpha = 0.25
            dates_run_pars = []

            for cm in coordination_mechanisms:

                sim_tag = f"{run_name}_{cm}_{date.strftime('%m%d')}_{max_iter}"

                run_pars += [
                    type_defs.RunParameters(
                        sim_horizon=96,
                        start_time=date,
                        max_market_iterations=5 if cm in ["admm", "second_order"] else max_iter,
                        coordination_mechanism=cm,
                        scenario=OPFINGEN_SCENARIO,
                        sim_tag=sim_tag,
                        plot=True,
                        show=False,
                        output_file_dir=base_dir / f"{date.strftime('%m%d')}" / sim_tag,
                        profile_run=True,
                    )
                ]

                dates_run_pars += [run_pars[-1]]

                opt_pars += [
                    type_defs.OptParameters(
                        rho=100.0,
                        mu=5000.0,
                        horizon=50,
                        alpha=_alpha,
                        solver_name="gurobi",
                        fc_type="perfect"
                    )
                ]

                grid_pars += [
                    type_defs.GridDescription(
                        p_lim=p_lim,  # Algorithm parameters
                    )
                ]
            run_par_sets += [dates_run_pars]


    # parallel_sim.parallel_sim(base_dir, run_pars, opt_pars, grid_pars, n_workers=12)


    # for run_pars in run_par_sets:
        # sim_batch_comparison.compare(run_pars)

    sim_batch_comparison.compare(run_pars)

    plt.show()


if __name__ == "__main__":
    run_congestion_cases("run_2")
