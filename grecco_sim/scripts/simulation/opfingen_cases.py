import os
import pathlib
import datetime
from dateutil import parser
import pytz
import pandas as pd

from grecco_sim.simulator import parallel_sim
from grecco_sim.util import type_defs

OPFINGEN_SCENARIO = {
    "name": "opfingen",
    # Create symlink into data directory!
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
    # "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023pf_all",
    # "data_path": "/home/agross/data/grecco/0_Opfingen_scenario_pv_50_ev_20_hp_10_2023"
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    / "weather_data.csv",
    "hp": True,
}

# This pypsa folder contains all time steps in which congestion and voltage issues are expected.
# Results come from indigo simulations
PATH_WITH_REDUCED_TS = (
    pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "Opfingen_Profiles_2023"
    / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all_ts_reduced"
)

# Comment out if you want to analyse periods of highes congestion in trafo.  These cases are nevertheless interesting to analyse
SELECTED_CASES = [
    datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.utc),
    datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc),
    datetime.datetime(2023, 7, 1, 0, 0, 0, tzinfo=pytz.utc),
    datetime.datetime(2023, 7, 16, 0, 0, 0, tzinfo=pytz.utc),
]


def select_congestion_cases(number_of_cases: int, season="", path=""):
    """
    Select dates with highest transformer congestions

    Current opfingen profiles show highest congestions in summer due to high PV feed in.
    For checking cases in winter, season="winter"

    number_of_cases will select the n-more congested times (in "winter" for the whole year)
    """

    # Dataframe matching index with dates
    snapshots = pd.read_csv(os.path.join(path, "snapshots.csv"))["snapshot"]

    trafo_load = pd.read_csv(os.path.join(path, "transformers-loading.csv"))
    trafo_load["snapshots"] = snapshots
    trafo_load = trafo_load.iloc[:, 2:]  # Leaving only snapshots and operating transformer

    if season == "winter":
        snapshots_filtered = [
            date for date in snapshots if parser.parse(date).month in [1, 11, 12]
        ]  # November, December, January

        trafo_load = trafo_load[trafo_load["snapshots"].isin(snapshots_filtered)]

    trafo_load_sorted = trafo_load.sort_values(ascending=False, by="1").head(number_of_cases)

    congestion_cases = []
    for i in trafo_load_sorted.index:
        congestion_cases.append(snapshots[i])
    congestion_dates = [
        parser.parse(date).replace(hour=0, minute=0, second=0, microsecond=0).isoformat() for date in congestion_cases
    ]  # Start anaylsis from midnight
    return congestion_dates


def run_sim(run_name: str, congestion_date):

    # coordination_mechanisms = ["second_order", "none", "local_self_suff", "admm"]
    coordination_mechanisms = ["plain_grid_fee"]
    # coordination_mechanisms = ["none", "plain_grid_fee", "local_self_suff", "central_optimization", "second_order"]

    base_dir = pathlib.Path(__file__).parent.parent.parent / "results" / "parallelized" / run_name

    run_pars = []
    opt_pars = []
    grid_pars = []

    for cm in coordination_mechanisms:

        market_iterations = [5] if cm in ["admm", "second_order"] else [1]
        # market_iterations = [1, 2, 3, 4, 5]

        for it in market_iterations:
            sim_tag = f"{run_name}_{cm}"

            run_pars += [
                type_defs.RunParameters(
                    sim_horizon=96,
                    start_time=congestion_date,
                    max_market_iterations=it,
                    coordination_mechanism=cm,
                    scenario=OPFINGEN_SCENARIO,
                    sim_tag=sim_tag,
                    plot=True,
                    show=True,
                    output_file_dir=base_dir / sim_tag,
                    profile_run=True,
                )
            ]

            opt_pars += [
                type_defs.OptParameters(
                    rho=100.0,
                    mu=5000.0,
                    horizon=40,
                    alpha=1.0,
                    solver_name="gurobi",
                )
            ]

            grid_pars += [
                type_defs.GridDescription(
                    p_lim=70,  # Algorithm parameters
                )
            ]

    parallel_sim.parallel_sim(base_dir, run_pars, opt_pars, grid_pars, n_workers=12)


def main(run_name: str):
    if not SELECTED_CASES:
        congestion_cases = select_congestion_cases(5, "", path=PATH_WITH_REDUCED_TS)
        for case, i in zip(congestion_cases, range(1, 1 + len(congestion_cases))):
            run_sim(f"{run_name}_{i}", case)
    else:
        print("simulating selected_dates")
        for case, i in zip(SELECTED_CASES, range(1, 1 + len(SELECTED_CASES))):
            run_sim(f"{run_name}_{i}", case)


if __name__ == "__main__":
    main("congestion")
