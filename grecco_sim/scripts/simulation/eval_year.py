import pathlib
import datetime
import pytz
import matplotlib.pyplot as plt

from grecco_sim.simulator import simulation_setup
from grecco_sim.util import type_defs

DUMMY_SCENARIO = {"name": "dummy_data", "n_agents": 4}
SAMPLE_SCENARIO = {"name": "sample_scenario", "focus": "pv_bat", "n_agents": 1}
HP_SCENARIO = {"name": "sample_scenario", "focus": "hp", "n_agents": 1}
OPFINGEN = {
    "name": "opfingen",
    "n_agents": 4,
    #     # Create symlink into data directory!
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "2024"
    / "2024_pf_all",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "2024"
    / "weather_data.csv",
    "ev_capacity_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "synpro_ev_data_pool.csv",  # if file not available, capacities are set to 60kWh
    "heat_demand_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "2024"
    / "2024"
    / "heat_demand.csv",
    "hp": True,
    "ev": True,
    "bat": True,
}


def main(coord_type: str, start_time: datetime.datetime, days: int = 7):

    # coord_type = "central_optimization"
    # coord_type = "second_order"
    # coord_type = "admm"
    # coord_type = "plain_grid_fee"
    # coord_type = "local_self_suff"
    # coord_type = "none"

    run_parameters = simulation_setup.RunParameters(
        sim_horizon=24 * 4 * days,
        # start_time=datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=pytz.utc),
        start_time=start_time,
        max_market_iterations=4,
        coordination_mechanism=coord_type,
        # scenario=SAMPLE_SCENARIO,
        scenario=OPFINGEN,
        sim_tag=f"{coord_type}",
        # inspection=[36],
        use_prev_signals=False,
        plot=True,
        show=False,
        profile_run=True,
        output_file_dir=pathlib.Path("default") / f"{coord_type}_{start_time.strftime('%y%m%d')}",
        # output_file_dir=pathlib.Path("whole_year")
    )

    opt_pars = type_defs.OptParameters(
        rho=100.0,
        mu=5000.0,
        horizon=12,
        alpha=0.05,
        solver_name="gurobi",
        fc_type="perfect",
    )

    grid_pars = type_defs.GridDescription(
        p_lim=150.0,  # Algorithm parameters
    )

    simulator = simulation_setup.SimulationSetup(run_parameters)
    simulator.run_sim(opt_pars, grid_pars)


if __name__ == "__main__":
    # weekly loop over the year
    time_weekly = {}
    for week in range(0, 51):
        start = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=pytz.utc) + datetime.timedelta(
            weeks=week
        )

        for coord in ["plain_grid_fee", "local_self_suff", "none"]:
            print("starting simulation for ", coord, " in week ", week)
            time_weekly[coord + f"_{week}"] = datetime.datetime.now().strftime("%y%m%d_%H%M")
            main(coord, start, days=365)

    time_year = {}
    for coord in ["plain_grid_fee", "local_self_suff", "none"]:
        print("starting year long simulation for ", coord)
        time_year[coord] = datetime.datetime.now().strftime("%y%m%d_%H%M")
        main(coord, datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=pytz.utc), days=365)

    print("Weekly times:")
    print(time_weekly)
    print("Full year times:")
    print(time_year)
