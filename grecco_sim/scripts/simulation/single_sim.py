import pathlib
import datetime
import pytz
import matplotlib.pyplot as plt

from grecco_sim.simulator import simulation_setup
from grecco_sim.util import type_defs, logger

YEAR = "2028"
OPFINGEN = {
    "name": f"opfingen_{YEAR}",
    "n_agents": 4,
    # Create symlink into data directory!
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / YEAR
    / f"{YEAR}_ev_conservative",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / YEAR
    / "weather_data.csv",
    "heat_demand_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / YEAR
    / YEAR
    / "heat_demand.csv",
    "energy_price_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / YEAR
    / "energy_prices_entso.csv",
    "hp": True,
    "ev": True,
    "bat": True,
}


def main(coord_type: str, start_time: datetime.datetime, days: int = 7, name: str = ""):

    date = start_time.date()
    if not name:
        name = f"{coord_type}_{date}"
    run_parameters = simulation_setup.RunParameters(
        sim_horizon=24*4*days,  # 24 * 4 * days, this is in intervals i.e. 15 min intervals
        # start_time=datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=pytz.utc),
        start_time=start_time,
        max_market_iterations=4,
        coordination_mechanism=coord_type,
        # scenario=SAMPLE_SCENARIO,
        scenario=OPFINGEN,
        sim_tag=f"{coord_type}_{date}",
        # inspection=[36],
        use_prev_signals=False,
        plot=True,
        show=False,
        profile_run=True,
        output_file_dir=pathlib.Path("default") / name,
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

    coord = "plain_grid_fee"  # "none", "plain_grid_fee", "local_self_suff", "central_optimization", "second_order", "admm"
    start = datetime.datetime(2019, 1, 1, 0, 0, 0, tzinfo=pytz.utc) # (yyyy, m, d, h, m, s) time resolution only 15 mins
    days = 3

    main(coord, start, days, name="test_28")
