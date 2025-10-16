import pathlib
import datetime
import zoneinfo as zi
import matplotlib.pyplot as plt

from grecco_sim.simulator import simulation_setup
from grecco_sim.util import type_defs, logger

YEAR = 2028
OPFINGEN = {
    "name": f"opfingen_{YEAR}_c",
    "n_agents": 4,
    # Create symlink into data directory!
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / f"{YEAR}"
    / f"{YEAR}_evconservative",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / f"{YEAR}"
    / "weather_data.csv",
    "ev_capacity_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "synpro_ev_data_pool.csv",  # if file not available, capacities are set to 60kWh
    "heat_demand_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / f"{YEAR}"
    / f"{YEAR}_evconservative"
    / "heat_demand.csv",
    "hp": True,
    "ev": True,
    "bat": True,
}


def main(coord_type: str, start_time: datetime.datetime, days: int = 7, name: str = ""):

    date = start_time.date()
    if not name:
        name = f"{coord_type}_{date}"
    run_parameters = simulation_setup.RunParameters(
        sim_horizon=days * 24 * 4,
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

    coord = "none"  # "none", "plain_grid_fee", "local_self_suff", "central_optimization", "second_order", "admm"
    start = datetime.datetime(2019, 1, 1, 0, 0, 0, tzinfo=zi.ZoneInfo("Europe/Berlin"))
    days = 10

    main(coord, start, days, name=f"{coord}_{YEAR}_hp_test")
