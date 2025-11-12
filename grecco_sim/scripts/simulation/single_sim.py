from typing import Optional
import pathlib
import datetime
import pytz
import matplotlib.pyplot as plt

from grecco_sim.simulator import simulation_setup
from grecco_sim.util import config, type_defs, logger


def build_opfingen_scenario(
    data_root: pathlib.Path, year: int, ev_scenario: Optional[str] = ""
) -> dict:
    """Create the OPFINGEN scenario dictionary."""

    return {
        "name": f"opfingen_{year}{ev_scenario}",
        "n_agents": 4,
        "grid_data_path": data_root / f"{year}" / f"{year}{ev_scenario}",
        "weather_data_path": data_root / f"{year}" / "weather_data.csv",
        "ev_capacity_data_path": data_root / "synpro_ev_data_pool.csv",
        "heat_demand_data_path": data_root / f"{year}" / f"{year}{ev_scenario}" / "heat_demand.csv",
        "energy_price_path": data_root/ f"{year}" / "energy_prices_entso.csv",
        "hp": True,
        "ev": True,
        "bat": True,
    }


def run_simulation(
    coord_type: str,
    start_time: datetime.datetime,
    days: int,
    scenario: dict,
    sim_name: Optional[str] = None,
):
    """Set up and run the simulation."""
    date = start_time.date()

    if sim_name is None:
        sim_name = f"{coord_type}_{date}"

    run_parameters = simulation_setup.RunParameters(
        sim_horizon=days * 24 * 4,
        start_time=start_time,
        max_market_iterations=4,
        coordination_mechanism=coord_type,
        scenario=scenario,
        sim_tag=coord_type,
        use_prev_signals=False,
        plot=True,
        show=False,
        profile_run=True,
        output_file_dir=pathlib.Path("default") / sim_name,
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
        p_lim=150.0,
    )

    simulator = simulation_setup.SimulationSetup(run_parameters)
    simulator.run_sim(opt_pars, grid_pars)


if __name__ == "__main__":

    # todo: It is confusing to have a year in "start" and "year" as variable.
    #   I think we should define the variable year via "start.year"
    #   If that is not possible, leave a comment on why.

    data_root = config.data_root()
    year = "full_el"

    # --- Configuration ---
    coordinators = ["none", "plain_grid_fee", "local_self_suff", "central"]
    coordinator_name = coordinators[0]
    n_days = 10

    start = datetime.datetime(year=2023, month=1, day=1, hour=0)

    ev_scenarios = [
        "_evconservative",
        "_evextreme",
        "",
    ]  # no difference for 2024, then either select conservative or extreme
    ev = ev_scenarios[0]

    scenario = build_opfingen_scenario(data_root, year, ev)

    sim_name = f"{coordinator_name}_{year}_hp_test"

    # --- Run Simulation ---
    run_simulation(coordinator_name, start, n_days, scenario, sim_name)
