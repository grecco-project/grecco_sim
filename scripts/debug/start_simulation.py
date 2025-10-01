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
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent.parent / "data" / "2024" / "2024",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "2024"
    / "2024",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "2024"
    / "weather_data.csv",
    "ev_capacity_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "synpro_ev_data_pool.csv",  # if file not available, capacities are set to 60kWh
    "hp": True,
    "ev": True,
    "bat": True,
}

OPFINGEN_20245_CONSERVATIVE = {
    "name": "opfingen_conservative",
    "n_agents": 4,
    #     # Create symlink into data directory!
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "opfingen_2045"
    / "2045_evconservative",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "opfingen_2045"
    / "weather_data.csv",
    # "ev_capacity_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    # / "data"
    # / "synpro_ev_data_pool.csv",  # if file not available, capacities are set to 60kWh
    # "heat_demand_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    # / "data"
    # / "2024"
    # / "2024"
    # / "heat_demand.csv",
    "hp": True,
    "ev": True,
    "bat": True,
}
OPFINGEN_20245_EXTREME = {
    "name": "opfingen_extreme",
    "n_agents": 4,
    "grid_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "opfingen_2045"
    / "2045_evconservative",
    "weather_data_path": pathlib.Path(__file__).parent.parent.parent.parent
    / "data"
    / "opfingen_2045"
    / "weather_data.csv",
    "hp": True,
    "ev": True,
    "bat": True,
}


def main():

    # coord_type = "central_optimization"
    # coord_type = "admm"
    coord_type = "plain_grid_fee"
    # coord_type = "none"
    # coord_type = "second_order"

    run_parameters = simulation_setup.RunParameters(
        sim_horizon=24 * 8,
        # Summer: peak feed in
        start_time=datetime.datetime(2022, 3, 11, 20, 0, 0, tzinfo=pytz.utc),
        # start_time = datetime.datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc),
        max_market_iterations=4,
        coordination_mechanism=coord_type,
        # scenario=SAMPLE_SCENARIO,
        scenario=OPFINGEN_20245_EXTREME,
        sim_tag=f"{coord_type}",
        # inspection=[36],
        use_prev_signals=False,
        plot=True,
        show=True,
        profile_run=True,
        output_file_dir=pathlib.Path("default") / datetime.datetime.now().strftime("%y%m%d_%H%M"),
        # output_file_dir=pathlib.Path("whole_year")
    )

    opt_pars = type_defs.OptParameters(
        rho=100.0,
        mu=5000.0,
        horizon=50,
        alpha=0.05,
        solver_name="osqp",
        fc_type="perfect",
        # buffer_n_solvers=10,
    )

    grid_pars = type_defs.GridDescription(
        p_lim=65.0,  # Algorithm parameters
    )

    simulator = simulation_setup.SimulationSetup(run_parameters)
    simulator.run_sim(opt_pars, grid_pars)

    plt.show()


if __name__ == "__main__":
    main()
