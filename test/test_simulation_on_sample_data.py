import datetime
import pytz
from grecco_sim.simulator import simulation_setup
from grecco_sim.util import type_defs


def test_sim():

    for preset in ["none", "plain_grid_fee"]:
        run_parameters = simulation_setup.RunParameters(
            sim_horizon=40,
            start_time=datetime.datetime(2023, 7, 11, 0, 0, 0, tzinfo=pytz.utc),
            max_market_iterations=20,
            coordination_mechanism=preset,  # out of [central, distributed, none, admm]
            scenario={"name": "sample_scenario", "n_agents": 2, "focus": "pv_bat"},
            # on_off=True,
            # inspection=[60]
            plot=False,
            sim_tag="test_simulation",
            use_prev_signals=False,
        )

        opt_pars = type_defs.OptParameters(
            rho=100.0, mu=5000.0, horizon=5, solver_name="osqp", alpha=0.05
        )

        grid_pars = type_defs.GridDescription(
            p_lim=35.0,  # Algorithm parameters
        )

        simulator = simulation_setup.SimulationSetup(run_parameters)
        simulator.run_sim(opt_pars, grid_pars)


if __name__ == "__main__":
    test_sim()
