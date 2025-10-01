import pathlib
import pickle

from matplotlib import pyplot as plt

from grecco_sim.util import style


def get_objects(pickled_file_name):
    with open(pickled_file_name, "rb") as nfile:
        obj = pickle.load(nfile)
    grid_nodes, coordinator, k = obj
    return grid_nodes, coordinator, k


def market_clearing(pickled_file_name):
    grid_nodes, coordinator, k = get_objects(pickled_file_name)

    # Get initial schedule of connected nodes without any central signal
    initial_futures = {
        node.sys_id: node.get_current_future(k, coordinator.horizon, coordinator.get_initial_signal())
        for node in grid_nodes
    }

    signals = coordinator.get_signals(initial_futures)

    print(f"Constraint violation: {coordinator.constraint_value}")
    viols = [coordinator.constraint_value]

    max_market_iterations = 25

    market_iterations = 1
    while (
        not coordinator.has_converged(k)
        and market_iterations <= max_market_iterations
    ):
        print(market_iterations)
        # Multiprocessing way to get futures is slower
        futures = {
            node.sys_id: node.get_current_future(k, coordinator.horizon, signals[node.sys_id]) for node in grid_nodes
        }

        # This function should update if the market has converged
        signals = coordinator.get_signals(futures)
        # viols += [coordinator.constraint_value]
        market_iterations += 1

    fig, ax = style.styled_plot(title="Constraint Violations", ylabel="Average violation / kW", figsize = "landscape")
    ax.plot(viols)
    plt.show()


if __name__ == "__main__":
    default_path = pathlib.Path("/home/agross/src/python/grecco/grecco_sim/results/default/")
    # market_clearing(default_path / "240906_1544/coordination_state_at_k_36.pkl" )  # central
    # market_clearing(default_path / "240906_1527/coordination_state_at_k_36.pkl" )  # second_order
    # market_clearing(default_path / "241010_1253/coordination_state_at_k_0.pkl")
    market_clearing(default_path / "241030_1639/coordination_state_at_k_40.pkl")
