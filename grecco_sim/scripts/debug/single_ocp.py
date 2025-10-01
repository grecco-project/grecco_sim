import matplotlib.pyplot as plt

from grecco_sim.local_problems import local_gradient_descent
from grecco_sim.util import data_io, style


def from_pickle(file_name: str):
    forecast, sys_id, system_sim_pars, opt_pars, state, signal = data_io.unpickle(file_name)

    fig, ax = style.styled_plot(figsize=(12, 8))

    for solver_name in ["gurobi", "osqp", "bonmin"]:
        opt_pars.solver_name = solver_name

        solver = local_gradient_descent.get_solver(
            forecast.fc_len, sys_id, system_sim_pars, opt_pars
        )
        solver.solve(state, forecast, signal)

        print(f"{solver_name}: {solver.nlp_solver.opt_objective()}")

        u = solver.get_u()
        ax.plot(u[sys_id + "_hp"], label=solver_name, drawstyle="steps-post")

    fig.legend()
    plt.show()


if __name__ == "__main__":
    from_pickle("hp_problem.pkl")
