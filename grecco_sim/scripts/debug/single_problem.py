import matplotlib.pyplot as plt
import numpy as np

from acados_template import latexify_plot

from grecco_sim.analysis import schedule_eval
from grecco_sim.local_control import problem_combiner
from grecco_sim.local_problems import local_admm
from grecco_sim.local_problems import local_solver_pv_bat
from grecco_sim.coordinators import coord_admm
from grecco_sim.util import style
from grecco_sim.util import logger

logger.set_logger()
logger.set_pandas_print()
latexify_plot()


def plot(
    parameters,
    lam,
    grid_combined,
    y_ks,
    u_ks,
    res_power_set=None,
    grid_plus_slack=None,
    lam_slack=None,
    show=True,
):

    # fig, ax = style.styled_plot(figsize="landscape", ylabel="Power", xlabel="Time Steps")
    fig, ax = plt.subplots(4, 1, sharex=True)
    # ax[2].plot(times, rho_all, label="rho", drawstyle="steps-post")
    if lam is not None:
        for a in lam:
            ax[2].plot(parameters.times, lam[a], label=f"lam[{a}]", drawstyle="steps-post")

    if lam_slack is not None:
        ax[2].plot(parameters.times, lam_slack, label="slack", drawstyle="steps-post")

    ax[3].plot(parameters.times, grid_combined, label="grid", drawstyle="steps-post")
    if grid_plus_slack is not None:
        ax[3].plot(
            parameters.times,
            grid_plus_slack,
            label="grid+slack (where != limit)",
            drawstyle="steps-post",
        )
    ax[3].plot(
        parameters.times,
        [parameters.p_lim] * len(parameters.times),
        label="Limit",
        drawstyle="steps-post",
    )
    ax[3].fill_between(
        parameters.times,
        [parameters.p_lim] * len(parameters.times),
        grid_combined,
        where=grid_combined > parameters.p_lim,
        step="post",
        alpha=0.2,
        color="red",
        label="Constraint",
    )

    for a in y_ks:
        ax[a].plot(
            parameters.times,
            y_ks[a],
            linestyle="dotted",
            drawstyle="steps-post",
            label=f"Grid " r"($x^s_a - x^f_{a}$)",
        )
        fac_bat_pow = parameters.sim_pars[a]["p_inv"] if parameters.on_off[a] else 1.0
        ax[a].plot(
            parameters.times,
            fac_bat_pow * u_ks[a],
            linestyle="dashed",
            drawstyle="steps-post",
            label=f"Control {a}",
        )
        if res_power_set is not None:
            ax[a].plot(
                parameters.times,
                res_power_set[a],
                linestyle="dashdot",
                drawstyle="steps-post",
                label="set",
            )

    legend_title = ["Agent 0", "Agent 1", None, None]
    for i in range(4):
        ax[i].legend(title=legend_title[i], loc="right")
    fig.tight_layout()

    if show:
        plt.show()


class Parameters:
    horizon = 12
    n_constr = horizon
    times = list(range(horizon))
    n_agents = 2
    p_lim = 15.0

    ev_connected = [1] * horizon
    soc_init = 0.0
    soc_target = 0.8
    # fc_res_load = [[x * 0.1 + 1. for x in times]] * n_agents
    # fc_res_load = [[1. for _ in times]] * n_agents
    fc_res_load = [[1.2] * 5 + [9, 9, 9, 9] + [0] * 3] * n_agents

    sim_pars = [
        dict(
            c_sup=[0.2 + 0.0005 * k for k in times],
            c_feed=0.0,
            capacity=30.0,
            dt_h=1.0,
            p_inv=5.0,
            name="ag0",
        ),  # Agent 0
        dict(
            c_sup=[0.41 + 0.0005 * k for k in times],
            c_feed=0.0,
            capacity=30.0,
            dt_h=1.0,
            p_inv=5.0,
            name="ag1",
        ),  # Agent 1
    ]
    on_off = [False, False]

    # ADMM parameters
    rho = 1.0

    max_iter = 200
    # 1 for no plots
    plot_every = 199


def centralized(parameters: Parameters):

    combiner = problem_combiner.ProblemCombiner(parameters.horizon)
    agents = {}
    for i in range(parameters.n_agents):
        _pars = parameters.sim_pars[i]
        if parameters.on_off[i]:
            agents[f"ag_{i:02d}"] = local_solver_pv_bat.OnOffPVBatSolver(
                parameters.horizon, f"ag_{i:02d}", _pars
            )
        else:
            agents[f"ag_{i:02d}"] = local_solver_pv_bat.CasadiSolverLocalPVBat(
                parameters.horizon, f"ag_{i:02d}", _pars, {"solver": "gurobi"}
            )

    combiner.combine(agents, parameters.p_lim)

    par_values = {
        f"ag_{i:02d}": {
            "x_init": [parameters.soc_init],
            "lam_a": [0.0] * parameters.horizon,
            "fc_p_unc": parameters.fc_res_load[i],
        }
        for i in range(parameters.n_agents)
    }
    res = combiner.solve(par_values)
    print(res)

    grid_combined = np.zeros(parameters.horizon)
    for a in res:
        print(f"Ag {a}: {np.sum(res[a]['yk'])}")
        grid_combined += res[a]["yk"]

    y_ks = {int(a[3:]): res[a]["yk"] for a in res}
    plot(parameters, None, grid_combined, y_ks, {int(a[3:]): res[a]["uk"] for a in res}, show=False)

    evaluator = schedule_eval.ScheduleEvaluator(y_ks, parameters.sim_pars)
    costs = evaluator.get_costs()
    constr = evaluator.get_constr_viol(parameters.p_lim)

    fig, ax = style.styled_plot(title="Centralized", figsize="landscape")
    ax.plot([0, 1], [costs, costs], label="costs centralized")
    ax.plot([0, 1], [constr, constr], label="summed constraint violations")
    ax.legend()
    fig.tight_layout()


def vujanic(parameters: Parameters):

    s_up = [np.array([-100000.0] * parameters.n_constr)] * parameters.n_agents
    s_lo = [np.array([100000] * parameters.n_constr)] * parameters.n_agents
    rho = [np.zeros(parameters.n_constr)] * parameters.n_agents
    # grid_combined = np.zeros(horizon)
    lam = np.zeros(parameters.horizon)

    y_ks = {a: np.zeros(parameters.horizon) for a in range(parameters.n_agents)}
    u_ks = {a: np.zeros(parameters.horizon) for a in range(parameters.n_agents)}

    viols = []

    lams = [[], []]
    lam_diffs = []
    costs = []

    for iteration in range(parameters.max_iter):
        print(f"===============Iteration {iteration} ==========")

        for a in range(parameters.n_agents):
            # for a in [iteration % n_agents]:
            # agent = local_solver_ev.CasadiSolverLocalEV(horizon)
            # # Idea: add this to lambda for agents to maybe divert the agents away from each other
            # # This lead to the agents being away from each other sometimes.
            # sigma = 10.
            # lambda_noise = np.random.normal(0., sigma, lam.shape)
            #
            # uk, yk = agent.solve(soc_init, soc_target, fc_res_load[a], {"lambda": lam}, ev_connected)
            _pars = parameters.sim_pars[a]
            if parameters.on_off[a]:
                agent = local_solver_pv_bat.OnOffPVBatSolver(
                    parameters.horizon, f"ag_{a:02d}", _pars
                )
            else:
                agent = local_solver_pv_bat.CasadiSolverLocalPVBat(
                    parameters.horizon, f"ag_{a:02d}", _pars, {"solver": "gurobi"}
                )
            uk, yk = agent.solve(parameters.soc_init, parameters.fc_res_load[a], {"lambda": lam})
            print(f"Updating agent {a} grid sup: {np.sum(yk)}")
            y_ks[a] = yk
            u_ks[a] = uk

            s_up[a] = np.maximum(s_up[a], yk)
            s_lo[a] = np.minimum(s_lo[a], yk)

            print(f"yk: {yk}  | sup: {s_up[a]} | s_lo: {s_lo[a]}")

            rho[a] = s_up[a] - s_lo[a]
            print(rho)

        rho_all = parameters.n_constr * np.array(rho).max(axis=0)
        print(f"rho all: {rho_all}")

        grid_combined = np.sum(np.array([np.array(y_ks[a]) for a in y_ks]), axis=0)
        lams[0] += [lam[1]]
        lams[1] += [lam[6]]
        lam_diffs += [lam[6] - lam[1]]

        # print(constr_violation)
        evaluator = schedule_eval.ScheduleEvaluator(y_ks, parameters.sim_pars)
        costs += [evaluator.get_costs()]
        viols += [evaluator.get_constr_viol(parameters.p_lim)]

        def alpha(i):
            return 0.01 / (i + 3.0)

        # lam = np.maximum(0., lam + alpha(iteration) * (grid_combined - parameters.p_lim + rho_all))
        lam = np.maximum(0.0, lam + alpha(iteration) * (grid_combined - parameters.p_lim))
        # lam = np.maximum(lam + grid_combined - p_lim, 0.)

        # def feed(yk, func):
        #     return [func(0, y) for y in yk]

        # feed_rew = np.sum((lam + 0.1) * np.array([feed(y_ks[a], min) for a in y_ks]))
        # supp_cost = np.sum((lam + 0.2) * np.array([feed(y_ks[a], max) for a in y_ks]))

        # lagr = - lam.sum() * parameters.p_lim + feed_rew + supp_cost
        # print(lagr)
        # constr_violation = np.where(grid_combined > p_lim)[0].sum()

        if iteration == parameters.max_iter - 1:
            plot(
                parameters,
                {i: lam for i in range(parameters.n_agents)},
                grid_combined,
                y_ks,
                u_ks,
                show=False,
            )

    fig, ax = style.styled_plot(
        ylabel="Summed Constraint violations", xlabel="Iterations", figsize="landscape"
    )
    ax.plot(viols, drawstyle="steps-post")
    fig.tight_layout()
    fig, ax = style.styled_plot(ylabel="lambda", xlabel="Iterations", figsize="landscape")
    ax.plot(lams[0], label="Early", drawstyle="steps-post")
    ax.plot(lams[1], label="Later", drawstyle="steps-post")
    ax.plot(lam_diffs, label="Diff", drawstyle="steps-post", linestyle="dashed")
    fig.tight_layout()
    ax.legend()
    # convergence behavior of costs
    fig, ax = style.styled_plot(ylabel="costs", xlabel="Iterations", figsize="landscape")
    ax.plot(costs, label="costs", drawstyle="steps-post")
    fig.tight_layout()
    ax.legend()


def single_admm(parameters: Parameters):

    res_power_set = {a: np.zeros(parameters.horizon) for a in range(parameters.n_agents)}
    lam = {a: np.zeros(parameters.horizon) for a in range(parameters.n_agents)}

    y_ks = {a: np.zeros(parameters.horizon) for a in range(parameters.n_agents)}
    # The u_ks are not really necessary for the algorithm
    u_ks = {a: np.zeros(parameters.horizon) for a in range(parameters.n_agents)}

    lam_slack = np.zeros(parameters.horizon)
    ref_power_grid_slack = np.zeros(parameters.horizon)

    viols = []
    costs = []

    g_fig, g_ax = style.styled_plot(
        title="ADMM schedule evolution", xlabel="time", ylabel="power", figsize="landscape"
    )

    for iteration in range(parameters.max_iter):

        for a in range(parameters.n_agents):
            # agent = local_solver_pv_bat.CasadiSolverLocalPVBat(horizon, f"ag_{a:02d}")
            # uk, yk = agent.solve(soc_init, fc_res_load[a], {"lambda": lam})

            signal = {"lambda": lam[a], "res_power_set": res_power_set[a]}

            _pars = parameters.sim_pars[a]
            _pars["on_off"] = parameters.on_off[a]
            agent = local_admm.LocalADMMSolver(
                parameters.horizon, a, _pars, {"rho": parameters.rho, "solver": "gurobi"}
            )

            uk, yk = agent.solve(parameters.soc_init, parameters.fc_res_load[a], signal)
            print(f"Updating agent {a}")

            y_ks[a] = yk
            u_ks[a] = uk

            lam[a] = lam[a] + parameters.rho * (y_ks[a] - res_power_set[a])

        sys_ids = [i for i in range(parameters.n_agents)]
        coord = coord_admm.ADMMCentralSolver(
            parameters.horizon, sys_ids, parameters.rho, parameters.p_lim
        )
        # For the slack agent
        yk_slack = coord.handle_slack_local_problem(
            {"lambda": lam_slack, "res_power_set": ref_power_grid_slack}
        )
        lam_slack = lam_slack + parameters.rho * (yk_slack - ref_power_grid_slack)

        grid_combined = np.sum(np.array([np.array(y_ks[a]) for a in y_ks]), axis=0)
        # lam = np.maximum(0., lam + 1. / (iteration + 1) * (grid_combined - parameters.p_lim))
        res_power_set, ref_power_grid_slack = coord.solve(lam, y_ks, lam_slack, yk_slack)
        grid_plus_slack = grid_combined + yk_slack

        evaluator = schedule_eval.ScheduleEvaluator(y_ks, parameters.sim_pars)
        costs += [evaluator.get_costs()]
        viols += [evaluator.get_constr_viol(parameters.p_lim)]

        # if iteration % 50 in [10, 15, 20, 40]:
        # if iteration % parameters.plot_every == 1:
        if iteration == parameters.max_iter - 1:
            plot(
                parameters,
                lam,
                grid_combined,
                y_ks,
                u_ks,
                res_power_set=res_power_set,
                grid_plus_slack=grid_plus_slack,
                lam_slack=lam_slack,
                show=False,
            )
            g_ax.plot(
                parameters.times,
                y_ks[0],
                label=f"Iteration {iteration}, agent 0",
                drawstyle="steps-post",
            )
            g_ax.plot(
                parameters.times,
                y_ks[1] + iteration * 0.1 / 400.0,
                label=f"Iteration {iteration}, agent 1",
                linestyle="dashed",
                drawstyle="steps-post",
            )

    g_ax.legend()
    g_fig.tight_layout()
    # convergence behavior of costs
    fig, ax = style.styled_plot(ylabel="costs", xlabel="Iterations", figsize="landscape")
    ax.plot(costs, label="costs", drawstyle="steps-post")
    fig.tight_layout()
    ax.legend()

    fig, ax = style.styled_plot(
        ylabel="Summed Constraint violations", xlabel="Iterations", figsize="landscape"
    )
    ax.plot(viols)


if __name__ == "__main__":
    # vujanic(Parameters())
    centralized(Parameters())
    single_admm(Parameters())
    plt.show()
