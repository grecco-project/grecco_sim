from ast import Dict
import json
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

# from acados_template import latexify_plot

from grecco_sim.analysis import schedule_eval
from grecco_sim.local_control import problem_combiner
from grecco_sim.local_problems import local_admm
from grecco_sim.local_problems import local_solver_pv_bat
from grecco_sim.util import type_defs
from grecco_sim.util import style
from grecco_sim.util import logger
from grecco_sim.coordinators import coord_admm
from grecco_sim.coordinators import coord_second_order


logger.set_logger()
logger.set_pandas_print()
# latexify_plot()


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
    title: str = None,
):

    # fig, ax = style.styled_plot(figsize="landscape", ylabel="Power", xlabel="Time Steps")
    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.suptitle(title)
    # ax[2].plot(times, rho_all, label="rho", drawstyle="steps-post")
    if lam is not None:
        for a in lam:
            ax[2].plot(parameters.times, lam[a], drawstyle="steps-post")  # label=f"lam[{a}]",

    if lam_slack is not None:
        ax[2].plot(parameters.times, lam_slack, label="lam_slack", drawstyle="steps-post")

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
    ax[3].set_ylim(0, 1.3 * parameters.p_lim)
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

    for a, sys_id in enumerate(y_ks):
        if a > 1:
            break
        ax[a].plot(
            parameters.times,
            y_ks[sys_id],
            linestyle="dotted",
            drawstyle="steps-post",
            label=f"Grid " r"($x^s_a - x^f_{a}$)",
        )
        fac_bat_pow = parameters.sim_pars[sys_id]["p_inv"] if parameters.on_off[sys_id] else 1.0
        ax[a].plot(
            parameters.times,
            fac_bat_pow * u_ks[sys_id],
            linestyle="dashed",
            drawstyle="steps-post",
            label=f"Control {sys_id}",
        )
        if res_power_set is not None:
            ax[a].plot(
                parameters.times,
                res_power_set[sys_id],
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


def plot_meta(c_admm, v_admm, c_vuja, v_vuja, c_cent, v_cent, c_sec, v_sec, title=None):

    fig, ax = plt.subplots(2, 1, sharex=True)
    if title is not None:
        fig.suptitle(title)
    ax[0].set_ylabel("Summed Constraint violations")

    # ax[0].plot(v_vuja, label="Vujanic", drawstyle="steps-post")
    ax[0].plot(v_admm, label="ADMM", drawstyle="steps-post")
    ax[0].plot(v_cent, label="Central", drawstyle="steps-post")
    ax[0].plot(v_sec, label="Second Order", drawstyle="steps-post")
    ax[0].legend()

    ax[1].set_ylabel("Costs")
    ax[1].set_xlabel("Iterations")

    # ax[1].plot(c_vuja, label="Vujanic", drawstyle="steps-post")
    ax[1].plot(c_admm, label="ADMM", drawstyle="steps-post")
    ax[1].plot(c_cent, label="Central", drawstyle="steps-post")
    ax[1].plot(c_sec, label="Second Order", drawstyle="steps-post")
    ax[1].legend()

    fig.tight_layout()


class Parameters:
    horizon = 13
    n_constr = horizon
    times = list(range(horizon))

    sys_ids = ["ag00", "ag01"]
    n_agents = len(sys_ids)
    p_lim = 15.0

    ev_connected = [1] * horizon
    # soc_init = 0.
    # soc_target = 0.8
    # fc_res_load = [[x * 0.1 + 1. for x in times]] * n_agents
    # fc_res_load = [[1. for _ in times]] * n_agents
    fc_res_load = {sys_id: [1.2] * 5 + [9, 9, 9, 9, 9] + [0] * 3 for sys_id in sys_ids}

    sim_pars: Dict[str, type_defs.SysParsPVBat] = dict(
        zip(
            sys_ids,
            [
                type_defs.SysParsPVBat(
                    name="ag0",
                    c_sup=[0.2 + 0.0005 * k for k in times],
                    c_feed=0.0,
                    capacity=30.0,
                    dt_h=1.0,
                    p_inv=5.0,
                    init_soc=0.0,
                ),  # Agent 0
                type_defs.SysParsPVBat(
                    name="ag1",
                    c_sup=[0.41 + 0.0005 * k for k in times],
                    c_feed=0.0,
                    capacity=30.0,
                    dt_h=1.0,
                    p_inv=5.0,
                    init_soc=0.0,
                ),  # Agent 1
            ],
        )
    )
    on_off = dict(zip(sys_ids, [False, False]))

    max_iter = 70
    # 1 for no plots
    plot_every = 199
    opt_pars = type_defs.OptParameters(
        solver_name="osqp", rho=1000.0, mu=100.0, horizon=horizon, alpha=1.0
    )

    def __init__(self, problem_descr: dict = {}) -> None:

        for sys_id in self.sys_ids:
            self.sim_pars[sys_id].on_off = self.on_off[sys_id]

        if problem_descr:
            self.fc_res_load = problem_descr["fc_res_load"]
            self.horizon = len(self.fc_res_load[list(self.fc_res_load.keys())[0]])
            self.n_constr = self.horizon
            self.times = list(range(self.horizon))

            self.sys_ids = list(self.fc_res_load.keys())
            self.n_agents = len(self.sys_ids)

            self.sim_pars = problem_descr["sim_pars"]
            for sys_id in self.sim_pars:
                p = problem_descr["sim_pars"][sys_id]
                self.sim_pars[sys_id] = type_defs.SysParsPVBat(
                    name=sys_id,
                    c_sup=p["c_sup"],
                    c_feed=p["c_feed"],
                    capacity=p["capacity"],
                    p_inv=p["capacity"],
                    dt_h=p["dt_h"],
                    init_soc=problem_descr["states"][sys_id]["soc"],
                )
                # self.sim_pars[sys_id].init_soc = problem_descr["states"][sys_id]["soc"]

            self.on_off = {sys_id: self.sim_pars[sys_id].on_off for sys_id in self.sim_pars}
            self.p_lim = problem_descr["p_lim"]

    def plot_fc_res(self):
        fig, ax = style.styled_plot()
        sum = np.zeros(self.horizon)
        for id, fc in self.fc_res_load.items():
            ax.plot(fc, label=id)

            sum += fc

        ax.plot(sum, label="sum")

        fig.legend()
        plt.show()


def centralized(parameters: Parameters):

    combiner = problem_combiner.ProblemCombiner(parameters.opt_pars, parameters.horizon)
    agents = {}
    for sys_id in parameters.sys_ids:
        _pars = parameters.sim_pars[sys_id]
        if parameters.on_off[sys_id]:
            raise ValueError("Implement vectorized version for on off solver")
            agents[sys_id] = local_solver_pv_bat.OnOffPVBatSolver(parameters.horizon, sys_id, _pars)
        else:
            agents[sys_id] = local_solver_pv_bat.CasadiSolverLocalPVBatVectorized(
                parameters.horizon, sys_id, _pars, parameters.opt_pars
            )

    combiner.combine(agents, parameters.p_lim)

    par_values = {
        sys_id: {
            "x_init": [parameters.sim_pars[sys_id].init_soc],
            "lam_a": [0.0] * parameters.horizon,
            "fc_p_unc": parameters.fc_res_load[sys_id],
        }
        for sys_id in parameters.sys_ids
    }

    res = combiner.solve(par_values)

    grid_combined = np.zeros(parameters.horizon)
    for a in res:
        grid_combined += res[a]["yk"]

    y_ks = {a: res[a]["yk"] for a in res}
    x_terms = {a: np.round(res[a]["xterm"], 2) for a in res}
    e_term = sum([x_terms[a] * parameters.sim_pars[a].capacity for a in parameters.sys_ids])
    plot(
        parameters,
        None,
        grid_combined,
        y_ks,
        {a: res[a]["uk"] for a in res},
        show=False,
        title=f"Central {e_term}",
    )

    evaluator = schedule_eval.ScheduleEvaluator(y_ks, parameters.sim_pars)
    costs = [evaluator.get_costs()] * parameters.max_iter
    viols = [evaluator.get_constr_viol(parameters.p_lim)] * parameters.max_iter

    return costs, viols


def vujanic(parameters: Parameters):
    raise NotImplementedError("This must be reimplemented before it is usable again.")

    # s_up = {
    #     sys_id: np.array([-100000.0] * parameters.n_constr)
    #     for sys_id in parameters.sys_ids
    # }
    # s_lo = {
    #     sys_id: np.array([100000] * parameters.n_constr)
    #     for sys_id in parameters.sys_ids
    # }
    # rho = {sys_id: np.zeros(parameters.n_constr) for sys_id in parameters.sys_ids}
    # # grid_combined = np.zeros(horizon)
    # lam = np.zeros(parameters.horizon)

    # y_ks = {a: np.zeros(parameters.horizon) for a in parameters.sys_ids}
    # u_ks = {a: np.zeros(parameters.horizon) for a in parameters.sys_ids}

    # viols = []

    # lams = [[], []]
    # lam_diffs = []
    # costs = []

    # for iteration in range(parameters.max_iter):
    #     print(f"===============Iteration {iteration} ==========")

    #     for a in parameters.sys_ids:
    #         # for a in [iteration % n_agents]:
    #         # agent = local_solver_ev.CasadiSolverLocalEV(horizon)
    #         # # Idea: add this to lambda for agents to maybe divert the agents away from each other
    #         # # This lead to the agents being away from each other sometimes.
    #         # sigma = 10.
    #         # lambda_noise = np.random.normal(0., sigma, lam.shape)
    #         #
    #         # uk, yk = agent.solve(soc_init, soc_target, fc_res_load[a], {"lambda": lam}, ev_connected)
    #         _pars = parameters.sim_pars[a]
    #         if parameters.on_off[a]:
    #             agent = local_solver_pv_bat.OnOffPVBatSolver(
    #                 parameters.horizon, a, _pars
    #             )
    #         else:
    #             agent = local_solver_pv_bat.CasadiSolverLocalPVBat(
    #                 parameters.horizon, a, _pars, parameters.opt_pars
    #             )
    #         uk, yk = agent.solve(
    #             parameters.sim_pars[a]["init_soc"],
    #             parameters.fc_res_load[a],
    #             {"lambda": lam},
    #         )
    #         y_ks[a] = yk
    #         u_ks[a] = uk

    #         s_up[a] = np.maximum(s_up[a], yk)
    #         s_lo[a] = np.minimum(s_lo[a], yk)

    #         # print(f"yk: {yk}  | sup: {s_up[a]} | s_lo: {s_lo[a]}")

    #         rho[a] = s_up[a] - s_lo[a]

    #     rho_all = parameters.n_constr * np.array(list(rho.values())).max(axis=0)

    #     grid_combined = np.sum(np.array([np.array(y_ks[a]) for a in y_ks]), axis=0)
    #     lams[0] += [lam[1]]
    #     lams[1] += [lam[6]]
    #     lam_diffs += [lam[6] - lam[1]]

    #     # print(constr_violation)
    #     evaluator = schedule_eval.ScheduleEvaluator(y_ks, parameters.sim_pars)
    #     costs += [evaluator.get_costs()]
    #     viols += [evaluator.get_constr_viol(parameters.p_lim)]

    #     def alpha(i):
    #         return 0.01 / (i + 3.0)

    #     # lam = np.maximum(0., lam + alpha(iteration) * (grid_combined - parameters.p_lim + rho_all))
    #     lam = np.maximum(
    #         0.0, lam + alpha(iteration) * (grid_combined - parameters.p_lim)
    #     )
    #     if iteration == parameters.max_iter - 1:
    #         plot(
    #             parameters,
    #             lam=None,
    #             lam_slack=lam,
    #             grid_combined=grid_combined,
    #             y_ks=y_ks,
    #             u_ks=u_ks,
    #             show=False,
    #             title="Vujanic",
    #         )

    # # convergence behavior of costs
    # return costs, viols


def single_admm(parameters: Parameters):

    coord = coord_admm.CoordinatorADMM(
        parameters.opt_pars, parameters.sys_ids, type_defs.GridDescription(parameters.p_lim)
    )

    viols = []
    costs = []

    y_ks = {a: np.zeros(parameters.horizon) for a in parameters.sys_ids}
    u_ks = {a: np.zeros(parameters.horizon) for a in parameters.sys_ids}

    current_signals = coord.get_signals(
        {
            a: type_defs.LocalFuture(yg=y_ks[a], _meta={"fc": u_ks[a], "state": {"k": 1}})
            for a in y_ks
        }
    )

    for iteration in range(parameters.max_iter):
        print(f"ADMM iteration {iteration}")
        x_term = {}
        for a in parameters.sys_ids:
            agent = local_admm.LocalADMMSolver(
                parameters.horizon, a, parameters.sim_pars[a], parameters.opt_pars, []
            )
            agent.solve(
                parameters.sim_pars[a].init_soc, parameters.fc_res_load[a], current_signals[a], []
            )

            x_term[a] = np.round(agent.nlp_solver.opt_vector(f"x_a_{a}")[-1], 2)
            y_ks[a] = agent.get_yg()
            u_ks[a] = agent.get_u()

        # add slack agent manually here

        e_term = sum([x_term[a] * parameters.sim_pars[a].capacity for a in parameters.sys_ids])

        current_signals = coord.get_signals(
            {
                a: type_defs.LocalFuture(yg=y_ks[a], _meta={"fc": u_ks[a], "state": {"k": 1}})
                for a in y_ks
            }
        )

        grid_combined = np.sum(np.array([np.array(y_ks[a]) for a in y_ks]), axis=0)
        grid_plus_slack = grid_combined

        evaluator = schedule_eval.ScheduleEvaluator(y_ks, parameters.sim_pars)
        costs += [evaluator.get_costs()]
        viols += [evaluator.get_constr_viol(parameters.p_lim)]

        if iteration == parameters.max_iter - 1:
            plot(
                parameters,
                coord.lam,
                grid_combined,
                y_ks,
                u_ks,
                coord.ref_power_grid,
                grid_plus_slack,
                show=False,
                title=f"ADMM E term: {e_term}",
            )
        # plt.show()

    return costs, viols


def single_second_order(parameters: Parameters):

    coord = coord_second_order.CoordinatorSecondOrder(
        parameters.opt_pars, parameters.sys_ids, type_defs.GridDescription(parameters.p_lim)
    )

    viols = []
    costs = []

    y_ks = {a: parameters.fc_res_load[a] for a in parameters.sys_ids}
    u_ks = {a: np.zeros(parameters.horizon) for a in parameters.sys_ids}
    gradients = {a: np.zeros(parameters.horizon) for a in parameters.sys_ids}
    active_constraints = {a: np.array([[]]) for a in parameters.sys_ids}

    current_signals = coord.get_signals(
        {
            a: type_defs.LocalFuture(
                yg=y_ks[a],
                grads=gradients[a],
                jacobian=active_constraints[a],
                _meta={"fc": u_ks[a], "state": {"k": None}},
            )
            for a in y_ks
        }
    )

    grid_combined = np.sum(np.array([np.array(y_ks[a]) for a in y_ks]), axis=0)
    grid_plus_slack = grid_combined
    plot(
        parameters,
        {a: coord.lam for a in y_ks},
        grid_combined,
        y_ks,
        u_ks,
        coord.ref_power_grid,
        grid_plus_slack,
        coord.lam,
        show=False,
        title="Second Order iteration initialization, terminal x = ",
    )

    yg_set = {a: current_signals[a].res_power_set for a in current_signals}
    set_viols = [
        schedule_eval.ScheduleEvaluator(yg_set, parameters.sim_pars).get_constr_viol(
            parameters.p_lim
        )
    ]
    lams = [coord.lam.sum()]

    for iteration in range(parameters.max_iter):

        print(f"Iteration {iteration}")

        gradients = {}
        x_term = {}
        active_constraints = {}

        for a in parameters.sys_ids:
            agent = local_admm.LocalADMMSolver(
                parameters.horizon, a, parameters.sim_pars[a], parameters.opt_pars, []
            )

            agent.solve(
                parameters.sim_pars[a].init_soc,
                parameters.fc_res_load[a],
                current_signals[a],
                prev_signals=[],
            )

            # x_term[a] = np.round(agent.nlp_solver.opt_vector(f"x_a_{a}")[-1], 2)

            gradients[a] = agent.get_local_gradients()
            active_constraints[a] = agent.get_active_constraint_jacobian()

            # fig, ax = style.styled_plot(xlabel="time", ylabel="gradient",
            #                             title=f"agent: {a}, iteration {iteration}", figsize="landscape")
            # ax.plot(gradients[a], drawstyle="steps-post")

            y_ks[a] = agent.get_yg()
            u_ks[a] = agent.get_u()

        # print(f"Absolute controls: {np.array([np.abs(u_ks[a]).sum() for a in u_ks]).sum(): 0.03f}")

        # e_term = sum([x_term[a] * parameters.sim_pars[a].capacity for a in parameters.sys_ids])

        # coord.update_central(y_ks, gradients)
        current_signals = coord.get_signals(
            {
                a: type_defs.LocalFuture(
                    yg=y_ks[a],
                    grads=gradients[a],
                    jacobian=active_constraints[a],
                    _meta={"fc": u_ks[a], "state": {"k": None}},
                )
                for a in y_ks
            }
        )

        lams += [coord.lam.sum()]

        yg_set = {a: current_signals[a].res_power_set for a in current_signals}
        set_viols += [
            schedule_eval.ScheduleEvaluator(yg_set, parameters.sim_pars).get_constr_viol(
                parameters.p_lim
            )
        ]

        grid_combined = np.sum(np.array([np.array(y_ks[a]) for a in y_ks]), axis=0)
        grid_plus_slack = grid_combined

        evaluator = schedule_eval.ScheduleEvaluator(y_ks, parameters.sim_pars)
        costs += [evaluator.get_costs()]
        viols += [evaluator.get_constr_viol(parameters.p_lim)]

        if iteration in [0, 1, 2, 3, 62, 63, parameters.max_iter - 1]:
            # if iteration == parameters.max_iter - 1:
            # if False:
            plot(
                parameters,
                {a: coord.lam for a in y_ks},
                grid_combined,
                y_ks,
                u_ks,
                coord.ref_power_grid,
                grid_plus_slack,
                coord.lam,
                show=False,
                title=f"Second Order iteration {iteration}, terminal x = ",
            )

    fig, ax = style.styled_plot(figsize="landscape")
    ax.plot(lams)

    return costs, viols, set_viols


def compare(parameters: Parameters):

    # c_cent, v_cent = centralized(parameters)

    c_admm, v_admm = single_admm(parameters)
    c_sec_ord, v_sec_ord, set_viols = single_second_order(parameters)

    c_vuja, v_vuja = c_sec_ord, set_viols
    # c_admm, v_admm = c_sec_ord, v_sec_ord
    c_cent, v_cent = c_sec_ord, v_sec_ord
    # c_vuja, v_vuja = vujanic(parameters)
    c_cent, v_cent = centralized(parameters)

    # plot_meta([1,2,3], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [6, 7, 8])
    plot_meta(
        c_admm,
        v_admm,
        c_vuja,
        v_vuja,
        c_cent,
        v_cent,
        c_sec_ord,
        v_sec_ord,
        title=f"Constraint violation: {np.array(v_sec_ord[20:]).sum()}",
    )

    plt.show()


if __name__ == "__main__":
    from numpy import array

    # inspection_file_name =  Path(__file__).parent.parent.parent / "results" / "default" / "2_agents.json"
    inspection_file_name = (
        "/home/agross/src/python/grecco/grecco_sim/results/default/240906_1100/inspection.json"
    )
    with open(inspection_file_name, "r") as in_file:
        descr = json.load(in_file)

    # print(descr)

    # pars = Parameters()
    pars = Parameters(descr)

    for a in pars.sim_pars:
        pars.sim_pars[a].init_soc = 0.0

    # pars.p_lim=4.2
    pars.opt_pars.mu = 500.0
    pars.opt_pars.rho = 2.5

    compare(pars)
