"""
Second order coordinator using constraint gradients and Jacobian.
"""

from typing import Dict, Tuple, Union
import numpy as np

from grecco_sim.analysis import inspection_plots
from grecco_sim.coordinators import coord_interface
from grecco_sim.local_control import mycas
from grecco_sim.util import sig_types
from grecco_sim.util import type_defs
from grecco_sim.util import helper


class IterationStateSecondOrder(object):
    """State of the optimization"""

    lam: np.ndarray = np.array([])


# Find some encapsulation level to bring together functionality with coordinator ADMM (and vujanic?)
class CoordinatorSecondOrder(coord_interface.CoordinatorInterface):
    """
    Interface definition for
    a dictionary for each agent with two entries:

    {
        "lambda": [l1, l2, ..., lN-1],  # horizon values for lamda_k.
        "res_power_set": [p_ref1, ..., p_refN-1]  # horizon values for reference grid power.
    }

    """

    coord_name = "Second Order Coordinator"
    SLACK_AGENT = "ag_slack"

    signal_tolerance = 0.1

    def __init__(
        self,
        controller_pars: type_defs.OptParameters,
        sys_ids: list[str],
        grid: type_defs.GridDescription,
    ):

        super().__init__(controller_pars.horizon)

        # Time the controller is currently in
        self.current_time = -1

        self.lam: np.ndarray = np.array([])
        self.ref_power_grid: Dict[str, np.ndarray] = {}

        self.constraint_value = 1e10
        self.prev_constraint_value = 2 * self.constraint_value

        self.opt_p = controller_pars
        self.grid_p = grid

    def get_initial_signal(self):
        return sig_types.NoneSignal()

    def __init_iteration(
        self, flex_futures: Dict[str, type_defs.LocalFuture], inflex_sum: np.ndarray
    ):
        self.lam = np.zeros(list(flex_futures.values())[0].horizon)
        self.ref_power_grid = {a: flex_futures[a].yg for a in flex_futures}

        self.ref_power_grid[self.SLACK_AGENT] = -np.array(
            [flex_futures[a].yg for a in flex_futures]
        ).sum(axis=0)

        self.constraint_value = 1e10
        self.prev_constraint_value = 2 * self.constraint_value

    def get_signal(self, sys_id) -> Union[sig_types.SecondOrderSignal, sig_types.FirstOrderSignal]:
        """Get signal for a specific agent.
        System must be solved before.

        Args:
            sys_id (str): id of agent

        Returns:
            dict: returns signal. For ADMM is a lambda and res_power_set vector for system
        """
        if sys_id in self.ref_power_grid:
            return sig_types.SecondOrderSignal(
                mul_lambda=self.lam, res_power_set=self.ref_power_grid[sys_id]
            )
        else:
            return sig_types.FirstOrderSignal(self.lam)

    def get_signals(
        self, futures: Dict[str, type_defs.LocalFuture]
    ) -> Dict[str, sig_types.SignalType]:

        self.horizon = list(futures.values())[0].horizon

        (flex_futures, stat_futures, flex_sum, inflex_sum) = helper.get_flex_and_inflex(futures)

        if self.current_time != list(futures.values())[0].k:
            self.__init_iteration(flex_futures, inflex_sum)
            self.current_time = list(futures.values())[0].k

        # Solve 'local' problem for the slack agent
        yk_static = handle_static_agent_problem(
            sig_types.SecondOrderSignal(
                mul_lambda=self.lam, res_power_set=self.ref_power_grid[self.SLACK_AGENT]
            ),
            self.opt_p.rho,
            self.grid_p.p_lim,
            inflex_sum,
            self.opt_p.solver_name,
        )

        y_ks = {a: flex_futures[a].yg for a in flex_futures}
        y_ks[self.SLACK_AGENT] = yk_static
        self.constraint_value = (
            np.abs(np.array([y_ks[a] for a in y_ks]).sum(axis=0)).sum() / self.horizon
        )

        # debug = True
        # if debug:
        #     import matplotlib.pyplot as plt
        #     inspection_plots.plot_grid_powers(flex_sum, inflex_sum, yk_static, self.constraint_value)

        grads = {a: flex_futures[a].grads for a in flex_futures}
        grads[self.SLACK_AGENT] = np.zeros(self.horizon)

        active_constraints = {a: flex_futures[a].jacobian for a in flex_futures}
        active_constraints[self.SLACK_AGENT] = _get_active_constraints_static_agent(
            yk_static, inflex_sum, self.grid_p.p_lim
        )

        self.ref_power_grid, self.lam = _solve_second_order_problem(
            self.lam, y_ks, grads, active_constraints, self.opt_p.mu, self.opt_p.solver_name
        )

        # inspection_plots.plot_sec_order_ref_power(self.ref_power_grid, inflex_sum, flex_sum, self.lam)

        return {a: self.get_signal(a) for a in futures}

    def has_converged(self, time_index):
        if not time_index == self.current_time:
            # No calculation for current time yet
            return False

        if abs(self.constraint_value) < self.signal_tolerance:
            # Algorithm converged to reasonable tolerance
            return True

        if abs(self.prev_constraint_value - self.constraint_value) < self.signal_tolerance:
            print(f"Breaking iteration with constraint violation of {self.constraint_value}")
            return True

        return False

    def get_cost_realization(
        self,
        initial_futures: Dict[str, type_defs.LocalFuture],
        realization_grid: Dict[str, float],
        signals: Dict[str, sig_types.SecondOrderSignal],
    ) -> Dict[str, float]:
        assert set(initial_futures.keys()) == set(realization_grid.keys()) == set(signals.keys())

        ret = {
            f"fee_{sys_id}": signal.mul_lambda[0] * realization_grid[sys_id]
            for sys_id, signal in signals.items()
        }

        ret["signal"] = signals[list(signals.keys())[0]].mul_lambda[0]
        return ret


# ==================================================== Solver
#     This controller is currently maintained
#     Developped after Houska, Frasch, Diehl 2016(?)
def _create_second_order_dual_problem(horizon, sys_ids, mu, qp_solver_name, active_constraints):

    pars = {}
    states = []
    grid = 0
    obj = 0

    # One slack variable per coupling constraint (-> length: horizon)
    qp_slack = mycas.MySX("qp_slack", -np.inf, np.inf, horizon=horizon)
    states += [qp_slack]

    pars["lam"] = mycas.MyPar("lam", horizon)

    constraints = []

    for sys_id in sys_ids:

        pars[f"local_solution_{sys_id}"] = mycas.MyPar(f"local_solution_{sys_id}", horizon)
        pars[f"grad_a_{sys_id}"] = mycas.MyPar(f"grad_a_{sys_id}", horizon)

        # Delta y in Houska Paper
        d_pg_ref_a = mycas.MySX(f"d_pg_ref_a_{sys_id}", -np.inf, np.inf, horizon=horizon)
        states += [d_pg_ref_a]

        # obj += 0.5 * d_pg_ref_a.sx ** 2. * hess_a_k[sys_id, k]
        obj += mycas.dot(pars[f"grad_a_{sys_id}"].sx, d_pg_ref_a.sx)
        obj += mycas.dot(d_pg_ref_a.sx, d_pg_ref_a.sx)
        # obj -= pars[f"lam_{sys_id}"].sx[k] * d_pg_ref_a.sx

        # PLim Constraint
        grid += pars[f"local_solution_{sys_id}"].sx + d_pg_ref_a.sx

        # Constraint jacobian combination
        for jac in active_constraints[sys_id]:
            if len(jac) == 0:
                continue

            constraints += [mycas.MyConstr(mycas.dot(d_pg_ref_a.sx, np.array(jac)), 0.0, 0.0)]

    # Add qp_slack terms to objective
    obj += mycas.dot(pars["lam"].sx, qp_slack.sx)
    obj += mu * 0.5 * mycas.dot(qp_slack.sx, qp_slack.sx)

    constraints += [mycas.MyConstr(grid - qp_slack.sx, 0.0, 0.0, name="grid_constraint")]

    ocp = mycas.MyOCP(obj, states, constraints, pars)
    nlp_solver = mycas.MyNLPSolver(ocp, solver=qp_solver_name)

    return nlp_solver


def _solve_second_order_problem(
    old_lam: np.ndarray,
    yg: Dict[str, np.ndarray],
    grad_a: Dict[str, np.ndarray],
    active_constraints: Dict[str, np.ndarray],
    mu: float,
    qp_solver_name: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Solve central problem.

    yg includes the local solutions of all agents (including slack agent)
    grad_a includes the respective gradients

    Returns a Tuple of (reference powers, new lambda)
    """
    assert set(yg.keys()) == set(grad_a.keys())

    sys_ids = list(yg.keys())
    horizon = len(old_lam)
    nlp_solver = _create_second_order_dual_problem(
        horizon, sys_ids, mu, qp_solver_name, active_constraints
    )

    par_values = {
        "lam": old_lam,
    }
    par_values.update({f"grad_a_{a}": grad_a[a] for a in grad_a})
    par_values.update({f"local_solution_{a}": yg[a] for a in grad_a})

    nlp_solver.solve(par_values)

    new_lam = nlp_solver.constr_multipliers("grid_constraint")
    # print(self.new_lam)

    ret_ref_grid = {}
    for a in grad_a:
        d_pg_a = nlp_solver.opt_vector(f"d_pg_ref_a_{a}")
        ret_ref_grid[a] = yg[a] + d_pg_a

    return ret_ref_grid, new_lam


# ============ Functions for handling of slack agent. ===========================


def handle_slack_local_problem(signal: sig_types.SecondOrderSignal, rho, p_lim, qp_solver="osqp"):
    """
    Solve the "local problem" of the slack agent in Second order method.
    """

    pars = {}
    constraints = []
    signal.validate()
    horizon = signal.signal_len

    pars["ref_grid_power_slack"] = mycas.MyPar("ref_grid_power_slack", horizon)
    pars["lam_slack"] = mycas.MyPar("lam_slack", horizon)
    y = mycas.MySX("y_slack", -p_lim, p_lim, horizon=horizon)
    states = [y]

    obj = mycas.dot(pars["lam_slack"].sx, y.sx)
    obj += (
        rho
        / 2.0
        * mycas.dot(y.sx - pars["ref_grid_power_slack"].sx, y.sx - pars["ref_grid_power_slack"].sx)
    )

    ocp = mycas.MyOCP(obj, states, constraints, pars)
    nlp_solver = mycas.MyNLPSolver(ocp, solver=qp_solver)
    nlp_solver.solve({"ref_grid_power_slack": signal.res_power_set, "lam_slack": signal.mul_lambda})

    return nlp_solver.opt_vector("y_slack")


def _get_active_constraints_static_agent(
    yk_static: np.ndarray, inflex_sum: np.ndarray, p_lim: float
):
    """Identify if the slack agent is at a boundary."""
    ret = []
    ret += np.eye(len(yk_static))[np.where(yk_static - inflex_sum - p_lim >= 0)].tolist()
    ret += (-np.eye(len(yk_static)))[np.where(-(yk_static - inflex_sum) - p_lim >= 0)].tolist()
    return ret


def handle_static_agent_problem(
    signal: sig_types.SecondOrderSignal,
    rho: float,
    p_lim: float,
    uncontrolled_agents: np.ndarray,
    qp_solver="osqp",
):
    """Solve the static agent problem.
    That is the agent which asserts grid limits given the profile of the uncontrolled agents.
    """
    pars = {}
    constraints = []
    signal.validate()
    horizon = signal.signal_len

    pars["ref_grid_power_static"] = mycas.MyPar("ref_grid_power_static", horizon)
    pars["lam_static"] = mycas.MyPar("lam_static", horizon)

    # This is the power profile of the 'static agent'.
    # The static agent is the power profile of uncontrollables + slack agent.
    # The slack agent is the mirrored value of all other agents:
    # y_slack + sum(controlled) + sum(uncontrolled) = 0
    y_static = mycas.MySX("y_static", -np.inf, np.inf, horizon=horizon)
    states = [y_static]

    # Implicitly this is the constraint of the slack agent.
    constraints += [
        mycas.MyConstr(y_static.sx - uncontrolled_agents, -p_lim, p_lim, "static grid limit")
    ]

    obj = mycas.dot(pars["lam_static"].sx, y_static.sx)
    obj += (
        rho
        / 2.0
        * mycas.dot(
            y_static.sx - pars["ref_grid_power_static"].sx,
            y_static.sx - pars["ref_grid_power_static"].sx,
        )
    )

    ocp = mycas.MyOCP(obj, states, constraints, pars)
    nlp_solver = mycas.MyNLPSolver(ocp, solver=qp_solver)
    nlp_solver.solve(
        {"ref_grid_power_static": signal.res_power_set, "lam_static": signal.mul_lambda}
    )

    return nlp_solver.opt_vector("y_static")


if __name__ == "__main__":
    pass
