import numpy as np
from typing import List, Dict

from grecco_sim.local_control import mycas
from grecco_sim.coordinators import coord_interface
from grecco_sim.util import sig_types
from grecco_sim.util import type_defs


class FirstOrderCoordinatorBase(coord_interface.CoordinatorInterface):
    """Base class for first-order coordination methods."""

    SLACK_AGENT = "ag_slack"

    def __init__(self, horizon: int):
        super().__init__(horizon)
        self.current_time = -1
        self.lam: np.ndarray = np.array([])

    def get_initial_signal(self) -> sig_types.NoneSignal:
        """Return initial signal for first order methods."""
        return sig_types.NoneSignal()

    def get_cost_realization(
        self,
        initial_futures: dict[str, type_defs.LocalFuture],
        realization_grid: dict[str, float],
        signals: dict[str, sig_types.SecondOrderSignal],
    ) -> Dict[str, float]:
        """
        Determine the cost contribution by the extra cordination term.

        Each first order method adds a term to the local cost function
        (the additional time variant grid fee).
        This function calculates this penalty for each agent from the 'measured'
        grid power.

        Currently this function will print if the promised grid power is different from
        the realized one. (Depending on baseline usage this has to change).

        Arguments:
            initial_futures (dict[str, type_defs.LocalFuture]):
                Futures sent to the coordinator initially (the baseline).
            realization_grid (dict[str, float]):
                Measured (realized) grid power of each agent.
            signals (dict[str, sig_types.SecondOrderSignal]):
                Signals that were communicated as binding to the agents.

        The set of keys of each argument has to match.

        Returns:
            costs (dict[str, float]): a single value of penalty from the realized grid power of each agent
        """
        assert set(initial_futures.keys()) == set(realization_grid.keys()) == set(signals.keys())

        ret = {
            f"fee_{sys_id}": signal.mul_lambda[0] * realization_grid[sys_id]
            for sys_id, signal in signals.items()
        }

        ret["signal"] = signals[list(signals.keys())[0]].mul_lambda[0]

        # false_promises = {
        #     sys_id: realization_grid[sys_id] - initial_futures[sys_id].yg[0]
        #     for sys_id in initial_futures
        #     if abs(realization_grid[sys_id] - initial_futures[sys_id].yg[0]) > 0.01
        # }
        # if false_promises:
        #     print(false_promises)

        return ret


class CoordinatorGradientDescent(FirstOrderCoordinatorBase):
    """
    Coordinator for plain gradient descent.
    """

    coord_name = "Gradient Descent Coordinator"
    SLACK_AGENT = "ag_slack"

    signal_tolerance = 0.1

    def __init__(
        self,
        opt_pars: type_defs.OptParameters,
        sys_ids: List[str],
        grid: type_defs.GridDescription,
    ):

        super().__init__(opt_pars.horizon)

        # Time the controller is currently in
        self.current_time = -1

        # Variables for opt iterations
        self.constraint_value = 1e10
        self.prev_constraint_value = 2 * self.constraint_value

        self.opt_p = opt_pars
        self.grid_p = grid

    def __init_iteration(self, futures: Dict[str, type_defs.LocalFuture]):
        self.lam = np.zeros(list(futures.values())[0].horizon)
        self.constraint_value = 1e10
        self.prev_constraint_value = 2 * self.constraint_value

    def get_signals(
        self, futures: Dict[str, type_defs.LocalFuture]
    ) -> Dict[str, sig_types.SignalType]:

        sys_ids = list(futures.keys())
        self.horizon = futures[sys_ids[0]].horizon

        if self.current_time != futures[sys_ids[0]].k:
            self.__init_iteration(futures)
            self.current_time = futures[sys_ids[0]].k

        # Solve 'local' problem for the slack agent
        yk_slack = _handle_slack_local_problem(
            sig_types.FirstOrderSignal(self.lam),
            self.grid_p,
            qp_solver=self.opt_p.solver_name,
        )

        y_ks = {a: futures[a].yg for a in futures}
        y_ks[self.SLACK_AGENT] = yk_slack

        # Update Lambda multiplier
        self.lam = self._update_lambda(y_ks, self.lam)

        # Update constraint value for convergence check
        self.constraint_value = (
            np.abs(np.array([y_ks[a] for a in y_ks]).sum(axis=0)).sum() / self.horizon
        )

        return {a: sig_types.FirstOrderSignal(mul_lambda=self.lam) for a in futures}

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

    def _update_lambda(self, y_ks: Dict[str, np.ndarray], lam: np.ndarray) -> np.ndarray:
        """Return Updated lambda for gradient descent."""
        return lam + self.opt_p.alpha * np.array([y_ks[a] for a in y_ks]).sum(axis=0)


class CoordinatorVujanic(coord_interface.CoordinatorInterface):
    """
    This class implements the coordination mechanism described in

    Falsone, Alessandro, Kostas Margellos, and Maria Prandini. “A Decentralized Approach to Multi-Agent MILPs:
    Finite-Time Feasibility and Performance Guarantees.” Automatica 103 (May 1, 2019): 141–50.
    https://doi.org/10.1016/j.automatica.2019.01.009.

    It is a coordination mechanism for distributed MILP problems.

    """

    MAX_CONSTR_VAL = 9999999999

    def __init__(self, controller_pars, sys_ids: List[str]):

        super().__init__(horizon=controller_pars["fc_horizon"])

        self.current_signal_time = -1
        self.p_lim = controller_pars["p_lim"]

        self.current_signal = None
        self.s_upper = None
        self.s_lower = None
        self.prev_rho = None

        self.constraint_value = self.MAX_CONSTR_VAL
        self.prev_constraint_value = 2 * self.MAX_CONSTR_VAL
        self.signal_tolerance = 0.001
        self.plotter = None

    def has_converged(self, time_index):
        if not time_index == self.current_signal_time:
            # No calculation for current time yet
            return False

        if abs(self.constraint_value) < self.signal_tolerance:
            # Algorithm converged to reasonable tolerance
            return True

        if abs(self.prev_constraint_value - self.constraint_value) < self.signal_tolerance:
            print(f"Breaking iteration with constraint violation of {self.constraint_value}")
            return True

        return False

    def _init_iteration(self, k, current_horizon, sys_ids):
        """
        Initialize the signal and auxiliary variables for new iteration

        :param k: current time index
        :param current_horizon: length of scheduling horizon in the iteration
        :param sys_ids: sys_ids of systems taking part in the iteration
        :return:
        """
        self.current_signal = {
            sys_id: {
                "lambda": np.zeros(current_horizon),
            }
            for sys_id in sys_ids
        }

        self.constraint_value = self.MAX_CONSTR_VAL
        self.prev_constraint_value = 2 * self.MAX_CONSTR_VAL

        self.s_upper = {
            sys_id: -self.MAX_CONSTR_VAL * np.ones(current_horizon) for sys_id in sys_ids
        }
        self.s_lower = {
            sys_id: self.MAX_CONSTR_VAL * np.ones(current_horizon) for sys_id in sys_ids
        }
        self.prev_rho = np.zeros(current_horizon)

        self.alpha_inv = 1.0

        self.current_signal_time = k

    def get_signals(self, futures):

        # ========= initialization ===============
        sys_ids = list(futures.keys())
        k = futures[sys_ids[0]]["state"]["k"]
        current_horizon = len(futures[sys_ids[0]]["fc"])

        if self.current_signal_time != k:
            self._init_iteration(k, current_horizon, sys_ids)
            # First iteration of coordination is to return zeros as the signal
            # (and request a schedule not respecting central constraints)
            return self.current_signal

        grid_lim = np.zeros(current_horizon)

        rho_a = {}
        for sys_id in sys_ids:
            y_a = futures[sys_id]["y_g"]
            grid_lim += y_a  # + s_a

            self.s_upper[sys_id] = np.maximum(self.s_upper[sys_id], y_a)
            self.s_lower[sys_id] = np.minimum(self.s_lower[sys_id], y_a)

            rho_a[sys_id] = self.s_upper[sys_id] - self.s_lower[sys_id]

        # This is the dimensionality of the coupling constraint
        n_constr = self.horizon
        rho_ak = np.array([rho_a[sys_id] for sys_id in rho_a])

        rho_all = n_constr * rho_ak.max(axis=0)
        # rho_all = [n_constr * 5.] * self.horizon

        new_lam = {}
        for sys_id in sys_ids:
            prev_lam = self.current_signal[sys_id]["lambda"]

            # Update Lambda
            new_lam[sys_id] = np.maximum(
                0.0, prev_lam + 1.0 / self.alpha_inv * (grid_lim - self.p_lim + rho_all)
            )

        self.alpha_inv += 1.0

        # Check constraints
        self.prev_constraint_value = self.constraint_value
        self.constraint_value = ((np.maximum(grid_lim - self.p_lim, 0)) ** 2).sum()

        if not self.has_converged(k):
            new_signal = {sys_id: {"lambda": new_lam[sys_id]} for sys_id in futures}
            self.current_signal = new_signal

        self.current_signal_time = k

        return self.current_signal


def _handle_slack_local_problem(
    signal: sig_types.FirstOrderSignal, grid: type_defs.GridDescription, qp_solver: str
) -> np.ndarray:

    pars = {}
    constraints = []
    signal.validate()
    horizon = signal.signal_len

    pars["lam_slack"] = mycas.MyPar("lam_slack", horizon)
    y = mycas.MySX("y_slack", -grid.p_lim, grid.p_lim, horizon=horizon)
    states = [y]

    obj = mycas.dot(pars["lam_slack"].sx, y.sx)

    nlp_solver = mycas.MyNLPSolver(obj, states, constraints, pars, solver=qp_solver)
    nlp_solver.solve({"lam_slack": signal.mul_lambda})

    return nlp_solver.opt_vector("y_slack")


class CoordinatorDailyGridFee(FirstOrderCoordinatorBase):
    """
    Coordinator for a static grid fee. (Maybe even just two steps HT/NT)
    """

    coord_name = "Static Grid Fee Coordinator"
    SLACK_AGENT = "ag_slack"

    def __init__(
        self,
        opt_pars: type_defs.OptParameters,
        sys_ids: List[str],
        grid: type_defs.GridDescription,
    ):

        super().__init__(opt_pars.horizon)

        # Time the controller is currently in
        self.current_time = -1

        # Variables for opt iterations
        self.opt_p = opt_pars
        self.grid_p = grid
        self.constraint_value = None

    def _update_lambda(self, y_ks: np.ndarray) -> np.ndarray:
        """Return Updated time varying grid fee."""
        current_grid_power = y_ks.sum(axis=0)
        lam = np.zeros(len(current_grid_power))

        lam[current_grid_power > self.grid_p.p_lim] = self.opt_p.alpha
        lam[current_grid_power < -self.grid_p.p_lim] = -self.opt_p.alpha

        return lam

    def get_signals(
        self, futures: Dict[str, type_defs.LocalFuture]
    ) -> Dict[str, sig_types.SignalType]:

        if self.current_time != list(futures.values())[0].k:
            self.lam = self._update_lambda(np.array([future.yg for future in futures.values()]))
            self.current_time = list(futures.values())[0].k

        return {a: sig_types.FirstOrderSignal(mul_lambda=self.lam) for a in futures}

    def has_converged(self, time_index):
        # Return if signal (=grid fee) has been adapted for current time index
        return time_index == self.current_time
