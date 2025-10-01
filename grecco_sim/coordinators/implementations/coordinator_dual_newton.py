import numpy as np

from grecco_sim.util import sig_types
from grecco_sim.util import type_defs
from grecco_sim.coordinators.implementations import coord_first_order


class CoordinatorDualNewton(coord_first_order.FirstOrderCoordinatorBase):
    """
    Coordinator for Reinhardt 2024 style Dual Newton steps.
    """

    coord_name = "Dual Newton coordinator"
    signal_tolerance = 0.1

    def __init__(
        self,
        opt_pars: type_defs.OptParameters,
        sys_ids: list[str],
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

    def __init_iteration(self, futures: dict[str, type_defs.LocalFuture]):
        self.lam = np.zeros(list(futures.values())[0].horizon)
        self.constraint_value = 1e10
        self.prev_constraint_value = 2 * self.constraint_value

    def get_signals(
        self, futures: dict[str, type_defs.LocalFuture]
    ) -> dict[str, sig_types.SignalType]:

        sys_ids = list(futures.keys())
        self.horizon = futures[sys_ids[0]].horizon

        if self.current_time != futures[sys_ids[0]].k:
            self.__init_iteration(futures)
            self.current_time = futures[sys_ids[0]].k

        # Solve 'local' problem for the slack agent
        # TODO get sensitivities
        yk_slack = coord_first_order._handle_slack_local_problem(
            sig_types.FirstOrderSignal(self.lam),
            self.grid_p,
            qp_solver=self.opt_p.solver_name,
        )

        y_ks = {a: futures[a].yg for a in futures}
        y_ks[self.SLACK_AGENT] = yk_slack

        sensitivities = {a: futures[a].multiplier_sensitivities for a in futures}

        # Update Lambda multiplier
        # self.lam = self._update_lambda(y_ks, sensitivities, self.lam)
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

    def _update_lambda(self, y_ks: dict[str, np.ndarray], lam: np.ndarray) -> np.ndarray:
        """Return Updated lambda for gradient descent."""
        return lam + self.opt_p.alpha * np.array([y_ks[a] for a in y_ks]).sum(axis=0)
