import numpy as np
from typing import Dict

from grecco_sim.coordinators import coord_interface
from grecco_sim.coordinators.implementations import coord_second_order
from grecco_sim.local_control import mycas
from grecco_sim.util import helper, sig_types
from grecco_sim.util import type_defs


class CoordinatorADMM(coord_interface.CoordinatorInterface):
    """
    Interface definition for ADMM:

    a dictionary for each agent with two entries:

    {
        "lambda": [l1, l2, ..., lN-1],  # horizon values for lamda_k.
        "res_power_set": [p_ref1, ..., p_refN-1]  # horizon values for reference grid power.
    }

    """

    coord_name = "ADMM"
    SLACK_AGENT = "ag_slack"

    signal_tolerance = 0.01

    def __init__(
        self,
        controller_pars: type_defs.OptParameters,
        sys_ids: list[str],
        grid: type_defs.GridDescription,
    ):

        super().__init__(controller_pars.horizon)

        # Time the controller is currently in
        self.current_time = -1

        self.opt_pars = controller_pars
        self.grid_pars = grid

        # Initialize containers for signal storage
        self.lam: Dict[str, np.ndarray] = {}
        self.ref_power_grid: Dict[str, np.ndarray] = {}
        self.ref_change = np.array([[np.inf]])
        self.constraint_value = 1e10
        self.prev_constraint_value = 2 * self.constraint_value

    def update_central(self, y_ks: Dict[str, np.ndarray]):
        """
        This function changes the state of the Coordinator object by one iteration:
        that means that the local variables for lambda and ref power are updated.

        :param y_ks: current grid power profiles of the connected agents (==futures)
        :return:
        """

        sys_ids = set(y_ks.keys())
        assert set(self.lam.keys()) == sys_ids

        for a in sys_ids:
            self.lam[a] = self.lam[a] + self.opt_pars.rho * (y_ks[a] - self.ref_power_grid[a])

        # Actual problem definition and solution for all agents including slack agent
        coord = ADMMCentralSolver(self.horizon, sys_ids, self.opt_pars.rho, self.grid_pars.p_lim)
        solution = coord.solve(self.lam, y_ks)

        self.ref_change = np.array([solution[a] - self.ref_power_grid[a] for a in solution])
        self.ref_power_grid = solution

    def get_signal(self, sys_id) -> sig_types.SecondOrderSignal:
        """Get signal for a specific agent.
        System must be solved before.

        Args:
            sys_id (str): id of agent

        Returns:
            dict: returns signal. For ADMM is a lambda and res_power_set vector for system
        """
        if sys_id in self.ref_power_grid:
            return sig_types.SecondOrderSignal(
                mul_lambda=self.lam[sys_id], res_power_set=self.ref_power_grid[sys_id]
            )
        else:
            return sig_types.FirstOrderSignal(np.zeros(1))

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

    def __init_iteration(self, futures: Dict[str, type_defs.LocalFuture]):
        """Initialize an iteration with the local solution."""
        self.lam = {a: np.zeros(futures[a].horizon) for a in futures}
        self.ref_power_grid = {a: futures[a].yg for a in futures}

        self.lam[self.SLACK_AGENT] = np.zeros(list(futures.values())[0].horizon)
        self.ref_power_grid[self.SLACK_AGENT] = -np.array([futures[a].yg for a in futures]).sum(
            axis=0
        )

        self.ref_change = np.array([[np.inf]])
        self.constraint_value = 1e10
        self.prev_constraint_value = 2 * self.constraint_value

    def get_signals(
        self, futures: Dict[str, type_defs.LocalFuture]
    ) -> Dict[str, sig_types.SignalType]:

        self.horizon = list(futures.values())[0].horizon

        (flex_futures, stat_futures, flex_sum, inflex_sum) = helper.get_flex_and_inflex(futures)

        if self.current_time != list(futures.values())[0].k:
            self.__init_iteration(flex_futures)
            self.current_time = list(futures.values())[0].k

        # Solve 'local' problem for the slack agent
        yk_slack = coord_second_order.handle_static_agent_problem(
            sig_types.SecondOrderSignal(
                self.ref_power_grid[self.SLACK_AGENT], self.lam[self.SLACK_AGENT]
            ),
            self.opt_pars.rho,
            self.grid_pars.p_lim,
            inflex_sum,
            qp_solver=self.opt_pars.solver_name,
        )

        y_ks = {a: flex_futures[a].yg for a in flex_futures}
        y_ks[self.SLACK_AGENT] = yk_slack

        # Get the constraint violation
        self.constraint_value = (
            np.abs(np.array([y_ks[a] for a in y_ks]).sum(axis=0)).sum() / self.horizon
        )

        self.update_central(y_ks)

        return {a: self.get_signal(a) for a in futures}

    def get_initial_signal(self) -> sig_types.SignalType | None:
        return sig_types.NoneSignal()


class ADMMCentralSolver(object):
    """
    This controller is currently maintained
    """

    qp_solver = "osqp"

    def __init__(self, horizon, sys_ids, rho, p_lim):

        self.horizon = horizon
        self.sys_ids = sys_ids
        self.p_lim = p_lim
        self.rho = rho

        self._create_problem()

    def _create_problem(self):

        self.pars = {}

        states = []
        grid = 0.0
        obj = 0.0

        for sys_id in self.sys_ids:

            self.pars[f"lam_{sys_id}"] = mycas.MyPar(f"lam_{sys_id}", self.horizon)
            self.pars[f"local_solution_{sys_id}"] = mycas.MyPar(
                f"local_solution_{sys_id}", self.horizon
            )

            ref_grid_a = mycas.MySX(f"ref_grid_a_{sys_id}", -np.inf, np.inf, horizon=self.horizon)
            states += [ref_grid_a]
            grid += ref_grid_a.sx

            obj += (
                self.rho
                / 2.0
                * mycas.dot(
                    ref_grid_a.sx - self.pars[f"local_solution_{sys_id}"].sx,
                    ref_grid_a.sx - self.pars[f"local_solution_{sys_id}"].sx,
                )
            )

            obj -= mycas.dot(self.pars[f"lam_{sys_id}"].sx, ref_grid_a.sx)

        constraints = [mycas.MyConstr(grid, 0.0, 0.0)]

        ocp = mycas.MyOCP(obj, states, constraints, self.pars)

        self.nlp_solver = mycas.MyNLPSolver(ocp, solver=self.qp_solver)

    def solve(self, lam, yg):
        """

        Solve the central ADMM QP created in initialization

        Args:
            lam (_type_): _description_
            yg (_type_): _description_
            lam_slack (_type_): _description_
            yk_slack (_type_): _description_

        Returns:
            _type_: _description_
        """

        par_values = {}
        par_values.update({f"lam_{a}": lam[a] for a in lam})
        par_values.update({f"local_solution_{a}": yg[a] for a in lam})

        self.nlp_solver.solve(par_values)

        ret_ref_grid = {a: self.nlp_solver.opt_vector(f"ref_grid_a_{a}") for a in lam}
        return ret_ref_grid
