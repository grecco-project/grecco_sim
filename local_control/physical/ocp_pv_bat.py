from typing import Any
import warnings
import numpy as np

from grecco_sim.local_control import mycas
from grecco_sim.local_control.solver import local_solver_base
from grecco_sim.util import style

from grecco_sim.util.type_defs import OptParameters, SysParsPVBat


def plain_pv_bat_problem(
    horizon: int, sys_id: str, sys_pars: SysParsPVBat
) -> tuple[mycas.MyOCP, mycas.MySX]:
    """Define plain OCP for pv battery system."""
    states = []
    constraints = []
    pars = {}
    obj: mycas.casadi.SX = 0
    user_functions = {}

    x = mycas.MySX(f"x_a_{sys_id}", sys_pars.x_lb, sys_pars.x_ub, horizon=horizon + 1)
    states += [x]

    pars["x_bat_init"] = mycas.MyPar(f"p_x_bat_init_a_{sys_id}")
    constraints.append(mycas.MyConstr(x.sx[0] - pars["x_bat_init"].sx, 0, 0, name="Initial state"))
    # constraints += [mycas.MyConstr(x.sx[horizon] - pars["x_init"].sx, 0, 0)]

    # Forecast of uncontrollable residual load on site (load - pv)
    pars["fc_p_unc"] = mycas.MyPar(f"fc_p_unc_a_{sys_id}", horizon)

    # =========== Define optimization variable time series ===============
    # Battery Power (charging and discharging). AC power
    p_ch = mycas.MySX(f"p_ch_a_{sys_id}", 0, sys_pars.p_inv, horizon=horizon)
    states += [p_ch]
    p_dch = mycas.MySX(f"p_dch_a_{sys_id}", 0, sys_pars.p_inv, horizon=horizon)
    states += [p_dch]

    user_functions["u_bat"] = p_ch.sx - p_dch.sx

    # Feed and supply
    xf = mycas.MySX(f"xf_a_{sys_id}", 0, np.inf, horizon=horizon)
    states += [xf]
    xs = mycas.MySX(f"xs_a_{sys_id}", 0, np.inf, horizon=horizon)
    states += [xs]

    # Constrain feed-in to pv production / residual generation.
    # Forecast of residual generation: this forecast is >= 0
    pars["fc_pv_prod"] = mycas.MyPar(f"fc_pv_prod_{sys_id}", horizon)
    # Here it is restricted to residual generation to be more restrictive.
    constraints += [
        mycas.MyConstr(
            xf.sx - pars["fc_pv_prod"].sx,
            -np.inf,
            0,
            name="No discharging into grid",
        )
    ]

    # Make a constraint that prevents charging the battery from grid.
    constraints.append(
        mycas.MyConstr(
            xs.sx - (pars["fc_p_unc"].sx + pars["fc_pv_prod"].sx),
            -np.inf,
            0.0,
            name="No charging from grid.",
        )
    )
    # Grid variable
    y_g = mycas.MySX(f"y_g_a_{sys_id}", -np.inf, np.inf, horizon=horizon)
    states += [y_g]

    # Handle electricity prices as a parameter time series
    pars["c_sup"] = mycas.MyPar(f"c_sup_a_{sys_id}", hor=horizon)
    pars["c_feed"] = mycas.MyPar(f"c_feed_a_{sys_id}", hor=horizon)
    obj += mycas.dot(xs.sx, pars["c_sup"].sx) - mycas.dot(xf.sx, pars["c_feed"].sx)

    # Formulate the NLP
    for k in range(horizon):
        # Supply - Feedin = Load - PV - discharge + charge
        constraints += [
            mycas.MyConstr(
                xs.sx[k] - xf.sx[k] - pars["fc_p_unc"].sx[k] + p_dch.sx[k] - p_ch.sx[k],
                0,
                0,
            )
        ]

        # Constrain the coupling variable to be supply - feed in
        constraints += [mycas.MyConstr(y_g.sx[k] - xs.sx[k] + xf.sx[k], 0, 0)]

        # Constrain state evolution
        x_plus = x.sx[k] + sys_pars.dt_h / sys_pars.capacity * (
            p_ch.sx[k] * sys_pars.eff - p_dch.sx[k] / sys_pars.eff
        )
        # f_x_next(x0=list_xk[-2].sx, u=uk.sx)["x_next"]
        constraints += [mycas.MyConstr(x_plus - x.sx[k + 1], 0.0, 0.0)]

    # Add terminal cost as bonus for energy in storage
    obj -= x.sx[-1] * sys_pars.capacity * sys_pars.dt_h * (sys_pars.c_sup + sys_pars.c_feed) * 0.5
    return mycas.MyOCP(obj, states, constraints, pars, user_functions), y_g


class CasadiSolverLocalPVBatVectorized(local_solver_base.LocalSolverBase):
    """Vectorized form of CasadiSolverLocalPVBat without lambda.

    Optimization parameters that need to be passed so solve:

    x_init: initial state of charge

    fc_p_unc: Time series forecast of residual load on site (load - pv gen)

    fc_pv_prod: Time series of PV production (> 0)

    """

    def __init__(
        self, horizon: int, sys_id: str, sys_pars: SysParsPVBat, opt_pars: OptParameters
    ) -> None:
        self.sys_pars = sys_pars
        self._create_problem(horizon, sys_id, sys_pars, opt_pars)

    def _create_problem(
        self, horizon: int, sys_id: str, sys_pars: SysParsPVBat, opt_pars: OptParameters
    ) -> None:

        # =================== Transform to casadi input ==============================
        self.problem, self._grid_var = plain_pv_bat_problem(horizon, sys_id, sys_pars)
        self.nlp_solver = mycas.MyNLPSolver(self.problem, solver=opt_pars.solver_name)

    def get_central_problem_contribution(
        self,
    ) -> tuple[mycas.MyOCP, mycas.MySX]:
        return self.problem, self._grid_var

    def get_user_functions(self) -> dict[str, mycas.casadi.SX]:
        # Get the state of charge out of the states to use as user function
        # Should be moved to the 'physical part' of the OCP
        # This is all just necessary because of difference between formulation for casadi and
        # in the ALADIN setup

        user_funcs = {}
        user_funcs["local_objective"] = self.problem[0]

        states = self.problem[1]

        u_ch = states[1]
        u_dch = states[2]
        user_funcs["u_ub"] = u_ch.sx - u_ch.ub
        user_funcs["u_lb"] = u_dch.sx - u_dch.ub

        pars = self.problem[3]
        xf = states[3]
        user_funcs["xf_ub"] = xf.sx - pars["fc_pv_prod"].sx

        # TODO
        # use sys.dt_h / sys_pars.capacity * (sys_pars.eff / sys_pars.eff) to construct
        # expressions for state of charge with respective bounds.
        x = (
            # Initial state of charge
            np.ones(u_ch.sx.shape) * states[0].sx[0]
            # contribution from discharging
            - mycas.mtimes(
                np.tril(np.ones((u_dch.sx.shape[0], u_dch.sx.shape[0])))  # longer triangular matrix
                * self.sys_pars.dt_h
                / self.sys_pars.capacity
                / self.sys_pars.eff,
                u_dch.sx,
            )
            # contribution from charging
            + mycas.mtimes(
                np.tril(np.ones((u_ch.sx.shape[0], u_ch.sx.shape[0])))
                * self.sys_pars.dt_h
                / self.sys_pars.capacity
                * self.sys_pars.eff,
                u_ch.sx,
            )
        )
        user_funcs["x_ub"] = x - 1
        user_funcs["x_lb"] = -x

        return user_funcs


class OnOffPVBatSolver(object):
    def __init__(self, horizon, sys_id, parameters):
        self.horizon = horizon
        self.sys_id = sys_id

        warnings.warn("deprecated. Reactivate and transform to vector. if needed")

        self._create_problem(parameters)

    def _create_problem(self, pars):
        c_sup = pars["c_sup"]
        c_feed = pars["c_feed"]

        self.capacity = pars["capacity"]
        self.p_max_charging = 5.0
        self.eff = 0.999

        x_lb = 0.0
        x_ub = 1.0

        a_very_high_number = 270099

        states = []
        constraints = []

        self.pars = {}

        list_xk = [mycas.MySX(f"x_k_0_a_{self.sys_id}", x_lb, x_ub)]
        states += [list_xk[-1]]

        self.pars["x_init"] = mycas.MyPar(f"p_x_init_a_{self.sys_id}")
        constraints += [mycas.MyConstr(list_xk[0].sx - self.pars["x_init"].sx, 0, 0)]

        # Lagrange multiplier
        self.pars["lam_a"] = mycas.MyPar(f"lam_a_{self.sys_id}", self.horizon)
        # Forecast of uncontrollable load on site
        self.pars["fc_p_unc"] = mycas.MyPar(f"fc_p_unc_a_{self.sys_id}", self.horizon)

        obj = 0

        self._grid_vars = []

        # Formulate the NLP
        for k in range(self.horizon):
            # Battery Power (charging and discharging). AC power
            p_ch = mycas.MySX(f"p_ch_k_{k}_a_{self.sys_id}", 0, 1, discrete=True)
            states += [p_ch]
            p_dch = mycas.MySX(f"p_dch_k_{k}_a_{self.sys_id}", 0, 1, discrete=True)
            states += [p_dch]

            # Feed and supply
            xfk = mycas.MySX(f"xf_k_{k}_a_{self.sys_id}", 0, a_very_high_number)
            states += [xfk]
            xsk = mycas.MySX(f"xs_k_{k}_a_{self.sys_id}", 0, a_very_high_number)
            states += [xsk]

            # Supply - Feedin = Load - PV + discharge - charge
            constraints += [
                mycas.MyConstr(
                    xsk.sx
                    - xfk.sx
                    - self.pars["fc_p_unc"].sx[k]
                    + self.p_max_charging * (p_dch.sx - p_ch.sx),
                    0,
                    0,
                )
            ]

            if isinstance(c_sup, list):
                obj += xsk.sx * c_sup[k] - xfk.sx * c_feed
            else:
                obj += xsk.sx * c_sup - xfk.sx * c_feed

            # TODO as in the ADMM case this coupling variable needs some revision.
            y_g = mycas.MySX(f"y_g_{k}_a_{self.sys_id}", -a_very_high_number, a_very_high_number)
            states += [y_g]
            self._grid_vars += [y_g]

            # warnings.warn("lambda switched off")
            obj += self.pars["lam_a"].sx[k] * y_g.sx
            # Constrain the coupling variable to be larger than supply - feed-in
            # (works only for shaving peak load)
            constraints += [mycas.MyConstr(y_g.sx - xsk.sx + xfk.sx, 0, 0)]

            # ============== Local variable and Constraints
            # The naming here is a little bad: Like this, there are two variables named x_k_0
            # (initial one and after 1 time step)
            xk = mycas.MySX(f"x_k_{k+1}_a_{self.sys_id}", x_lb, x_ub)
            states += [xk]
            list_xk += [xk]

            # Constrain state evolution
            x_plus = list_xk[-2].sx + pars["dt_h"] / self.capacity * (
                self.p_max_charging * p_ch.sx * self.eff - self.p_max_charging * p_dch.sx / self.eff
            )
            # f_x_next(x0=list_xk[-2].sx, u=uk.sx)["x_next"]
            constraints += [mycas.MyConstr(x_plus - xk.sx, 0.0, 0.0)]

        # =================== Transform to casadi input ==============================

        self.nlp_solver = mycas.MyNLPSolver(obj, states, constraints, self.pars, solver="gurobi")
        self.problem = obj, states, constraints, self.pars

    def get_central_problem_contribution(self):
        return self.problem, self._grid_vars

    def solve(self, soc_init, fc_res_load, signal):
        par_values = {
            "x_init": [soc_init],
            "lam_a": signal["lambda"],
            "fc_p_unc": fc_res_load,
        }

        # print(par_values)

        # print(self.problem[0])
        self.nlp_solver.solve(par_values)
        ret_uk = self.nlp_solver.opt_vars(
            [f"p_ch_k_{k}_a_{self.sys_id}" for k in range(self.horizon)]
        ) - self.nlp_solver.opt_vars([f"p_dch_k_{k}_a_{self.sys_id}" for k in range(self.horizon)])
        ret_yg = self.nlp_solver.opt_vars([f"y_g_{k}_a_{self.sys_id}" for k in range(self.horizon)])

        if False:
            ret_x = self.nlp_solver.opt_vars(
                [f"x_k_{k + 1}_a_{self.sys_id}" for k in range(self.horizon)]
            )
            ret_xs = self.nlp_solver.opt_vars(
                [f"xs_k_{k}_a_{self.sys_id}" for k in range(self.horizon)]
            )
            ret_xf = self.nlp_solver.opt_vars(
                [f"xf_k_{k}_a_{self.sys_id}" for k in range(self.horizon)]
            )

            fig, ax = style.styled_plot(figsize="landscape", ylabel="power grid")
            # ax.plot(ret_x, label=f"SoC {self.sys_id}")
            ax.plot(ret_yg, label="yg")
            ax.plot(ret_xs, label="xs")
            ax.plot(ret_xf, label="xf")
            # ax.plot(fc_res_load, label="fc_res")
            ax.legend()
            fig.tight_layout()
            # plt.show()

        return ret_uk, ret_yg


# ++++++++++++++++++++++++ Experimental with non-smooth casadi functions +++++++++++++++++


def _get_pv_bat_non_smooth(
    sys_id: str, horizon: int, sys_pars: SysParsPVBat, opt_pars: OptParameters
) -> tuple[
    tuple[mycas.casadi.SX, list[mycas.MySX], list[mycas.MyConstr], dict[str, mycas.MyPar]],
    mycas.MySX,
]:
    """"""

    import casadi

    def _stage_cost(sys_pars: SysParsPVBat):
        z = casadi.SX.sym("z", 1)
        cost = casadi.if_else(z > 0, z * sys_pars.c_sup, z * sys_pars.c_feed)
        return casadi.Function("stage_cost", [z], [cost], ["z"], ["cost"])

    def _bat_evol_step(sys_pars: SysParsPVBat):
        x_prev = casadi.SX.sym("x_prev", 1)
        u_bat = casadi.SX.sym("u_bat", 1)
        p_net = casadi.if_else(u_bat > 0, u_bat * sys_pars.eff, u_bat / sys_pars.eff)
        d_soc = p_net * sys_pars.dt_h / sys_pars.capacity

        return casadi.Function(
            "x_evolve",
            [x_prev, u_bat],
            [x_prev + d_soc],
            ["x_prev", "u_bat"],
            ["x_next"],
        )

    def _soc_vec(sys_pars: SysParsPVBat, horizon):
        x_0 = casadi.SX.sym("x_0", 1)
        u_bat = casadi.SX.sym("u_bat", horizon)

        evol = _bat_evol_step(sys_pars)

        soc = [x_0]
        for k in range(horizon):
            soc += [evol(soc[-1], u_bat[k])]

        soc_vec = casadi.vertcat(*soc[1:])

        return casadi.Function("soc_vec", [x_0, u_bat], [soc_vec], ["x_0", "u_bat"], ["soc_vec"])

    # Grid variable
    y_g = mycas.MySX(f"y_g_a_{sys_id}", -np.inf, np.inf, horizon=horizon)
    states = [y_g]

    constraints = []
    pars = {}
    # Forecast of residual generation
    pars["fc_p_unc"] = mycas.MyPar(f"fc_p_unc_a_{sys_id}", horizon)
    # Constrain feed-in to pv production / residual generation.
    pars["fc_pv_prod"] = mycas.MyPar(f"fc_pv_prod_{sys_id}", horizon)
    pars["x_init"] = mycas.MyPar(f"p_x_init_a_{sys_id}")

    # Here it is restricted to residual generation to be more restrictive.
    constraints += [mycas.MyConstr(y_g.sx - pars["fc_pv_prod"].sx, -np.inf, 0)]

    obj: mycas.casadi.SX = 0
    stage_cost = _stage_cost(sys_pars)
    for k in range(horizon):
        obj += stage_cost(y_g.sx[k])

    # Charging power constraint
    constraints += [mycas.MyConstr(y_g.sx - pars["fc_p_unc"].sx - sys_pars.p_inv, -np.inf, 0)]
    # Discharging power constraint
    constraints += [mycas.MyConstr(-y_g.sx + pars["fc_p_unc"].sx - sys_pars.p_inv, -np.inf, 0)]

    soc_vec = _soc_vec(sys_pars, horizon)
    # Lower SoC bound
    constraints += [
        mycas.MyConstr(-soc_vec(pars["x_init"].sx, y_g.sx - pars["fc_p_unc"].sx), -np.inf, 0)
    ]
    # Upper SoC bound
    constraints += [
        mycas.MyConstr(soc_vec(pars["x_init"].sx, y_g.sx - pars["fc_p_unc"].sx) - 1, -np.inf, 0)
    ]

    obj += mycas.dot(
        soc_vec(pars["x_init"].sx, y_g.sx - pars["fc_p_unc"].sx),
        soc_vec(pars["x_init"].sx, y_g.sx - pars["fc_p_unc"].sx),
    )

    # =================== Transform to casadi input ==============================
    return ((obj, states, constraints, pars), y_g)


if __name__ == "__main__":
    pass
