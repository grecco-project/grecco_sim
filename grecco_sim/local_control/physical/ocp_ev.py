"""Module to provide the plain casadi model for the EV charger."""

import dataclasses
import matplotlib.pyplot as plt
import numpy as np

from grecco_sim.local_control import mycas
from grecco_sim.util import style, type_defs


@dataclasses.dataclass
class ChargeProcess(object):

    # Start charging at the first index
    k_first: int

    # last index at which charging is happening
    k_last: int

    # SOC at start and required at end
    soc_init: float
    soc_target: float

    # Time after the current opt horizon to finish CP
    time_after_h: float

    @property
    def horizon(self):
        """Return horizon of charge process."""
        return self.k_last - self.k_first + 1


def get_central_problem_contribution(
    horizon: int,
    sys_id: str,
    sys_pars: type_defs.SysParsEV,
    charge_process: ChargeProcess,
    opt_pars: type_defs.OptParameters,
) -> tuple[mycas.MyOCP, mycas.MySX]:
    """Get the central problem contribution of the plain EV charger model.

    Returns the OCP as a MyOcp object and the grid variable.
    """
    states = []
    constraints = []
    pars = {}
    obj: mycas.casadi.SX = 0
    user_functions = {}

    # constraints += [mycas.MyConstr(x.sx[horizon] - pars["x_init"].sx, 0, 0)]

    # Forecast of uncontrollable residual load on site (load - pv)
    pars["fc_p_unc"] = mycas.MyPar(f"fc_p_unc_a_{sys_id}", horizon)

    # =========== Define optimization variable time series ===============
    # Battery charging AC power
    p_ev = mycas.MySX(f"p_ch_a_{sys_id}", 0, sys_pars.p_inv, horizon=horizon)
    states += [p_ev]
    user_functions["u_ev"] = p_ev.sx

    # Feed and supply
    xf = mycas.MySX(f"xf_a_{sys_id}", 0, np.inf, horizon=horizon)
    states += [xf]
    xs = mycas.MySX(f"xs_a_{sys_id}", 0, np.inf, horizon=horizon)
    states += [xs]

    # Grid variable
    y_g = mycas.MySX(f"y_g_a_{sys_id}", -np.inf, np.inf, horizon=horizon)
    states += [y_g]

    # Handle electricity prices as a parameter time series
    pars["c_sup"] = mycas.MyPar(f"c_sup_a_{sys_id}", hor=horizon)
    pars["c_feed"] = mycas.MyPar(f"c_feed_a_{sys_id}", hor=horizon)
    obj += mycas.dot(xs.sx, pars["c_sup"].sx) - mycas.dot(xf.sx, pars["c_feed"].sx)

    # Supply - Feedin = Load - PV + charge
    constraints.append(
        mycas.MyConstr(
            xs.sx - xf.sx - pars["fc_p_unc"].sx - p_ev.sx,
            0,
            0,
        )
    )

    # Formulate the NLP
    for k in range(horizon):
        # Constrain the coupling variable to be supply - feed in
        constraints += [mycas.MyConstr(y_g.sx[k] - xs.sx[k] + xf.sx[k], 0, 0)]

    x_ev = mycas.MySX(
        f"x_ev_a_{sys_id}",
        sys_pars.x_lb,
        sys_pars.x_ub,
        horizon=charge_process.horizon + 1,
    )
    states += [x_ev]

    constraints.append(
        mycas.MyConstr(x_ev.sx[0] - charge_process.soc_init, 0, 0, name="Initial state")
    )

    # Terminal SOC is handled via penalty not constraint
    soc_missing = mycas.MySX(f"soc_diff_ev_{sys_id}", 0.0, np.inf)
    states += [soc_missing]
    constraints += [
        mycas.MyConstr(
            x_ev.sx[charge_process.horizon] + soc_missing.sx,
            charge_process.soc_target,
            np.inf,
            "Missing SoC at end of CP",
        )
    ]
    obj += soc_missing.sx * sys_pars.capacity * opt_pars.slack_penalty_missing_soc

    for k in np.arange(charge_process.horizon):
        # Constrain state evolution
        x_plus = (
            x_ev.sx[k]
            + sys_pars.dt_h / sys_pars.capacity * p_ev.sx[k + charge_process.k_first] * sys_pars.eff
        )
        # f_x_next(x0=list_xk[-2].sx, u=uk.sx)["x_next"]
        constraints.append(mycas.MyConstr(x_plus - x_ev.sx[k + 1], 0.0, 0.0))

    if charge_process.k_first > 0:
        constraints.append(mycas.MyConstr(p_ev.sx[: charge_process.k_first], 0, 0, "EV gone start"))
    if charge_process.k_last + 1 < horizon:
        constraints.append(
            mycas.MyConstr(p_ev.sx[charge_process.k_last + 1 :], 0, 0, "EV gone end")
        )

    # Add terminal cost as bonus for energy in storage
    obj -= (
        x_ev.sx[-1] * sys_pars.capacity * sys_pars.dt_h * (sys_pars.c_sup + sys_pars.c_feed) * 0.5
    )

    # =================== Transform to casadi input ==============================
    return mycas.MyOCP(obj, states, constraints, pars, user_functions), y_g


# class CasadiSolverLocalEV(object):
#     """Not maintained but kept for reference."""

#     def __init__(self, horizon):
#         self.dt_h = 0.25
#         self.capacity = 40.0
#         self.horizon = horizon

#         self.p_nom_ch = 11.0
#         self.eff = 0.9

#         self._create_problem()

#     def _create_problem(self):
#         c_sup = 0.2
#         c_feed = 0.1

#         x_lb = 0.0
#         x_ub = 1.0

#         a_very_high_number = 270099

#         f_x_next = mycas.function_x_next_ev_on_off(
#             self.p_nom_ch, self.dt_h, self.capacity, self.eff
#         )

#         states = []
#         constraints = []

#         self.pars = {}

#         list_xk = [mycas.MySX(f"x_k_0", x_lb, x_ub)]
#         states += [list_xk[-1]]

#         self.pars["x_init"] = mycas.MyPar("p_x_init")
#         constraints += [mycas.MyConstr(list_xk[0].sx - self.pars["x_init"].sx, 0, 0)]
#         self.pars["x_term"] = mycas.MyPar("p_x_term")

#         # Lagrange multiplier
#         self.pars["lam_a"] = mycas.MyPar("lam_a", self.horizon)
#         # Forecast of uncontrollable load on site
#         self.pars["fc_p_unc"] = mycas.MyPar("fc_p_unc", self.horizon)

#         self.pars["ev_connected"] = mycas.MyPar("ev_connected", self.horizon)

#         obj = 0

#         # Formulate the NLP
#         for k in range(self.horizon):
#             # New NLP variable for the control
#             # uk = mycas.MySX(f"u_k_{k}", 0, self.pars["ev_connected"].sx[k], discrete=True)
#             uk = mycas.MySX(f"u_k_{k}", 0, 1.0, discrete=True)
#             states += [uk]

#             constraints += [mycas.MyConstr(self.pars["ev_connected"].sx[k] - uk.sx, 0.0, 1.0)]

#             xfk = mycas.MySX(f"xf_k_{k}", 0, a_very_high_number)
#             states += [xfk]
#             xsk = mycas.MySX(f"xs_k_{k}", 0, a_very_high_number)
#             states += [xsk]

#             # Supply - Feedin = load_fc + charging_power
#             constraints += [
#                 mycas.MyConstr(
#                     xsk.sx - xfk.sx - self.pars["fc_p_unc"].sx[k] - self.p_nom_ch * uk.sx,
#                     0,
#                     0,
#                 )
#             ]

#             obj += xsk.sx * c_sup - xfk.sx * c_feed

#             y_g = mycas.MySX(f"y_g_{k}", -a_very_high_number, a_very_high_number)
#             states += [y_g]

#             obj += self.pars["lam_a"].sx[k] * y_g.sx
#             # Constrain the coupling variable to be larger than supply - feed-in (works only for shaving peak load)
#             constraints += [mycas.MyConstr(y_g.sx - xsk.sx + xfk.sx, 0, 0)]

#             # ============== Local variable and Constraints
#             # The naming here is a little bad: Like this, there are two variables named x_k_0
#             # (initial one and after 1 time step)
#             xk = mycas.MySX(f"x_k_{k+1}", x_lb, x_ub)
#             states += [xk]
#             list_xk += [xk]

#             # Constrain state evolution
#             x_plus = f_x_next(x0=list_xk[-2].sx, u=uk.sx)["x_next"]
#             constraints += [mycas.MyConstr(x_plus - xk.sx, 0.0, 0.0)]

#         constraints += [
#             mycas.MyConstr(list_xk[-1].sx - self.pars["x_term"].sx, 0, a_very_high_number)
#         ]

#         # =================== Transform to casadi input ==============================

#         self.nlp_solver = mycas.MyNLPSolver(obj, states, constraints, self.pars, solver="gurobi")

#     def solve(self, soc_init, soc_target, fc_res_load, signal, ev_connected):
#         par_values = {
#             "x_init": [soc_init],
#             "x_term": [soc_target],
#             "lam_a": signal["lambda"],
#             "fc_p_unc": fc_res_load,
#             "ev_connected": ev_connected,
#         }

#         # print(par_values)

#         self.nlp_solver.solve(par_values)
#         ret_uk = self.nlp_solver.opt_vars([f"u_k_{k}" for k in range(self.horizon)])
#         ret_yg = self.nlp_solver.opt_vars([f"y_g_{k}" for k in range(self.horizon)])

#         # fig, ax = style.styled_plot()
#         # ax.plot(ret_uk, label="uk")
#         # ax.plot(ret_yg, label="yg")
#         # ax.plot(fc_res_load, label="fc_res")
#         # ax.legend()
#         # plt.show()

#         return ret_uk, ret_yg
