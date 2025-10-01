"""Technical contributions for thermal system OCP."""

import numpy as np

from grecco_sim.local_control import mycas
from grecco_sim.local_control.solver import local_solver_base
from grecco_sim.util import type_defs


class SolverThermalSystem(local_solver_base.LocalSolverBase):
    """
    Solver for local system with only local objectives and constraints.

    This is a pure interface class. Function implementation below.
    """

    def __init__(
        self,
        horizon: int,
        sys_id: str,
        sys_pars: type_defs.SysParsHeatPump,
        opt_pars: type_defs.OptParameters,
    ) -> None:
        self._create_problem(horizon, sys_id, sys_pars, opt_pars)

    def _create_problem(
        self,
        horizon: int,
        sys_id: str,
        sys_pars: type_defs.SysParsHeatPump,
        opt_pars: type_defs.OptParameters,
    ) -> None:
        self._problem, self._grid_var = _get_therm_system_contributions(
            sys_id, horizon, sys_pars, opt_pars
        )

    def get_central_problem_contribution(self) -> tuple[mycas.MyOCP, mycas.MySX]:
        """Return the systems contributions to a combined OCP"""
        return self._problem, self._grid_var


def _get_therm_system_contributions(
    sys_id: str,
    horizon: int,
    sys_pars: type_defs.SysParsHeatPump,
    opt_pars: type_defs.OptParameters,
) -> tuple[mycas.MyOCP, mycas.MySX]:
    """Get the technical optimization problem for the OCP of a thermal system.

    Returns (objective, states, constraints, parameters), grid variables.
    All as MySX
    """

    states = []
    constraints = []
    pars = {}
    obj = 0

    # House temperature
    temp = mycas.MySX(f"x_a_{sys_id}", -np.inf, np.inf, horizon=horizon + 1)
    states += [temp]

    # Introduce slack constraint to ensure feasible satisfaction of temperature constraints.
    slack_th = mycas.MySX(f"slack_th_a_{sys_id}", 0, np.inf, horizon=horizon + 1)
    states += [slack_th]
    constraints += [mycas.MyConstr(temp.sx + slack_th.sx, sys_pars.temp_min_heat, np.inf)]
    constraints += [mycas.MyConstr(-temp.sx + slack_th.sx, -sys_pars.temp_max_heat, np.inf)]

    obj += mycas.dot(slack_th.sx, np.ones(slack_th.horizon) * opt_pars.slack_penalty_thermal)

    pars["temp_init"] = mycas.MyPar(f"temp_init_a_{sys_id}")
    constraints += [mycas.MyConstr(temp.sx[0] - pars["temp_init"].sx, 0, 0)]

    # Outside temperature
    pars["temp_outside"] = mycas.MyPar(f"fc_temp_outside_{sys_id}", horizon)
    pars["solar_heat_gain"] = mycas.MyPar(f"fc_solar_heat_gain_{sys_id}", horizon)

    # Forecast of uncontrollable load on site
    pars["fc_p_unc"] = mycas.MyPar(f"fc_p_unc_a_{sys_id}", horizon)

    # =========== Define optimization variable time series ===============
    p_norm = mycas.MySX(f"_p_norm_{sys_id}", 0.0, 1.0, horizon=horizon, discrete=True)
    states += [p_norm]

    p_el_hp = mycas.MySX(f"p_el_hp_{sys_id}", 0, sys_pars.p_max, horizon=horizon)
    states += [p_el_hp]
    constraints += [mycas.MyConstr(p_norm.sx * sys_pars.p_max - p_el_hp.sx, 0, 0.0)]

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

    # Formulate the NLP
    for k in range(horizon):
        # Supply - Feedin = Load - PV + p_hp
        constraints += [
            mycas.MyConstr(xs.sx[k] - xf.sx[k] - pars["fc_p_unc"].sx[k] - p_el_hp.sx[k], 0, 0)
        ]

        # Constrain the coupling variable to be supply - feed in
        constraints += [mycas.MyConstr(y_g.sx[k] - xs.sx[k] + xf.sx[k], 0, 0)]

        q_net = (
            pars["solar_heat_gain"].sx[k] * sys_pars.absorbance * sys_pars.irradiance_area / 1000.0
            + sys_pars.heat_rate * (pars["temp_outside"].sx[k] - temp.sx[k])
            + sys_pars.cop * p_el_hp.sx[k]
        )
        # Constrain state evolution
        x_plus = temp.sx[k] + q_net * sys_pars.dt_h * 3600.0 / sys_pars.thermal_mass
        constraints += [mycas.MyConstr(x_plus - temp.sx[k + 1], 0.0, 0.0)]

    # =================== Transform to casadi input ==============================

    return (mycas.MyOCP(obj, states, constraints, pars), y_g)
