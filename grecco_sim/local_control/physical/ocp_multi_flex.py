"""The module to define the OCP for multiple flexibilities.

The equations are not autmatiocally updated if the respective one in the single flex OCP changes.
"""

import numpy as np

from grecco_sim.local_control import mycas
from grecco_sim.local_control.physical import ocp_ev
from grecco_sim.util import type_defs


HP_DISCRETE = True


def get_bat_hp_ocp(
    sys_id: str,
    horizon: int,
    sys_pars_bat: type_defs.SysParsPVBat,
    sys_pars_hp: type_defs.SysParsHeatPump,
    opt_pars: type_defs.OptParameters,
) -> tuple[mycas.MyOCP, mycas.MySX]:
    """Get the combined OCP of battery and heat pump."""

    states = []
    constraints = []
    pars = {}
    obj = 0
    user_functions = {}

    # ========== Define variables for battery system ===========
    x_bat = mycas.MySX(
        f"x_bat_a_{sys_id}", sys_pars_bat.x_lb, sys_pars_bat.x_ub, horizon=horizon + 1
    )
    states += [x_bat]

    pars["x_bat_init"] = mycas.MyPar(f"p_x_bat_init_a_{sys_id}")
    constraints.append(
        mycas.MyConstr(x_bat.sx[0] - pars["x_bat_init"].sx, 0, 0, name="Initial state")
    )

    # Battery Power (charging and discharging). AC power
    p_ch = mycas.MySX(f"p_ch_a_{sys_id}", 0, sys_pars_bat.p_inv, horizon=horizon)
    states += [p_ch]
    p_dch = mycas.MySX(f"p_dch_a_{sys_id}", 0, sys_pars_bat.p_inv, horizon=horizon)
    states += [p_dch]
    user_functions["u_bat"] = p_ch.sx - p_dch.sx

    # ========== Define variables for heat pump ===========
    # House temperature
    temp = mycas.MySX(f"x_a_{sys_id}", -np.inf, np.inf, horizon=horizon + 1)
    states += [temp]

    # Introduce slack constraint to ensure feasible satisfaction of temperature constraints.
    slack_th = mycas.MySX(f"slack_th_a_{sys_id}", 0, np.inf, horizon=horizon + 1)
    states += [slack_th]
    constraints += [mycas.MyConstr(temp.sx + slack_th.sx, sys_pars_hp.temp_min_heat, np.inf)]
    constraints += [mycas.MyConstr(-temp.sx + slack_th.sx, -sys_pars_hp.temp_max_heat, np.inf)]

    obj += mycas.dot(slack_th.sx, np.ones(slack_th.horizon) * opt_pars.slack_penalty_thermal)

    pars["temp_init"] = mycas.MyPar(f"temp_init_a_{sys_id}")
    constraints += [mycas.MyConstr(temp.sx[0] - pars["temp_init"].sx, 0, 0)]

    # Outside temperature
    pars["temp_outside"] = mycas.MyPar(f"fc_temp_outside_{sys_id}", horizon)
    pars["solar_heat_gain"] = mycas.MyPar(f"fc_solar_heat_gain_{sys_id}", horizon)

    # =========== Define optimization variable time series ===============
    p_norm_hp = mycas.MySX(f"_p_norm_hp_{sys_id}", 0.0, 1.0, horizon=horizon, discrete=HP_DISCRETE)
    states += [p_norm_hp]

    p_el_hp = mycas.MySX(f"p_el_hp_{sys_id}", 0, sys_pars_hp.p_max, horizon=horizon)
    states += [p_el_hp]
    constraints += [mycas.MyConstr(p_norm_hp.sx * sys_pars_hp.p_max - p_el_hp.sx, 0, 0.0)]

    # ============= overall system variables =============

    # Forecast of uncontrollable load on site
    pars["fc_p_unc"] = mycas.MyPar(f"fc_p_unc_a_{sys_id}", horizon)

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

    # Constrain the coupling variable to be supply - feed in
    constraints += [mycas.MyConstr(y_g.sx - xs.sx + xf.sx, 0, 0)]

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

    # Supply - Feedin = Load - PV + p_hp - discharge + charge
    constraints += [
        mycas.MyConstr(xs.sx - xf.sx - pars["fc_p_unc"].sx - p_el_hp.sx + p_dch.sx - p_ch.sx, 0, 0)
    ]

    # ===================== State evolutions ==========================

    # Formulate the NLP
    for k in range(horizon):

        # Heatpump
        q_net = (
            pars["solar_heat_gain"].sx[k]
            * sys_pars_hp.absorbance
            * sys_pars_hp.irradiance_area
            / 1000.0
            + sys_pars_hp.heat_rate * (pars["temp_outside"].sx[k] - temp.sx[k])
            + sys_pars_hp.cop * p_el_hp.sx[k]
        )
        # Constrain state evolution
        x_plus_hp = temp.sx[k] + q_net * sys_pars_hp.dt_h * 3600.0 / sys_pars_hp.thermal_mass
        constraints += [mycas.MyConstr(x_plus_hp - temp.sx[k + 1], 0.0, 0.0)]

        # Battery
        # Constrain state evolution
        x_plus_bat = x_bat.sx[k] + sys_pars_bat.dt_h / sys_pars_bat.capacity * (
            p_ch.sx[k] * sys_pars_bat.eff - p_dch.sx[k] / sys_pars_bat.eff
        )
        # f_x_next(x0=list_xk[-2].sx, u=uk.sx)["x_next"]
        constraints += [mycas.MyConstr(x_plus_bat - x_bat.sx[k + 1], 0.0, 0.0)]

    # ============= Terminal costs
    # Add terminal cost as bonus for energy in storage TODO for heat pump
    obj -= (
        x_bat.sx[-1]
        * sys_pars_bat.capacity
        * sys_pars_bat.dt_h
        * (sys_pars_bat.c_sup + sys_pars_bat.c_feed)
        * 0.5
    )

    # =================== Transform to casadi input ==============================
    return mycas.MyOCP(obj, states, constraints, pars, user_functions), y_g


def get_bat_ev_ocp(
    sys_id: str,
    horizon: int,
    sys_pars_bat: type_defs.SysParsPVBat,
    sys_pars_ev: type_defs.SysParsEV,
    charge_process: ocp_ev.ChargeProcess,
    opt_pars: type_defs.OptParameters,
) -> tuple[mycas.MyOCP, mycas.MySX]:
    """Get the combined OCP of battery and ev pump."""

    states = []
    constraints = []
    pars = {}
    obj = 0
    user_functions = {}

    # ========== Define variables for battery system ===========
    x_bat = mycas.MySX(
        f"x_bat_a_{sys_id}", sys_pars_bat.x_lb, sys_pars_bat.x_ub, horizon=horizon + 1
    )
    states += [x_bat]

    pars["x_bat_init"] = mycas.MyPar(f"p_x_bat_init_a_{sys_id}")
    constraints.append(
        mycas.MyConstr(x_bat.sx[0] - pars["x_bat_init"].sx, 0, 0, name="Initial state")
    )

    # Battery Power (charging and discharging). AC power
    p_ch = mycas.MySX(f"p_ch_a_{sys_id}", 0, sys_pars_bat.p_inv, horizon=horizon)
    states += [p_ch]
    p_dch = mycas.MySX(f"p_dch_a_{sys_id}", 0, sys_pars_bat.p_inv, horizon=horizon)
    states += [p_dch]

    user_functions["u_bat"] = p_ch.sx - p_dch.sx

    # ========== Define variables for the electric vehicle ===========

    p_ev = mycas.MySX(f"p_ch_a_{sys_id}", 0, sys_pars_ev.p_inv, horizon=horizon)
    states += [p_ev]
    user_functions["u_ev"] = p_ev.sx

    x_ev = mycas.MySX(
        f"x_ev_a_{sys_id}",
        sys_pars_ev.x_lb,
        sys_pars_ev.x_ub,
        horizon=charge_process.horizon + 1,
    )
    states += [x_ev]

    constraints.append(
        mycas.MyConstr(x_ev.sx[0] - charge_process.soc_init, 0, 0, name="Initial state EV")
    )

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
    obj += soc_missing.sx * sys_pars_ev.capacity * opt_pars.slack_penalty_missing_soc

    # Terminal SOC is handled via penalty not constraint -> TODO
    if charge_process.k_first > 0:
        constraints.append(mycas.MyConstr(p_ev.sx[: charge_process.k_first], 0, 0, "EV gone start"))
    if charge_process.k_last + 1 < horizon:
        constraints.append(
            mycas.MyConstr(p_ev.sx[charge_process.k_last + 1 :], 0, 0, "EV gone end")
        )

    # ============= overall system variables =============

    # Forecast of uncontrollable load on site
    pars["fc_p_unc"] = mycas.MyPar(f"fc_p_unc_a_{sys_id}", horizon)

    # Feed and supply
    xf = mycas.MySX(f"xf_a_{sys_id}", 0, np.inf, horizon=horizon)
    states += [xf]
    xs = mycas.MySX(f"xs_a_{sys_id}", 0, np.inf, horizon=horizon)
    states += [xs]

    # Grid variable
    y_g = mycas.MySX(f"y_g_a_{sys_id}", -np.inf, np.inf, horizon=horizon)
    states += [y_g]

    # Constrain the coupling variable to be supply - feed in
    constraints += [mycas.MyConstr(y_g.sx - xs.sx + xf.sx, 0, 0)]

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

    # Supply - Feedin = Load - PV + p_ev - discharge + charge
    constraints += [
        mycas.MyConstr(xs.sx - xf.sx - pars["fc_p_unc"].sx - p_ev.sx + p_dch.sx - p_ch.sx, 0, 0)
    ]

    # Handle electricity prices as a parameter time series
    pars["c_sup"] = mycas.MyPar(f"c_sup_a_{sys_id}", hor=horizon)
    pars["c_feed"] = mycas.MyPar(f"c_feed_a_{sys_id}", hor=horizon)
    obj += mycas.dot(xs.sx, pars["c_sup"].sx) - mycas.dot(xf.sx, pars["c_feed"].sx)

    # ===================== State evolutions ==========================

    for k in range(charge_process.horizon):
        # SOC of EV
        # Constrain state evolution
        x_plus_ev = (
            x_ev.sx[k]
            + sys_pars_ev.dt_h
            / sys_pars_ev.capacity
            * p_ev.sx[k + charge_process.k_first]
            * sys_pars_ev.eff
        )
        constraints.append(mycas.MyConstr(x_plus_ev - x_ev.sx[k + 1], 0.0, 0.0))

    # State evolutions for battery
    for k in range(horizon):
        # Constrain state evolution
        x_plus_bat = x_bat.sx[k] + sys_pars_bat.dt_h / sys_pars_bat.capacity * (
            p_ch.sx[k] * sys_pars_bat.eff - p_dch.sx[k] / sys_pars_bat.eff
        )
        # f_x_next(x0=list_xk[-2].sx, u=uk.sx)["x_next"]
        constraints += [mycas.MyConstr(x_plus_bat - x_bat.sx[k + 1], 0.0, 0.0)]

    # ============= Terminal costs
    # Add terminal cost as bonus for energy in storage TODO for heat pump
    obj -= (
        x_bat.sx[-1]
        * sys_pars_bat.capacity
        * sys_pars_bat.dt_h
        * (sys_pars_bat.c_sup + sys_pars_bat.c_feed)
        * 0.5
    )

    # =================== Transform to casadi input ==============================
    return mycas.MyOCP(obj, states, constraints, pars, user_functions), y_g


def get_hp_ev_ocp(
    sys_id: str,
    horizon: int,
    sys_pars_hp: type_defs.SysParsHeatPump,
    sys_pars_ev: type_defs.SysParsEV,
    charge_process: ocp_ev.ChargeProcess,
    opt_pars: type_defs.OptParameters,
) -> tuple[mycas.MyOCP, mycas.MySX]:
    """Get the combined OCP of EV and heat pump."""

    states = []
    constraints = []
    pars = {}
    obj = 0
    user_functions = {}

    # ========== Define variables for heat pump ===========
    # House temperature
    temp = mycas.MySX(f"x_a_{sys_id}", -np.inf, np.inf, horizon=horizon + 1)
    states += [temp]

    # Introduce slack constraint to ensure feasible satisfaction of temperature constraints.
    slack_th = mycas.MySX(f"slack_th_a_{sys_id}", 0, np.inf, horizon=horizon + 1)
    states += [slack_th]
    constraints += [mycas.MyConstr(temp.sx + slack_th.sx, sys_pars_hp.temp_min_heat, np.inf)]
    constraints += [mycas.MyConstr(-temp.sx + slack_th.sx, -sys_pars_hp.temp_max_heat, np.inf)]

    obj += mycas.dot(slack_th.sx, np.ones(slack_th.horizon) * opt_pars.slack_penalty_thermal)

    pars["temp_init"] = mycas.MyPar(f"temp_init_a_{sys_id}")
    constraints += [mycas.MyConstr(temp.sx[0] - pars["temp_init"].sx, 0, 0)]

    # Outside temperature
    pars["temp_outside"] = mycas.MyPar(f"fc_temp_outside_{sys_id}", horizon)
    pars["solar_heat_gain"] = mycas.MyPar(f"fc_solar_heat_gain_{sys_id}", horizon)

    # =========== Define optimization variable time series ===============
    p_norm_hp = mycas.MySX(f"_p_norm_hp_{sys_id}", 0.0, 1.0, horizon=horizon, discrete=HP_DISCRETE)
    states += [p_norm_hp]

    p_el_hp = mycas.MySX(f"p_el_hp_{sys_id}", 0, sys_pars_hp.p_max, horizon=horizon)
    states += [p_el_hp]
    constraints += [mycas.MyConstr(p_norm_hp.sx * sys_pars_hp.p_max - p_el_hp.sx, 0, 0.0)]

    # ========== Define variables for the electric vehicle ===========

    p_ev = mycas.MySX(f"p_ch_a_{sys_id}", 0, sys_pars_ev.p_inv, horizon=horizon)
    states += [p_ev]
    user_functions["u_ev"] = p_ev.sx

    x_ev = mycas.MySX(
        f"x_ev_a_{sys_id}",
        sys_pars_ev.x_lb,
        sys_pars_ev.x_ub,
        horizon=charge_process.horizon + 1,
    )
    states += [x_ev]

    constraints.append(
        mycas.MyConstr(x_ev.sx[0] - charge_process.soc_init, 0, 0, name="Initial state EV")
    )

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
    obj += soc_missing.sx * sys_pars_ev.capacity * opt_pars.slack_penalty_missing_soc

    # Terminal SOC is handled via penalty not constraint -> TODO
    if charge_process.k_first > 0:
        constraints.append(mycas.MyConstr(p_ev.sx[: charge_process.k_first], 0, 0, "EV gone start"))
    if charge_process.k_last + 1 < horizon:
        constraints.append(
            mycas.MyConstr(p_ev.sx[charge_process.k_last + 1 :], 0, 0, "EV gone end")
        )

    # ============= overall system variables =============

    # Forecast of uncontrollable load on site
    pars["fc_p_unc"] = mycas.MyPar(f"fc_p_unc_a_{sys_id}", horizon)

    # Feed and supply
    xf = mycas.MySX(f"xf_a_{sys_id}", 0, np.inf, horizon=horizon)
    states += [xf]
    xs = mycas.MySX(f"xs_a_{sys_id}", 0, np.inf, horizon=horizon)
    states += [xs]

    # Grid variable
    y_g = mycas.MySX(f"y_g_a_{sys_id}", -np.inf, np.inf, horizon=horizon)
    states += [y_g]

    # Constrain the coupling variable to be supply - feed in
    constraints += [mycas.MyConstr(y_g.sx - xs.sx + xf.sx, 0, 0)]

    # ==================== Main Energy Conservation Equation ===================
    # Supply - Feedin = Load - PV + p_ev - discharge + charge
    constraints += [
        mycas.MyConstr(xs.sx - xf.sx - pars["fc_p_unc"].sx - p_el_hp.sx - p_ev.sx, 0, 0)
    ]

    # Handle electricity prices as a parameter time series
    pars["c_sup"] = mycas.MyPar(f"c_sup_a_{sys_id}", hor=horizon)
    pars["c_feed"] = mycas.MyPar(f"c_feed_a_{sys_id}", hor=horizon)
    obj += mycas.dot(xs.sx, pars["c_sup"].sx) - mycas.dot(xf.sx, pars["c_feed"].sx)

    # ===================== State evolutions ==========================

    for k in range(charge_process.horizon):
        # SOC of EV
        # Constrain state evolution
        x_plus_ev = (
            x_ev.sx[k]
            + sys_pars_ev.dt_h
            / sys_pars_ev.capacity
            * p_ev.sx[k + charge_process.k_first]
            * sys_pars_ev.eff
        )
        constraints.append(mycas.MyConstr(x_plus_ev - x_ev.sx[k + 1], 0.0, 0.0))

    # State evolutions for battery
    for k in range(horizon):
        # Constrain state evolution
        # Heatpump
        q_net = (
            pars["solar_heat_gain"].sx[k]
            * sys_pars_hp.absorbance
            * sys_pars_hp.irradiance_area
            / 1000.0
            + sys_pars_hp.heat_rate * (pars["temp_outside"].sx[k] - temp.sx[k])
            + sys_pars_hp.cop * p_el_hp.sx[k]
        )
        # Constrain state evolution
        x_plus_hp = temp.sx[k] + q_net * sys_pars_hp.dt_h * 3600.0 / sys_pars_hp.thermal_mass
        constraints += [mycas.MyConstr(x_plus_hp - temp.sx[k + 1], 0.0, 0.0)]

    # ============= Terminal costs
    # Add terminal cost as bonus for energy in storage TODO for heat pump

    # =================== Transform to casadi input ==============================
    return mycas.MyOCP(obj, states, constraints, pars, user_functions), y_g
