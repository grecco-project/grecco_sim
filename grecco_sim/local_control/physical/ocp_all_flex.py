"""Formulate OCPs for an arbitrary combination of flexibilities."""

import numpy as np

from grecco_sim.local_control import mycas
from grecco_sim.local_control.physical import ocp_ev
from grecco_sim.util import logger, type_defs


HP_DISCRETE = True


def _add_bat(
    ocp: mycas.MyOCP, flex_id: str, sys_pars_bat: type_defs.SysParsPVBat, horizon: int
) -> tuple[mycas.MyOCP, mycas.casadi.SX]:
    """Add contribution of single battery to OCP.

    Returns OCP and the AC battery power variable to be added to the energy balance constraint.
    """
    # ========== Define variables for battery system ===========
    x_bat = mycas.MySX(f"{flex_id}_soc", sys_pars_bat.x_lb, sys_pars_bat.x_ub, horizon=horizon + 1)
    ocp.states += [x_bat]

    ocp.parameters[f"{flex_id}_soc_init"] = mycas.MyPar(f"{flex_id}_soc_init")
    ocp.constraints.append(
        mycas.MyConstr(
            x_bat.sx[0] - ocp.parameters[f"{flex_id}_soc_init"].sx,
            0,
            0,
            name=f"Initial state {flex_id}",
        )
    )

    # Battery Power (charging and discharging). AC power
    p_ch = mycas.MySX(f"{flex_id}_p_ch", 0, sys_pars_bat.p_inv, horizon=horizon)
    ocp.states += [p_ch]
    p_dch = mycas.MySX(f"{flex_id}_p_dch", 0, sys_pars_bat.p_inv, horizon=horizon)
    ocp.states += [p_dch]
    ocp.user_functions[f"h{abs(hash(flex_id))}_control"] = p_ch.sx - p_dch.sx
    # Battery
    # Constrain state evolution
    x_plus_bat = x_bat.sx[:-1] + sys_pars_bat.dt_h / sys_pars_bat.capacity * (
        p_ch.sx * sys_pars_bat.eff - p_dch.sx / sys_pars_bat.eff
    )
    # Constrain x_plus[k] to x_bat[k+1]
    ocp.constraints += [mycas.MyConstr(x_plus_bat - x_bat.sx[1:], 0.0, 0.0)]

    # ============= Terminal costs
    # Add terminal cost as bonus for energy in storage
    ocp.obj -= (
        x_bat.sx[-1]
        * sys_pars_bat.capacity
        * sys_pars_bat.dt_h
        * (sys_pars_bat.c_sup + sys_pars_bat.c_feed)
        * 0.5
    )

    return ocp, p_ch.sx - p_dch.sx


def _add_hp(
    ocp: mycas.MyOCP,
    flex_id: str,
    sys_pars_hp: type_defs.SysParsHeatPump,
    horizon: int,
    opt_pars: type_defs.OptParameters,
) -> tuple[mycas.MyOCP, mycas.casadi.SX]:
    """Add heat pump contributions to OCP.

    Returns altered OCP and electric power variable to be added to energy balance.
    """
    # ========== Define variables for heat pump ===========
    # House temperature
    temp = mycas.MySX(f"{flex_id}_temp", -np.inf, np.inf, horizon=horizon + 1)
    ocp.states += [temp]

    # Introduce slack constraint to ensure feasible satisfaction of temperature constraints.
    slack_th = mycas.MySX(f"{flex_id}_slack_th", 0, np.inf, horizon=horizon + 1)
    ocp.states += [slack_th]
    ocp.constraints += [mycas.MyConstr(temp.sx + slack_th.sx, sys_pars_hp.temp_min_heat, np.inf)]
    ocp.constraints += [mycas.MyConstr(-temp.sx + slack_th.sx, -sys_pars_hp.temp_max_heat, np.inf)]

    ocp.obj += mycas.dot(slack_th.sx, np.ones(slack_th.horizon) * opt_pars.slack_penalty_thermal)

    ocp.parameters[f"{flex_id}_temp_init"] = mycas.MyPar(f"{flex_id}_temp_init")
    ocp.constraints += [
        mycas.MyConstr(temp.sx[0] - ocp.parameters[f"{flex_id}_temp_init"].sx, 0, 0)
    ]

    # Outside temperature
    ocp.parameters[f"{flex_id}_temp_outside"] = mycas.MyPar(f"{flex_id}_temp_outside", horizon)
    ocp.parameters[f"{flex_id}_solar_heat_gain"] = mycas.MyPar(
        f"{flex_id}_solar_heat_gain", horizon
    )

    # =========== Define optimization variable time series ===============
    p_norm_hp = mycas.MySX(f"{flex_id}_p_norm_hp", 0.0, 1.0, horizon=horizon, discrete=HP_DISCRETE)
    ocp.states += [p_norm_hp]

    p_el_hp = mycas.MySX(f"{flex_id}_p_el_hp", 0, sys_pars_hp.p_max, horizon=horizon)
    ocp.states += [p_el_hp]
    ocp.constraints += [mycas.MyConstr(p_norm_hp.sx * sys_pars_hp.p_max - p_el_hp.sx, 0, 0.0)]

    # Make the control available through a user function
    ocp.user_functions[f"h{abs(hash(flex_id))}_control"] = p_el_hp.sx

    # Heatpump
    q_net = (
        ocp.parameters[f"{flex_id}_solar_heat_gain"].sx
        * sys_pars_hp.absorbance
        * sys_pars_hp.irradiance_area
        / 1000.0
        + sys_pars_hp.heat_rate * (ocp.parameters[f"{flex_id}_temp_outside"].sx - temp.sx[:-1])
        + sys_pars_hp.cop * p_el_hp.sx
    )
    # Constrain state evolution
    x_plus_hp = temp.sx[:-1] + q_net * sys_pars_hp.dt_h * 3600.0 / sys_pars_hp.thermal_mass
    ocp.constraints += [mycas.MyConstr(x_plus_hp - temp.sx[1:], 0.0, 0.0)]

    return ocp, p_el_hp.sx


def _add_ev(
    ocp: mycas.MyOCP,
    flex_id: str,
    sys_pars_ev: type_defs.SysParsEV,
    horizon: int,
    opt_pars: type_defs.OptParameters,
    charge_process: ocp_ev.ChargeProcess,
) -> tuple[mycas.MyOCP, mycas.casadi.SX]:
    # ========== Define variables for the electric vehicle ===========

    p_ev = mycas.MySX(f"{flex_id}_p_ev", 0, sys_pars_ev.p_inv, horizon=horizon)
    ocp.states += [p_ev]
    ocp.user_functions[f"h{abs(hash(flex_id))}_control"] = p_ev.sx

    x_ev = mycas.MySX(
        f"{flex_id}_soc",
        sys_pars_ev.x_lb,
        sys_pars_ev.x_ub,
        horizon=charge_process.horizon + 1,
    )
    ocp.states += [x_ev]

    ocp.constraints.append(
        mycas.MyConstr(x_ev.sx[0] - charge_process.soc_init, 0, 0, name="Initial state EV")
    )

    soc_missing = mycas.MySX(f"{flex_id}_soc_diff", 0.0, np.inf)
    ocp.states += [soc_missing]
    ocp.constraints += [
        mycas.MyConstr(
            x_ev.sx[charge_process.horizon] + soc_missing.sx,
            charge_process.soc_target,
            np.inf,
            "Missing SoC at end of CP",
        )
    ]
    ocp.obj += soc_missing.sx * sys_pars_ev.capacity * opt_pars.slack_penalty_missing_soc

    # Constrain charging power to be zero when the EV is not at the charger
    if charge_process.k_first > 0:
        ocp.constraints.append(
            mycas.MyConstr(p_ev.sx[: charge_process.k_first], 0, 0, "EV gone start")
        )
    if charge_process.k_last + 1 < horizon:
        ocp.constraints.append(
            mycas.MyConstr(p_ev.sx[charge_process.k_last + 1 :], 0, 0, "EV gone end")
        )

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
        ocp.constraints.append(mycas.MyConstr(x_plus_ev - x_ev.sx[k + 1], 0.0, 0.0))

    return ocp, p_ev.sx


def get_ocp(
    sys_id: str,
    horizon: int,
    sys_pars: dict[str, type_defs.SysParsPVBat | type_defs.SysParsHeatPump | type_defs.SysParsEV],
    opt_pars: type_defs.OptParameters,
    charge_processes: dict[str, ocp_ev.ChargeProcess],
) -> tuple[mycas.MyOCP, mycas.MySX]:
    """Get a combined OCP of an arbitrary flexibility configuration.


    Pass a dictionary mapping flex_ids to respective system parameters.
    """

    # Initialize empty OCP
    ocp = mycas.MyOCP(0.0, [], [], {}, {})

    # Variable to combine AC power at grid_connection coming from battery
    ac_power_bats_sx = 0.0
    ac_power_hp_ev_sx = 0.0

    # Flag if a flexibility is actually part of the OCP
    has_flex = False

    for flex_id, sys_pars in sys_pars.items():
        match sys_pars.system:
            case "load":
                continue
            case "pv":
                continue
            case "bat":
                ocp, ac_power = _add_bat(ocp, flex_id, sys_pars, horizon)
                ac_power_bats_sx += ac_power
                has_flex = True
                continue
            case "ev":
                if flex_id not in charge_processes:
                    continue
                ocp, ac_power = _add_ev(
                    ocp, flex_id, sys_pars, horizon, opt_pars, charge_processes[flex_id]
                )
                ac_power_hp_ev_sx += ac_power
                has_flex = True
                continue
            case "hp":
                ocp, ac_power = _add_hp(ocp, flex_id, sys_pars, horizon, opt_pars)
                ac_power_hp_ev_sx += ac_power
                has_flex = True
                continue
            case _:
                raise ValueError(
                    f"Unknown flexibility type for flex_id '{flex_id}': {sys_pars.system}"
                )

    if not has_flex:
        return ocp, 0.0
    # ============= overall system variables =============

    # Forecast of uncontrollable load on site
    ocp.parameters["fc_p_unc"] = mycas.MyPar(f"{sys_id}_fc_p_unc", horizon)

    # Feed and supply
    xf = mycas.MySX(f"{sys_id}_xf", 0, np.inf, horizon=horizon)
    ocp.states.append(xf)
    xs = mycas.MySX(f"{sys_id}_xs", 0, np.inf, horizon=horizon)
    ocp.states.append(xs)

    # Handle electricity prices as a parameter time series
    ocp.parameters["c_sup"] = mycas.MyPar(f"{sys_id}_c_sup", hor=horizon)
    ocp.parameters["c_feed"] = mycas.MyPar(f"{sys_id}_c_feed", hor=horizon)
    ocp.obj += mycas.dot(xs.sx, ocp.parameters["c_sup"].sx) - mycas.dot(
        xf.sx, ocp.parameters["c_feed"].sx
    )

    # Grid variable
    y_g = mycas.MySX(f"{sys_id}_y_g", -np.inf, np.inf, horizon=horizon)
    ocp.states.append(y_g)

    # Constrain the coupling variable to be supply - feed in
    ocp.constraints += [mycas.MyConstr(y_g.sx - xs.sx + xf.sx, 0, 0)]

    # Grid Power = Load - PV + p_hp + Ac power battery + EV Charge power
    ocp.constraints += [
        mycas.MyConstr(
            y_g.sx - ocp.parameters["fc_p_unc"].sx - ac_power_bats_sx - ac_power_hp_ev_sx, 0, 0
        )
    ]

    # Constrain feed-in to pv production / residual generation.
    # Forecast of residual generation: this forecast is >= 0
    ocp.parameters["fc_pv_prod"] = mycas.MyPar(f"fc_pv_prod_{sys_id}", horizon)
    # Here it is restricted to residual generation to be more restrictive.
    ocp.constraints += [
        mycas.MyConstr(
            xf.sx - ocp.parameters["fc_pv_prod"].sx,
            -np.inf,
            0,
            name="No discharging into grid",
        )
    ]

    # Make a constraint that prevents charging the battery from grid.
    # This constraint means: p_bat < residual generation
    ocp.constraints.append(
        mycas.MyConstr(
            ac_power_bats_sx - ocp.parameters["fc_pv_prod"].sx,
            -np.inf,
            0.0,
            name="No battery charging from grid.",
        )
    )

    return ocp, y_g
