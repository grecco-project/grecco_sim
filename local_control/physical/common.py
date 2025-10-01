import numpy as np
from grecco_sim.local_control import mycas
from grecco_sim.util import type_defs

from grecco_sim.local_control.physical import ocp_all_flex, ocp_ev
from grecco_sim.local_control.physical import ocp_thermal_system
from grecco_sim.local_control.physical import ocp_pv_bat
from grecco_sim.local_control.physical import ocp_multi_flex


def get_plain_ocp(
    sys_id,
    horizon,
    model_configuration: tuple[str, ...],
    sys_pars: dict[str, type_defs.SysPars],
    opt_pars: type_defs.OptParameters,
    charge_processes: dict[str, ocp_ev.ChargeProcess],
) -> tuple[mycas.MyOCP, mycas.MySX]:
    """
    Take the local problem and add the reference power parameter.
    Previous signals is cropped and assumed to start at the current time index.
    """
    match model_configuration:
        case ("bat", "load", "pv") | ("bat", "load"):
            bat_pars = [pars for pars in sys_pars.values() if pars.system == "bat"][0]
            if opt_pars.experimental_non_smooth:
                raise NotImplementedError("Reimplement nonsmooth PV bat if needed.")
                # (plain_obj, states, constraints, pars), grid_var = (
                # local_solver_pv_bat._get_pv_bat_non_smooth(sys_id, horizon, bat_pars, opt_pars)
                # )
            else:
                return ocp_pv_bat.plain_pv_bat_problem(horizon, sys_id, bat_pars)

        case ("hp", "load", "pv") | ("hp", "load"):
            hp_pars = [pars for pars in sys_pars.values() if pars.system == "hp"][0]
            single_controller = ocp_thermal_system.SolverThermalSystem(
                horizon, sys_id, hp_pars, opt_pars
            )
            return single_controller.get_central_problem_contribution()

        case ("bat", "hp", "load", "pv"):

            bat_pars = [pars for pars in sys_pars.values() if pars.system == "bat"][0]
            hp_pars = [pars for pars in sys_pars.values() if pars.system == "hp"][0]
            return ocp_multi_flex.get_bat_hp_ocp(sys_id, horizon, bat_pars, hp_pars, opt_pars)

        case ("bat", "ev", "load", "pv"):
            bat_pars = [pars for pars in sys_pars.values() if pars.system == "bat"][0]
            flex_id_ev, ev_pars = [
                (flex_id_ev, pars) for flex_id_ev, pars in sys_pars.items() if pars.system == "ev"
            ][0]
            return ocp_multi_flex.get_bat_ev_ocp(
                sys_id, horizon, bat_pars, ev_pars, charge_processes[flex_id_ev], opt_pars
            )
        case ("ev", "hp", "load", "pv"):
            flex_id_ev, ev_pars = [
                (flex_id_ev, pars) for flex_id_ev, pars in sys_pars.items() if pars.system == "ev"
            ][0]
            hp_pars = [pars for pars in sys_pars.values() if pars.system == "hp"][0]
            return ocp_multi_flex.get_hp_ev_ocp(
                sys_id, horizon, hp_pars, ev_pars, charge_processes[flex_id_ev], opt_pars
            )
        case ("ev", "load") | ("ev", "load", "pv"):
            flex_id_ev, ev_pars = [
                (flex_id_ev, pars) for flex_id_ev, pars in sys_pars.items() if pars.system == "ev"
            ][0]
            return ocp_ev.get_central_problem_contribution(
                horizon, sys_id, ev_pars, charge_processes[flex_id_ev], opt_pars
            )
        case _:
            return ocp_all_flex.get_ocp(sys_id, horizon, sys_pars, opt_pars, charge_processes)


def get_charge_process(state, horizon, ev_pars: type_defs.SysParsEV) -> ocp_ev.ChargeProcess | None:

    if not state["ev_connected"]:
        return None

    k_last = int(np.floor(state["ev_remaining_time_h"] * 4.0) - 1)
    if k_last <= -1:
        return None

    # Crop at simulation horizon
    k_last = min(horizon - 1, k_last)
    remaining_time = max(0, state["ev_remaining_time_h"] - horizon * ev_pars.dt_h)

    return ocp_ev.ChargeProcess(0, k_last, state["ev_soc"], state["ev_target_soc"], remaining_time)
