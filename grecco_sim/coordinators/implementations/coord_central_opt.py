"""
This module provides a centralized controller with the goal of limiting power to a peak value.

"""

import numpy as np

from grecco_sim.coordinators import coord_interface
from grecco_sim.local_control import mycas
from grecco_sim.local_control.physical import ocp_all_flex
from grecco_sim.local_control.physical import common as phys_common
from grecco_sim.local_control.solver import common as solver_common
from grecco_sim.util import sig_types
from grecco_sim.util import type_defs
from grecco_sim.util import helper


class CentralOptimizationCoordinator(coord_interface.CoordinatorInterface):
    """
    This controller is a completely centralized controller for the case of limiting grid power to a certain value
    It solves an OCP at every timestep and passes the controls back to the connected nodes.
    """

    coord_name = "Central Control"

    def __init__(
        self,
        controller_pars: type_defs.OptParameters,
        sys_ids: list[str],
        grid: type_defs.GridDescription,
    ):
        super().__init__(controller_pars.horizon)
        self.current_signal_time = -1

        self.grid = grid
        self.opt_pars = controller_pars

        self.constraint_value = 1e10

    def has_converged(self, time_index):
        return time_index == self.current_signal_time

    def get_signals(
        self, futures: dict[str, type_defs.LocalFuture]
    ) -> dict[str, sig_types.SignalType]:

        current_horizon = list(futures.values())[0].horizon

        flex_futures, inflex_futures, flex_sum, inflex_sum = helper.get_flex_and_inflex(futures)

        if (inflex_sum > self.grid.p_lim).any() and True:
            print(inflex_sum)

        flex_ocps = {}
        par_values = {}
        all_sys_pars = {}
        # Collect sys_ids of those systems that get inflexible because there is no EV charge process
        poppables = []

        for sys_id, future in flex_futures.items():
            sys_pars = future._meta["_model_pars"]
            all_sys_pars[sys_id] = sys_pars

            charge_processes = {
                flex_id: phys_common.get_charge_process(
                    future._meta["state"], self.horizon, ev_pars
                )
                for flex_id, ev_pars in sys_pars.items()
                if ev_pars.system == "ev"
            }
            charge_processes = {
                key: val for key, val in charge_processes.items() if val is not None
            }

            ocp, grid_var = ocp_all_flex.get_ocp(
                sys_id, current_horizon, sys_pars, self.opt_pars, charge_processes
            )

            par_values[sys_id] = solver_common.get_par_values_arbitrary_naming(
                sys_pars, future._meta["state"], future._meta["fc"]
            )

        # for sys_id in poppables:
        # inflex_sum += flex_futures[sys_id].yg
        # inflex_futures[sys_id] = flex_futures.pop(sys_id)

        nlp_solver = _combine(flex_ocps, inflex_sum, self.grid, self.opt_pars)

        res = _solve(nlp_solver, par_values, current_horizon)
        if False:
            import matplotlib.pyplot as plt

            plt.plot(inflex_sum)
            flex_sol = np.array([ag_res["yk"] for ag_res in res.values()]).sum(axis=0)
            plt.plot(flex_sol)
            plt.plot(flex_sol + inflex_sum)
            plt.show()

        self.constraint_value = (
            np.abs(np.array([ag_res["yk"] for ag_res in res.values()]).sum(axis=1)).sum()
            / self.horizon
        )

        self.current_signal_time = list(futures.values())[0]._meta["state"]["k"]

        ret_flex = {sys_id: sig_types.DirectControlSignal(res[sys_id]["uk"]) for sys_id in res}
        ret_inflex = {
            sys_id: sig_types.DirectControlSignal(
                {f"{sys_id}_pv": np.zeros(current_horizon), sys_id: np.zeros(current_horizon)}
            )
            for sys_id in inflex_futures
        }

        return {**ret_flex, **ret_inflex}

    def _get_parameter_values(self, forecast: type_defs.Forecast, state, model_configuration):
        par_values = {"fc_p_unc": forecast.fc_res_load}

        par_values["c_sup"] = forecast.add_fc["c_sup"]
        par_values["c_feed"] = forecast.add_fc["c_feed"]

        if "bat" in model_configuration:
            par_values["x_bat_init"] = [state["soc"]]
            par_values["fc_pv_prod"] = -forecast.negative_part

        if "hp" in model_configuration:
            par_values["temp_init"] = state["temp"]
            par_values["temp_outside"] = forecast.add_fc["temp_outside"]
            par_values["solar_heat_gain"] = forecast.add_fc["solar_heat_gain"]

        if "ev" in model_configuration:
            par_values["ev_soc_init"] = state["ev_soc"]

        return par_values


def _combine(
    single_ocps: dict[str, tuple[mycas.MyOCP, mycas.MySX]],
    inflex_sum: np.ndarray,
    grid_descr: type_defs.GridDescription,
    opt_pars: type_defs.OptParameters,
) -> mycas.MyNLPSolver:

    states = []
    constraints = []
    pars = {}
    obj = 0.0
    user_functions = {}

    grid_combined = 0.0

    # Combine the individual OCPs
    for ag_tag, (ocp, grid_var) in single_ocps.items():

        grid_combined += grid_var.sx

        obj += ocp.obj
        states += ocp.states
        constraints += ocp.constraints

        pars.update(
            {f"{ag_tag}_{par_name}": ocp.parameters[par_name] for par_name in ocp.parameters}
        )

        # Merge the user functions for individual controls
        user_functions.update(
            {key + "_" + ag_tag: ufunc for key, ufunc in ocp.user_functions.items()}
        )
        user_functions[f"grid_var_{ag_tag}"] = grid_var.sx

    coeff_slack = 100.0

    slack_ub = mycas.MySX("s_g_ub", 0.0, np.inf, horizon=len(inflex_sum))
    slack_lb = mycas.MySX("s_g_lb", 0.0, np.inf, horizon=len(inflex_sum))
    states += [slack_ub, slack_lb]

    constraints += [
        # Upper bound of grid
        mycas.MyConstr(
            grid_combined + inflex_sum - slack_ub.sx, -np.inf, grid_descr.p_lim, "Grid upper"
        ),
        # lower bound
        mycas.MyConstr(
            grid_combined + inflex_sum + slack_lb.sx, -grid_descr.p_lim, np.inf, "Grid lower"
        ),
    ]
    obj += mycas.dot(slack_ub.sx, np.ones(slack_ub.horizon) * coeff_slack)
    obj += mycas.dot(slack_lb.sx, np.ones(slack_lb.horizon) * coeff_slack)

    central_ocp = mycas.MyOCP(obj, states, constraints, pars, user_functions)
    return mycas.MyNLPSolver(central_ocp, solver=opt_pars.solver_name)


def _solve(solver: mycas.MyNLPSolver, par_sets: dict[str, dict], horizon):

    par_values = {
        f"{ag_tag}_{par_name}": par_sets[ag_tag][par_name]
        for ag_tag in par_sets
        for par_name in par_sets[ag_tag]
    }

    solver.solve(par_values)

    ret_dict = {}
    for ag_tag in par_sets:
        ret_dict[ag_tag] = _get_agent_solution(solver, ag_tag, model_confs[ag_tag], horizon)

    return ret_dict


def _get_agent_solution(
    solver: mycas.MyNLPSolver, sys_id: str, model_conf: tuple[str], horizon: int
):
    """Get solutions for the flexible system and map to controlled system."""

    ret = {sys_id: np.zeros(horizon)}

    if "pv" in model_conf:
        ret[f"{sys_id}_pv"] = np.zeros(horizon)

    if "bat" in model_conf:
        bat_control = solver.get_custom_function_value(f"u_bat_{sys_id}")
        ret[f"{sys_id}_bat"] = bat_control

    if "hp" in model_conf:
        hp_control = solver.opt_vector(f"p_el_hp_{sys_id}")
        ret[f"{sys_id}_hp"] = hp_control

    if "ev" in model_conf:
        ret[f"{sys_id}_ev"] = solver.get_custom_function_value(f"u_ev_{sys_id}")

    return {
        "uk": ret,
        "yk": solver.get_custom_function_value(f"grid_var_{sys_id}"),
        # "xterm": xterm
    }
