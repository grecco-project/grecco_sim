from collections import Counter

import numpy as np

from grecco_sim.local_control import mycas
from grecco_sim.util import sig_types
from grecco_sim.util import type_defs

from grecco_sim.local_control.physical import common, ocp_ev
from grecco_sim.local_control.solver import common as solver_commons


def _add_previous_cost_contributions(
    ocp: mycas.MyOCP,
    grid_var,
    opt_pars: type_defs.OptParameters,
    signal_lengths: list[int],
) -> mycas.MyOCP:
    for signal_len, count in sorted(Counter(signal_lengths).items()):
        for number in range(count):
            ocp.parameters[f"lam_{signal_len}_{number}"] = mycas.MyPar(
                f"lam_{signal_len}_{number}", signal_len
            )
            ocp.parameters[f"ref_grid_power_{signal_len}_{number}"] = mycas.MyPar(
                f"ref_grid_power_{signal_len}_{number}", signal_len
            )

            ocp.obj += mycas.dot(
                ocp.parameters[f"lam_{signal_len}_{number}"].sx, grid_var.sx[:signal_len]
            )
            ocp.obj += (
                opt_pars.rho
                / 2.0
                * mycas.dot(
                    grid_var.sx[:signal_len]
                    - ocp.parameters[f"ref_grid_power_{signal_len}_{number}"].sx,
                    grid_var.sx[:signal_len]
                    - ocp.parameters[f"ref_grid_power_{signal_len}_{number}"].sx,
                )
            )

    return ocp


def _create_problem_admm(
    sys_id,
    horizon,
    model_configuration,
    sys_pars: dict[str, type_defs.SysPars],
    opt_pars: type_defs.OptParameters,
    charge_process: ocp_ev.ChargeProcess | None = None,
    signal_lengths: list[int] = [],
) -> mycas.MyNLPSolver:

    ocp, grid_var = common.get_plain_ocp(
        sys_id, horizon, model_configuration, sys_pars, opt_pars, charge_process
    )

    ocp = _add_previous_cost_contributions(ocp, grid_var, opt_pars, signal_lengths)
    ocp = _add_admm_terms(sys_id, horizon, ocp, grid_var, opt_pars)
    return mycas.MyNLPSolver(ocp, opt_pars.solver_name)


def _add_admm_terms(
    sys_id: str,
    horizon: int,
    ocp: mycas.MyOCP,
    grid_var: mycas.MySX,
    opt_pars: type_defs.OptParameters,
):

    ocp.parameters["ref_grid_power"] = mycas.MyPar(f"ref_grid_power_{sys_id}", horizon)
    ocp.parameters["lam_a"] = mycas.MyPar(f"lam_a_{sys_id}", horizon)

    ocp.user_functions["grid_var"] = grid_var.sx

    ocp.obj += mycas.dot(ocp.parameters["lam_a"].sx, grid_var.sx)

    # Add cost terms from Lagrangian augmentation.
    diff = grid_var.sx - ocp.parameters["ref_grid_power"].sx
    ocp.obj += opt_pars.rho / 2.0 * mycas.dot(diff, diff)

    return ocp


def _add_admm_parameters(
    par_values: dict,
    signal: sig_types.SignalType,
    prev_signals: list[sig_types.SecondOrderSignal] = [],
):
    """Fill parameter values and solve problem."""
    par_values["lam_a"] = signal.mul_lambda
    par_values["ref_grid_power"] = signal.res_power_set

    signals_seen: dict[int, int] = {}

    for signal in prev_signals:
        number = signals_seen.get(signal.signal_len, 0)
        signals_seen[signal.signal_len] = number + 1
        par_values[f"lam_{signal.signal_len}_{number}"] = signal.mul_lambda
        par_values[f"ref_grid_power_{signal.signal_len}_{number}"] = signal.res_power_set

    return par_values

    # TODO reactivate for ALADIN
    # def get_local_gradients(self):

    #     # Doesn't make sense: returns state bounds
    #     # multipliers = self.nlp_solver.opt_multipliers([f"y_g_{k}_a_{self.sys_id}" for k in range(self.horizon)])
    #     # grads = self.nlp_solver.opt_gradient("local_objective", f"y_g_a_{self.sys_id}")
    #     # TODO: This calculation is wrong. must be either grad_f or grad_s depending on which variable is active

    #     xf = self.nlp_solver.opt_vector(f"xf_a_{self.sys_id}")
    #     xs = self.nlp_solver.opt_vector(f"xs_a_{self.sys_id}")

    #     grads_f = self.nlp_solver.opt_gradient("local_objective", f"xf_a_{self.sys_id}")[0, :]
    #     grads_s = self.nlp_solver.opt_gradient("local_objective", f"xs_a_{self.sys_id}")[0, :]

    #     return -grads_f * (xf > xs) + grads_s * (xs >= xf)

    # def get_active_constraint_jacobian(self) -> np.ndarray:
    # """Get the gradient of the constraints that are active."""
    # jac = []
    # for constr in ["u_lb", "u_ub", "x_ub", "x_lb", "xf_ub"]:
    #     constr_val = self.nlp_solver.get_custom_function_value(constr)
    #     constr_val = constr_val[:, 0]  # take all lines but only first column (value is scalar)

    #     p_ch = self.nlp_solver.opt_vector(f"p_ch_a_{self.sys_id}")
    #     p_dc = self.nlp_solver.opt_vector(f"p_dch_a_{self.sys_id}")

    #     # select the active gradient
    #     constr_gradient = self.nlp_solver.opt_gradient(constr, f"p_ch_a_{self.sys_id}") * (
    #         p_ch >= p_dc
    #     ) - self.nlp_solver.opt_gradient(constr, f"p_dch_a_{self.sys_id}") * (p_ch < p_dc)

    #     jac += constr_gradient[np.where(constr_val >= 0)].tolist()

    # return np.array(jac)


_created_solvers: dict[str, solver_commons.SolverWrapper] = {}


def get_solver(
    horizon: int,
    signal_lengths: list[int],
    sys_id: str,
    sys_parameters: dict[str, type_defs.SysPars],
    controller_pars: type_defs.OptParameters,
) -> solver_commons.SolverWrapper:
    """Make sure that a solver with a certain configuration is reused."""

    solver_hash = f"{sys_id}_{horizon}_" + "_".join(
        [f"{n}_{count}" for n, count in sorted(Counter(signal_lengths).items())]
    )

    if solver_hash in _created_solvers:
        return _created_solvers[solver_hash]
    else:
        model_conf = solver_commons.dict_to_model_conf(sys_parameters)

        solver_class = (
            solver_commons.SOLVER_CLASSES[model_conf]
            if model_conf in solver_commons.SOLVER_CLASSES
            else solver_commons.LocalSolverGradientDescentArbitrary
        )
        solver = solver_class(
            horizon,
            sys_id,
            sys_parameters,
            controller_pars,
            _add_admm_terms,
            _add_admm_parameters,
        )

        if controller_pars.buffer_n_solvers is None or controller_pars.buffer_n_solvers > len(
            _created_solvers
        ):
            _created_solvers[solver_hash] = solver

        return solver
