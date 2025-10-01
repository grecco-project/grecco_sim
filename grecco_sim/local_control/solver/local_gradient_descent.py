from grecco_sim.local_control import mycas
from grecco_sim.local_control.physical import common as phys_commons
from grecco_sim.local_control.solver import common as solver_commons
from grecco_sim.util import sig_types
from grecco_sim.local_control.physical import ocp_ev
from grecco_sim.util import type_defs


def _add_gradient_descent_terms(
    sys_id: str,
    horizon: int,
    ocp: mycas.MyOCP,
    grid_var: mycas.MySX,
    opt_pars: type_defs.OptParameters,
):

    # Add cost term from Lagrangian augmentation.
    ocp.parameters["lam_a"] = mycas.MyPar(f"lam_a_{sys_id}", horizon)
    ocp.obj += mycas.dot(ocp.parameters["lam_a"].sx, grid_var.sx)

    ocp.user_functions["grid_var"] = grid_var.sx
    return ocp


def _create_problem_gradient_descent(
    sys_id,
    horizon,
    model_configuration: tuple[str],
    sys_pars: dict[str, type_defs.SysPars],
    opt_pars: type_defs.OptParameters,
    charge_processes: dict[str, ocp_ev.ChargeProcess],
) -> mycas.MyNLPSolver:

    ocp, grid_var = phys_commons.get_plain_ocp(
        sys_id, horizon, model_configuration, sys_pars, opt_pars, charge_processes
    )
    ocp = _add_gradient_descent_terms(sys_id, horizon, ocp, grid_var)
    return mycas.MyNLPSolver(ocp, opt_pars.solver_name)


def _add_parameters_gradient_descent(par_values: dict, signal: sig_types.SignalType):
    par_values["lam_a"] = signal.mul_lambda

    return par_values


_created_solvers: dict[str, solver_commons.LocalSolverGradientDescent] = {}


def get_solver(
    horizon: int,
    sys_id: str,
    sys_parameters: dict[str, type_defs.SysPars],
    controller_pars: type_defs.OptParameters,
) -> solver_commons.SolverWrapper:
    """Make sure that a solver with a certain configuration is reused."""

    solver_hash = f"{sys_id}_{horizon}_{controller_pars.solver_name}"

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
            _add_gradient_descent_terms,
            _add_parameters_gradient_descent,
        )

        if controller_pars.buffer_n_solvers is None or controller_pars.buffer_n_solvers > len(
            _created_solvers
        ):
            _created_solvers[solver_hash] = solver

        return solver
