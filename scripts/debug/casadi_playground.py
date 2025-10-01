from grecco_sim.local_control import mycas
from grecco_sim.local_problems import local_gradient_descent, local_solver_pv_bat as pv_bat
from grecco_sim.util import type_defs


def play_around():

    x = mycas.MySX("x", 0, 1, horizon=2)
    states = [x]

    p = mycas.MyPar("p", 1)

    obj = x.sx[0] * x.sx[0] + x.sx[1] * x.sx[1]

    constraints = [mycas.MyConstr(x.sx[0] + x.sx[1] - p.sx[0], 0, 0)]
    parameters = {"p": p}
    solver = "sqpmethod"

    user_funcs = {"x": x.sx}

    solver = mycas.MyNLPSolver(
        obj, states, constraints, parameters, solver, user_functions=user_funcs
    )

    par_values = {"p": 2}
    init_guess = [0.3, 0.7]

    solver.solve(par_values, init_guess)

    print(solver.sol)
    jac = solver.nlp_solver.jacobian()
    sens = jac(**solver.solve_args, **{f"out_{k}": v for k, v in solver.sol.items()})

    solver.opt_par_sensitivity("x", "p")


def adavanced():
    horizon = 4

    opt_pars = type_defs.OptParameters(
        solver_name="gurobi", rho=1, mu=1.0, alpha=1.0, horizon=horizon
    )
    sys_pars = type_defs.SysParsPVBat(
        "ag_00", dt_h=0.25, c_sup=0.3, c_feed=0.1, init_soc=0.5, capacity=10.0, p_inv=5, eff=1.0
    )

    model = pv_bat.CasadiSolverLocalPVBatVectorized(4, "ag_00", sys_pars, opt_pars)
    (obj, states, constraints, pars), grid_var = model.get_central_problem_contribution()

    solver = "sqpmethod"

    solver = mycas.MyNLPSolver(obj, states, constraints, pars, solver)

    par_values = {"x_init": 2, "fc_p_unc": [0.0] * horizon, "fc_pv_prod": [0.0] * horizon}

    # init_guess = [0.3, 0.7]
    init_guess = None

    solver.solve(par_values, init_guess)

    print(solver.sol)
    jac = solver.nlp_solver.jacobian()
    sens = jac(**solver.solve_args, **{f"out_{k}": v for k, v in solver.sol.items()})

    solver.opt_par_sensitivity("x", "p")


def silence_gurobi():

    opt_pars = type_defs.OptParameters(solver_name="gurobi", rho=1, mu=1.0, alpha=1.0, horizon=20)
    sys_pars = {
        "bat": type_defs.SysParsPVBat(
            "ag_00", dt_h=0.25, c_sup=0.3, c_feed=0.1, init_soc=0.5, capacity=10.0, p_inv=5, eff=1.0
        ),
        "load": type_defs.SysParsLoad("load_id", dt_h=0.25, c_sup=0.3, c_feed=0.1),
        "pv": type_defs.SysParsPV("pv_id", dt_h=0.25, c_sup=0.3, c_feed=0.1),
    }

    ocp, grid_var = local_gradient_descent.get_plain_ocp(
        "test_id", 20, ("bat", "load", "pv"), sys_pars, opt_pars
    )

    solver = mycas.MyNLPSolver(ocp, "gurobi")

    par_values = {"fc_p_unc": [2.0] * 10 + [0.0] * 5 + [-2] * 5}
    par_values["x_bat_init"] = [0.5]
    par_values["fc_pv_prod"] = [0.0] * 15 + [2] * 5

    solver.solve(parameter_values=par_values)


if __name__ == "__main__":
    # play_around()
    # adavanced()
    silence_gurobi()
