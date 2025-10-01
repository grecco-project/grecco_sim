"""
Created by: Arne Gross
"""

import os
import dataclasses
import typing
import warnings

import casadi
import numpy as np

from grecco_sim.util import config, logger

if sim_config := config.get_config():
    os.environ["GRB_LICENSE_FILE"] = sim_config["gurobi_license_file"]
    os.environ["GUROBI_VERSION"] = sim_config["gurobi_version"]
    os.environ["GRB_OUTPUTFLAG"] = "0"


def dot(x: casadi.SX | np.ndarray, y: casadi.SX | np.ndarray) -> casadi.SX:
    """Provide the dot product to other modules.

    Product requires that x and y match in shape.
    Will reduce 2 vectors to scalar.
    """
    if isinstance(x, np.ndarray):
        x = casadi.DM(x)
    if isinstance(y, np.ndarray):
        y = casadi.DM(y)

    # assert x.size() == y.size(), f"Dimension of x ({x.shape}) does not match y dimension of y ({y.size()})"
    return casadi.dot(x, y)


def mtimes(x, y):
    """Wrap casadi matrix multiplication to provide to other modules."""
    return casadi.mtimes(x, y)


def to_cas_DM(x: np.ndarray):
    return casadi.DM(x)


class MySX(object):
    """Wrapper around casadi SX variable to remember how to use it."""

    def __init__(self, name, lb, ub, discrete: bool = False, horizon: int = 1):

        self.name = name
        self.lb = np.array([lb] * horizon)
        self.ub = np.array([ub] * horizon)
        self.horizon = horizon

        self.sx = casadi.SX.sym(name, horizon)
        self.discrete = discrete

    def __repr__(self):
        return self.name


class MyConstr(object):
    """Custom class to handle constraints in a better way than Casadi

    Args:
        object (_type_): _description_
    """

    def __init__(self, expression, lb, ub, name: str = ""):
        self.expr = expression
        self.lb = np.ones(self.expr.shape) * lb
        self.ub = np.ones(self.expr.shape) * ub

        self.name = name

    def __repr__(self):
        return self.name + ": " + str(self.expr)


class MyPar(object):
    """Wrapper around casadi to remember how to define a parameter."""

    def __init__(self, name, hor: int = 1):
        self.name = name
        self.horizon = hor
        self.sx = casadi.SX.sym(name, hor)


A_VERY_HIGH_NUMBER = np.inf


# ============= Modeling functions (for state evolution) ==================
def function_x_next_ev_on_off(p_nom_ch, dt_h, capacity, eff) -> typing.Callable:
    x = casadi.SX.sym("x", 1)
    u = casadi.SX.sym("u", 1)
    x_next = x + u * p_nom_ch * dt_h / capacity * eff
    return casadi.Function("xnext", [x, u], [x_next], ["x0", "u"], ["x_next"])


def function_loss_conversion(p_nom, p_a, u_a, r_a):
    """
    This is a loss function for a converter's conversion losses modeled as Schmidt2007
    :param p_nom: nominal power of inverter
    :param p_a:
    :param u_a:
    :param r_a:
    :return:
    """
    p_in = casadi.SX.sym("p_in", 1)
    loss = p_nom * (p_a + u_a * p_in / p_nom + r_a * p_in / p_nom * p_in / p_nom)
    return casadi.Function("loss", [p_in], [loss], ["p_in"], ["loss"])


@dataclasses.dataclass
class MyOCP(object):
    """Description of an OCP."""

    obj: casadi.SX
    states: list[MySX]
    constraints: list[MyConstr]
    parameters: dict[str, MyPar]
    user_functions: dict[str, casadi.SX] = dataclasses.field(default_factory=dict)

    @property
    def empty(self):
        """Return is there are no states in the OCP."""
        return len(self.states) == 0


# ============= Handling of states etc -> casadi problem ================
class MyNLPSolver:
    """
    Wrapper around the casadi solver framework, mainly to not have to remember the casadi usage.
    """

    solvers = {
        # NLP solver
        "ipopt": (casadi.nlpsol, "ipopt"),
        # MINLP Solver (slow)
        "bonmin": (casadi.nlpsol, "bonmin"),
        # MIQP solvers
        "cbc": (casadi.qpsol, "cbc"),
        # For Gurobi, the LD_LIBRARY_PATH environment variable has to be set to the lib folder
        "gurobi": (casadi.qpsol, "gurobi"),
        "osqp": (casadi.qpsol, "osqp"),
        "sqpmethod": (casadi.nlpsol, "sqpmethod"),
        "qpoases": (casadi.qpsol, "qpoases"),
    }

    solver_specific_settings = {
        "gurobi": {
            "OutputFlag": 0,
            "TimeLimit": 10,
        },
        "bonmin": {"iteration_limit": 1000},
        "osqp": {"verbose": False},
        # "qpoases": {"printLevel": "PL_NONE"}
        # "sqpmethod": {"qpsol": "qrqp"},
    }

    def __init__(self, ocp: MyOCP, solver="ipopt"):

        self.solver_name = solver
        self.nlp_solver = self._init_solver(
            ocp.obj, ocp.states, ocp.constraints, ocp.parameters, ocp.user_functions
        )
        # TBD handle obj, make it a function to be able to calculate the gradient

        self.sol = None
        self.solution_stats = {}

    def _init_solver(
        self,
        obj,
        states,
        constraints: list[MyConstr],
        parameters,
        user_functions: dict[str, casadi.SX],
    ):
        # Concatenate decision variables and constraint terms
        _x = casadi.vertcat(*[x.sx for x in states])
        _g = casadi.vertcat(*[g.expr for g in constraints])
        _p = casadi.vertcat(*[parameters[par_name].sx for par_name in parameters])
        # Create an NLP solver
        nlp_prob = {"f": obj, "x": _x, "g": _g, "p": _p}

        self._custom_functions: dict[str, casadi.Function] = {
            name: casadi.Function(name, [_x, _p], [user_functions[name]]) for name in user_functions
        }

        # MILP detection
        is_milp = any([x.discrete for x in states])
        discrete = list(np.concatenate([np.array([x.discrete] * x.horizon) for x in states]))

        n_discrete = np.array(discrete).sum()
        # if is_milp:
        #     print(f"OCP has {n_discrete} integer variables.")

        # Initialize solver options potentially with solver specific options
        solver_params = {
            "error_on_fail": False,
            "discrete": discrete,
            # "qpsol": "qrqp"
        }
        if self.solver_name in self.solver_specific_settings:
            solver_params[self.solver_name] = self.solver_specific_settings[self.solver_name]
        if self.solver_name == "qpoases":
            solver_params["printLevel"] = "none"

        if is_milp and self.solver_name == "ipopt":
            warnings.warn("Select solver IPOPT for a problem with integer variables")

        with logger.suppress_output():
            # with logger.show_output():
            nlp_solver = self.solvers[self.solver_name][0](
                "nlp_solver", self.solvers[self.solver_name][1], nlp_prob, solver_params
            )

        self.x_lb = np.concatenate([x.lb for x in states])
        self.x_ub = np.concatenate([x.ub for x in states])
        self.g_lb = np.concatenate([g.lb for g in constraints]) if constraints else []
        self.g_ub = np.concatenate([g.ub for g in constraints]) if constraints else []

        self.state_names = [x.name for x in states]
        self.state_dim = np.array([x.horizon for x in states]).sum()

        # Make a Table of content which segments of the w Vector belong to which state
        _idx_start = 0
        self.state_toc = {}
        for _x in states:
            self.state_toc[_x.name] = (_idx_start, _idx_start + _x.horizon)
            _idx_start += _x.horizon

        # Some remembering for parameters
        self.parameter_names = [par_name for par_name in parameters]
        # self.par_lengths = {par_name: par.sx.shape[0] for par_name, par in parameters.items()}

        _idx_start = 0
        self.par_toc = {}
        for p_name in parameters:
            _p = parameters[p_name]
            self.par_toc[p_name] = (_idx_start, _idx_start + _p.horizon)
            _idx_start += _p.horizon

        # Same for constraints
        _idx_start = 0
        self.constr_toc = {}
        for c in constraints:
            self.constr_toc[c.name] = (_idx_start, _idx_start + c.expr.shape[0])
            _idx_start += c.expr.shape[0]

        return nlp_solver

    def solve(self, parameter_values, init_guess=None):
        """Solve the OCP."""
        try:
            _parameters = casadi.vertcat(
                *[parameter_values[par_name] for par_name in self.parameter_names]
            )
        except KeyError:
            print(f"{parameter_values.keys()}, {self.parameter_names}")
            raise

        if init_guess is None:
            _init_guess = casadi.vertcat(*([0] * self.state_dim))
        else:
            assert len(init_guess) == self.state_dim
            _init_guess = casadi.vertcat(*init_guess)

        self.solve_args = dict(
            x0=_init_guess,
            lbx=self.x_lb,
            ubx=self.x_ub,
            lbg=self.g_lb,
            ubg=self.g_ub,
            p=_parameters,
        )

        self.sol = self._pure_casadi_solve(self.solve_args)
        self._p_vec = _parameters

        # Use the following functions for evaluation of the result
        self.solution_stats = self.nlp_solver.stats()
        stats = self.nlp_solver.stats()
        # print(f"Solver status: {stats['return_status']}")
        if (
            not stats["return_status"] in ["OPTIMAL", "SUCCESS"]
            and stats["unified_return_status"] != "SOLVER_RET_SUCCESS"
        ):
            warnings.warn(
                f"Solver did not converge to optimal solution (status == {self.nlp_solver.stats()['return_status']})"
            )
        # print(self.sol)

    def _pure_casadi_solve(self, solve_args: dict):
        with logger.show_output():
            # with logger.suppress_output():
            sol = self.nlp_solver(**solve_args)
        return sol

    # ============ solution access =================== -> source out of the solver class

    def _access_vars(self, x, list_var_names: list[str]):
        assert (
            len(self.state_names) == self.state_dim
        ), "Access with opt_vector function! This is deprecated."

        def get(var_name):
            idx = self.state_names.index(var_name)
            return float(x[idx])

        return np.array([get(var_name) for var_name in list_var_names])

    def _access_vector(self, x, vector_name: str):
        return x[self.state_toc[vector_name][0] : self.state_toc[vector_name][1]]

    def opt_vars(self, list_var_names: list[str]):
        """Function to recover values of variables at the solution.

        Warning, this will only work correctly if exclusively one-dimensional variables are used.
        """
        assert (
            len(self.state_names) == self.state_dim
        ), "Access with opt_vector function! This is deprecated."

        if self.sol is None:
            raise ValueError("solve OCP first before accessing solution!")

        x = self.sol["x"]

        def get(var_name):
            idx = self.state_names.index(var_name)
            return float(x[idx])

        return np.array([get(var_name) for var_name in list_var_names])

    def opt_vector(self, var_name: str) -> np.ndarray:
        """Access solution if states are vectors."""

        assert isinstance(var_name, str), f"Pass variable name here: ({var_name}) as string!"

        if self.sol is None:
            raise ValueError("solve OCP first before accessing solution!")

        x = self.sol["x"]
        cas_x = x[self.state_toc[var_name][0] : self.state_toc[var_name][1]]

        # For some reason, casadi DM -> np.array returns a Nx1 array
        np_x = np.array(cas_x).squeeze(axis=1)

        return np_x

    def opt_multipliers(self, list_var_names: list[str]):
        if self.sol is None:
            raise ValueError("solve OCP first before accessing solution!")

        return self._access_vars(self.sol["lam_x"], list_var_names)

    def constr_multipliers(self, constraint_name: str) -> np.ndarray:
        if self.sol is None:
            raise ValueError("solve OCP first before accessing solution!")

        vec = self.sol["lam_g"][
            self.constr_toc[constraint_name][0] : self.constr_toc[constraint_name][1]
        ]

        return vec.full().squeeze(axis=1)

    def constr_values(self) -> np.ndarray:
        """Get the value of the constraints at the solution."""
        if self.sol is None:
            raise ValueError("solve OCP first before solution access.")

        return self.sol["g"]

    def opt_objective(self):
        if self.sol is None:
            raise ValueError("solve OCP first before accessing solution!")

        return float(self.sol["f"])

    def get_custom_function_value(self, func_name: str):
        if self.sol is None:
            raise ValueError("solve OCP first before accessing solution!")

        return self._custom_functions[func_name](self.sol["x"], self._p_vec).full()

    def opt_gradient(self, func_name: str, wrt_var_name: str) -> np.ndarray:
        """
        Get values of user function gradients at solution.

        Returns matrix of shape (dim(user_function), dim(wrt_vector))
        """
        if self.sol is None:
            raise ValueError("solve OCP first before accessing solution!")

        x = self.sol["x"]
        func_val = self._custom_functions[func_name](x, self._p_vec)

        if self.solver_name == "osqp":
            grad_x, grad_p = self._custom_functions[func_name].jacobian()(x, self._p_vec, func_val)
        else:
            grad_x, grad_p = self._custom_functions[func_name].jacobian()(x, self._p_vec, func_val)
            # grad_x = _grad[:self.state_dim]
            # grad_p = _grad[:self.state_dim]

        return grad_x[:, self.state_toc[wrt_var_name][0] : self.state_toc[wrt_var_name][1]].full()

        # return self.obj_func.jacobian()(self.sol["x"])[self.state_toc[wrt_var_name][0]:self.state_toc[wrt_var_name][1]]

    def opt_par_sensitivity(self, func_name: str, wrt_par_name: str) -> np.ndarray:
        if self.sol is None:
            raise ValueError("solve OCP first before accessing solution!")

        x = self.sol["x"]
        func_val = self._custom_functions[func_name](x, self._p_vec)

        if self.solver_name == "osqp":
            grad_x, grad_p = self._custom_functions[func_name].jacobian()(x, self._p_vec, func_val)
        else:
            grad_x, grad_p = self._custom_functions[func_name].jacobian()(x, self._p_vec, func_val)
            # grad_x = _grad[:self.state_dim]
            # grad_p = _grad[:self.state_dim]

        return grad_p[:, self.par_toc[wrt_par_name][0] : self.par_toc[wrt_par_name][1]].full()
