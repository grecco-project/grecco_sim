"""
This class offers a solver interface that solves the local problem
and returns the primal solution together with a sensitivity of the local solution
on the multiplier d y_g / d lam


With that the central coordinator can make a dual update step.


Questions:
- Will that work with non-smooth sensitivities?
- How will that work with integer variables?

How do the sensitivities look like typically?

"""

from collections import Counter
import numpy as np

from grecco_sim.local_control.solver.common import LocalSolverGradientDescent
from grecco_sim.util import type_defs


class LocalSolverDualNewton(LocalSolverGradientDescent):
    def get_sensitivities(self) -> np.ndarray:

        sensitivies = self.nlp_solver.opt_par_sensitivity("grid_var", "lam_a")

        return sensitivies


_created_solvers: dict[str, LocalSolverDualNewton] = {}


def get_solver(
    horizon: int,
    signal_lengths: list[int],
    sys_id: str,
    sys_parameters: type_defs.SysParsPVBat,
    controller_pars: type_defs.OptParameters,
) -> LocalSolverDualNewton:
    """Make sure that a solver with a certain configuration is reused."""

    solver_hash = f"{sys_id}_{horizon}_" + "_".join(
        [f"{n}_{count}" for n, count in sorted(Counter(signal_lengths).items())]
    )

    if solver_hash in _created_solvers:
        return _created_solvers[solver_hash]
    else:
        _created_solvers[solver_hash] = LocalSolverDualNewton(
            horizon, sys_id, sys_parameters, controller_pars
        )

    return _created_solvers[solver_hash]
