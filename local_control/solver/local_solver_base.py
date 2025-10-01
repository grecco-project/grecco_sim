import abc
from grecco_sim.local_control import mycas


class LocalSolverBase(abc.ABC):
    """Base class for the physical optimization model formulation."""

    @abc.abstractmethod
    def get_central_problem_contribution(self) -> tuple[mycas.MyOCP, mycas.MySX]:
        """
        Return contribution of a single physical system to OCP.

        Returns tuple of OCP and grid variable.
        """

    def get_user_functions(self) -> dict[str, mycas.casadi.SX]:
        """Extend with user functions. (See paper which user functions are necessary.)"""
        return {}
