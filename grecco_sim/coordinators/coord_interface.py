import abc
from typing import Dict

from grecco_sim.util import sig_types
from grecco_sim.util import type_defs


class CoordinatorInterface(abc.ABC):
    """
    This class is the abstract base class for all coordinators.

    It defines the functions that must be implemented in the GreCCo framework.
    """


    coord_name = "Unnamed Coordinator"

    def __init__(self, horizon: int):
        self.horizon = horizon

    # @abc.abstractmethod
    def get_initial_signal(self) -> sig_types.SignalType | None:
        """
        Return Initial signal in market clearing.
        
        This signal is used to initiate the coordination loop.
        """
        return None

    @abc.abstractmethod
    def has_converged(self, time_index: int) -> bool:
        """Return true if the Coordination procedure has converged."""

    @abc.abstractmethod
    def get_signals(
        self, futures: Dict[str, type_defs.LocalFuture]
    ) -> Dict[str, sig_types.SignalType]:
        """Return potentially personalized signals to be sent to local agents."""

    def get_cost_realization(
        self,
        initial_futures: Dict[str, type_defs.LocalFuture],
        realization_grid: Dict[str, float],
        signals: Dict[str, sig_types.SignalType],
    ) -> Dict[str, float]:
        """Assign a reward per agent for collaboration."""
        assert set(initial_futures.keys()) == set(realization_grid.keys()) == set(signals.keys())
        return {f"fee_{key}": 0. for key in realization_grid.keys()}
