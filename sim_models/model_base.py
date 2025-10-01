import abc

import numpy as np


class Model(abc.ABC):
    """
    The Model interface
    """

    def __init__(self, sys_id, horizon, dt_h, *args):
        self.sys_id = sys_id
        self.horizon = horizon
        self.dt_h = dt_h

        self.k = 0

    @property
    @abc.abstractmethod
    def system_type(self) -> str:
        """Return the type of the system e.g. 'pv' or 'load'"""

    @abc.abstractmethod
    def apply_control(self, control):
        """Abstract method to apply a control to the system."""

    @abc.abstractmethod
    def get_state(self) -> dict:
        """Return the state of the system."""

    @abc.abstractmethod
    def get_output(self) -> dict[str, np.ndarray]:
        """Abstract method to return an output.

        This method should return a dictionary with outpput time series.
        """

    @abc.abstractmethod
    def get_grid_power_at(self, k: int) -> float:
        """Abstractmethod returning the grid power at a given time point."""

    @abc.abstractmethod
    def get_grid_power(self) -> np.ndarray:
        """Implement with a function returning the vector of grid power of system."""
