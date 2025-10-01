
"""
    Ttwo heuristics / baseline algorithms for central coordination
"""

from grecco_sim.coordinators.coord_interface import CoordinatorInterface


class NoCentralControl(CoordinatorInterface):
    """
    This central controller is used when all control should be determined locally.
    E.g. self-sufficiency or no control or similar
    """

    coord_name = "Uncoordinated"

    def __init__(self, *args, **kwargs):
        super().__init__(horizon=1)

    def has_converged(self, time_index):
        return True

    def get_signals(self, futures):
        return {sys_id: None for sys_id in futures}

