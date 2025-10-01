from dataclasses import dataclass
from typing import Type

from grecco_sim.local_control import local_control_basic
from grecco_sim.local_control import local_control_distr_opt

from .implementations.coord_second_order import CoordinatorSecondOrder
from .implementations.coord_central_opt import CentralOptimizationCoordinator
from .implementations.coord_admm import CoordinatorADMM
from .implementations.coord_first_order import CoordinatorVujanic
from .implementations.coord_first_order import CoordinatorGradientDescent
from .implementations.coord_first_order import CoordinatorDailyGridFee
from .implementations.coordinator_dual_newton import CoordinatorDualNewton

from .coord_base import NoCentralControl


@dataclass
class CoordinationMechanism:
    """Data structure to specify the used coordination mechanism."""

    name: str
    controller_local: dict[tuple, Type]
    coordinator: Type


CM_CENTRAL_OPTIMIZATION = CoordinationMechanism(
    "central_optimization",
    {
        ("load",): local_control_basic.LocalControllerNoControl,
        ("load", "pv"): local_control_basic.LocalControllerNoControl,
        ("bat", "load"): local_control_basic.LocalControllerPassControl,
        ("bat", "load", "pv"): local_control_basic.LocalControllerPassControl,
        ("ev", "load"): local_control_basic.LocalControllerPassControl,
        ("ev", "load", "pv"): local_control_basic.LocalControllerPassControl,
        ("hp", "load"): local_control_basic.LocalControllerPassControl,
        ("hp", "load", "pv"): local_control_basic.LocalControllerPassControl,
        ("bat", "hp", "load", "pv"): local_control_basic.LocalControllerPassControl,
        ("ev", "hp", "load", "pv"): local_control_basic.LocalControllerPassControl,
        ("bat", "ev", "load", "pv"): local_control_basic.LocalControllerPassControl,
        # from 2033
        ("bat", "hp", "load"): local_control_basic.LocalControllerPassControl,
        ("ev", "hp", "load"): local_control_basic.LocalControllerPassControl,
        ("ev", "ev", "load"): local_control_basic.LocalControllerPassControl,
        ("bat", "ev", "hp", "load", "pv"): local_control_basic.LocalControllerPassControl,
    },
    CentralOptimizationCoordinator,
)

CM_VUJANIC = CoordinationMechanism("vujanic", {}, CoordinatorVujanic)
CM_ADMM = CoordinationMechanism(
    "admm",
    {
        ("load",): local_control_basic.LocalControllerNoControl,
        ("load", "pv"): local_control_basic.LocalControllerNoControl,
        ("bat", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
        ("hp", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
        ("hp", "load"): local_control_distr_opt.LocalControllerSecondOrder,
        ("bat", "hp", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
        ("ev", "hp", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
        ("bat", "ev", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
        ("ev", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
        ("ev", "load"): local_control_distr_opt.LocalControllerSecondOrder,
        # from 2033
        ("bat", "hp", "load"): local_control_distr_opt.LocalControllerSecondOrder,
        ("ev", "hp", "load"): local_control_distr_opt.LocalControllerSecondOrder,
        ("ev", "ev", "load"): local_control_distr_opt.LocalControllerSecondOrder,
        ("bat", "ev", "hp", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
    },
    CoordinatorADMM,
)

CM_GRADIENT_DESCENT = CoordinationMechanism(
    "gradient_descent",
    {
        "bat": local_control_distr_opt.LocalControllerFirstOrder,
        "load": local_control_basic.LocalControllerNoControl,
        "pv": local_control_basic.LocalControllerNoControl,
    },
    CoordinatorGradientDescent,
)

CM_PLAIN_GRID_FEE = CoordinationMechanism(
    "plain_grid_fee",
    {
        ("load",): local_control_basic.LocalControllerNoControl,
        ("load", "pv"): local_control_basic.LocalControllerNoControl,
        ("bat", "load"): local_control_distr_opt.LocalControllerFirstOrder,
        ("bat", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("ev", "load"): local_control_distr_opt.LocalControllerFirstOrder,
        ("ev", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("hp", "load"): local_control_distr_opt.LocalControllerFirstOrder,
        ("hp", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("bat", "hp", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("ev", "hp", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("bat", "ev", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        # from 2033
        ("bat", "hp", "load"): local_control_distr_opt.LocalControllerFirstOrder,
        ("ev", "hp", "load"): local_control_distr_opt.LocalControllerFirstOrder,
        ("ev", "ev", "load"): local_control_distr_opt.LocalControllerFirstOrder,
        ("bat", "ev", "hp", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        # Several EV
        ("ev", "ev", "hp", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("ev", "ev", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("bat", "ev", "ev", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("ev", "ev", "load", "pv"): local_control_distr_opt.LocalControllerFirstOrder,
        ("ev", "ev", "load"): local_control_distr_opt.LocalControllerFirstOrder,
    },
    CoordinatorDailyGridFee,
)


CM_ALADIN = CoordinationMechanism(
    "second_order",
    {
        ("bat", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
        ("load",): local_control_basic.LocalControllerNoControl,
        ("load", "pv"): local_control_basic.LocalControllerNoControl,
        ("hp", "load", "pv"): local_control_distr_opt.LocalControllerSecondOrder,
        ("bat", "ev", "load", "pv"): local_control_basic.LocalControllerNoControl,
        ("ev", "load", "pv"): local_control_basic.LocalControllerNoControl,
        ("ev", "load"): local_control_basic.LocalControllerNoControl,
    },
    CoordinatorSecondOrder,
)

CM_NONE = CoordinationMechanism(
    "none",
    {
        ("load", "pv"): local_control_basic.LocalControllerNoControl,
        ("load",): local_control_basic.LocalControllerNoControl,
        ("bat", "load", "pv"): local_control_basic.LocalControllerNoControl,
        ("bat", "load"): local_control_basic.LocalControllerNoControl,
        ("ev", "load", "pv"): local_control_basic.LocalControllerEVBaseline,
        ("ev", "ev", "load", "pv"): local_control_basic.LocalControllerEVBaseline,
        ("ev", "ev", "load"): local_control_basic.LocalControllerEVBaseline,
        ("ev", "load"): local_control_basic.LocalControllerEVBaseline,
        ("hp", "load"): local_control_basic.LocalControllerNoControl,
        ("hp", "load", "pv"): local_control_basic.LocalControllerNoControl,
        # multiple flexis
        ("bat", "ev", "load", "pv"): local_control_basic.LocalControllerNoControl,
        ("bat", "ev", "ev", "load", "pv"): local_control_basic.LocalControllerNoControl,
        ("bat", "hp", "load", "pv"): local_control_basic.LocalControllerNoControl,
        ("ev", "hp", "load", "pv"): local_control_basic.LocalControllerEVBaseline,
        ("ev", "ev", "hp", "load", "pv"): local_control_basic.LocalControllerEVBaseline,
        ("ev", "ev", "load", "pv"): local_control_basic.LocalControllerEVBaseline,
        # from 2033
        ("bat", "hp", "load"): local_control_basic.LocalControllerNoControl,
        ("ev", "hp", "load"): local_control_basic.LocalControllerEVBaseline,
        ("ev", "ev", "load"): local_control_basic.LocalControllerEVBaseline,
        ("bat", "ev", "hp", "load", "pv"): local_control_basic.LocalControllerNoControl,
    },
    NoCentralControl,
)

CM_LOCAL_SELF_SUFF = CoordinationMechanism(
    "local_self_suff",
    {
        # no PVs
        ("load", "pv"): local_control_basic.LocalControllerNoControl,
        ("ev", "load"): local_control_basic.LocalControllerEVBaseline,
        ("hp", "load"): local_control_basic.LocalControllerNoControl,
        ("load",): local_control_basic.LocalControllerNoControl,
        # with PVs
        ("bat", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
        ("bat", "load"): local_control_basic.LocalControllerSelfSuff,
        ("ev", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
        ("ev", "ev", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
        ("hp", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
        # TODO: This needs some specific consideration in the controller logic!
        ("bat", "ev", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
        ("bat", "hp", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
        ("ev", "hp", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
        ("ev", "ev", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
        # from 2033
        ("bat", "hp", "load"): local_control_basic.LocalControllerSelfSuff,
        (
            "ev",
            "hp",
            "load",
        ): local_control_basic.LocalControllerEVBaseline,  # TODO: needs a new controller
        ("ev", "ev", "load"): local_control_basic.LocalControllerEVBaseline,
        ("bat", "ev", "hp", "load", "pv"): local_control_basic.LocalControllerSelfSuff,
    },
    NoCentralControl,
)

CM_DUAL_NEWTON = CoordinationMechanism(
    "dual_newton",
    {
        "bat": local_control_distr_opt.LocalControllerDualNewton,
        "load": local_control_basic.LocalControllerNoControl,
        "pv": local_control_basic.LocalControllerNoControl,
        "heatpump": local_control_distr_opt.LocalControllerDualNewton,
    },
    CoordinatorDualNewton,
)

CM_DUAL_NEWTON = CoordinationMechanism(
    "dual_newton",
    {
        "bat": local_control_distr_opt.LocalControllerDualNewton,
        "load": local_control_basic.LocalControllerNoControl,
        "pv": local_control_basic.LocalControllerNoControl,
        "heatpump": local_control_distr_opt.LocalControllerDualNewton,
    },
    CoordinatorDualNewton,
)

# ToDo: Choose Either call objects 'coordination_mechanism' or 'coordinator'.
AVAILABLE_COORDINATORS = [
    CM_ALADIN,
    CM_CENTRAL_OPTIMIZATION,
    CM_VUJANIC,
    CM_ADMM,
    CM_GRADIENT_DESCENT,
    CM_NONE,
    CM_LOCAL_SELF_SUFF,
    CM_PLAIN_GRID_FEE,
    CM_DUAL_NEWTON,
]

# ToDo: Discuss compromise: No loop in grid_node.py, but also no extra method.
ALL_COORDINATORS = {
    "second_order": CM_ALADIN,
    "central_optimization": CM_CENTRAL_OPTIMIZATION,
    "vujanic": CM_VUJANIC,
    "admm": CM_ADMM,
    "gradient_descent": CM_GRADIENT_DESCENT,
    "none": CM_NONE,
    "local_self_suff": CM_LOCAL_SELF_SUFF,
    "plain_grid_fee": CM_PLAIN_GRID_FEE,
    "dual_newton": CM_DUAL_NEWTON,
}
