"""Provide data type definitions."""

import datetime
import json
import dataclasses
import os
from pathlib import Path

import numpy as np

from grecco_sim.util import config


class EnhancedJSONEncoder(json.JSONEncoder):
    """Provide a way to serialize data classes."""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, Path):
            return str(o)
        elif isinstance(o, datetime.timedelta):
            return o.total_seconds()
        elif isinstance(o, datetime.datetime):
            return o.isoformat()
        elif isinstance(o, np.ndarray):
            return list(o)

        return super().default(o)


@dataclasses.dataclass
class RunParameters:
    """Parameterization of a simulation run."""

    # Horizon of the simulation. Loaded data is cropped. To start_time + 15min * horizon
    sim_horizon: int
    # Intended start time of the simulation
    start_time: datetime.datetime

    # Maximum number of iterations before breaking the market with suboptimal result
    max_market_iterations: int
    # Parameters specifying which algorithm type is run
    coordination_mechanism: str  # out of [central, distributed, none, admm]
    scenario: dict  # Scenario description. Should at least include a key: 'name' as identififer

    sim_tag: str  # unique identifier of simulation

    # The community optimization problem occuring in a certain time step can be saved to a file.
    inspection: list | None = None
    # path to store simulation output (e.g. time series, analysis results)
    output_file_dir: Path = Path(__file__).parent.parent.parent / "results" / "default"

    # Simulation time step in datetime timedelta
    dt: datetime.timedelta = datetime.timedelta(minutes=15)

    # Use previous signals in scheduling to augment local objective
    use_prev_signals: bool = False

    # Plotting parameters
    plot: bool = False
    show: bool = False

    # Debug settings
    profile_run: bool = False  # switch if profiling is requested when running sim.

    def __post_init__(self):
        self.output_file_dir = Path(self.output_file_dir)

        if not self.output_file_dir.is_absolute():
            self.output_file_dir = (
                Path(__file__).parent.parent.parent / "results" / self.output_file_dir
            )

        if isinstance(self.start_time, str):
            self.start_time = datetime.datetime.fromisoformat(self.start_time)
        assert self.start_time.tzinfo is not None, "Specify a time zone for simulation range."

    @property
    def dt_h(self) -> float:
        """Simulation time step in hours as float."""
        return self.dt.total_seconds() / 3600.0


@dataclasses.dataclass
class Forecast(object):
    """Datatype for forecast communication."""

    # Forecast of residual generation. Is relevant for all households.
    fc_res_load: np.ndarray

    # Additional time series e.g. weather only relevant to certrain systems.
    add_fc: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    @property
    def fc_len(self):
        return len(self.fc_res_load)

    @property
    def negative_part(self):
        """Return only negative values (net residual generation)."""
        return np.clip(self.fc_res_load, a_min=None, a_max=0)

    @property
    def positive_part(self):
        """Return only positive values (net residual load)."""
        return np.clip(self.fc_res_load, a_min=0, a_max=None)


ALLOWED_FLEX_TYPES = ["inflexible", "continuous", "discrete"]


@dataclasses.dataclass
class LocalFuture(object):
    """This is the message, a local agents sends to the coordinator."""

    # ToDo: Here would be some documentation/explanations very nice.

    yg: np.ndarray
    """Profile of power at grid connection point. This is used in coordination."""

    u: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)  # not necessary
    """Control profile. Used mainly for logging etc. Not used in coordination."""

    grads: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    # For ALADIN second order method use a jacobian matrix to avoid going against local constraints.
    jacobian: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))

    multiplier_sensitivities: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))

    _meta: dict = dataclasses.field(default_factory=dict)

    flex_type: str = "inflexible"

    @property
    def horizon(self):
        """Return horizon (= future length)."""
        return self._meta["fc"].fc_len

    @property
    def k(self):
        """Return current time step where future starts."""
        return self._meta["state"]["k"]

    def validate(self):
        """Validate that communicated data is suitable for central coordinator."""
        if self.grads is not None:
            assert (
                self.grads.shape == self.yg.shape
            ), "Grid power and gradients must have the same shape."

        if self.jacobian is not None:
            assert (
                self.jacobian.shape[1] == self.yg.shape[1]
            ), "Grid power horizon and jacobian horizon must be the same."

        if self.multiplier_sensitivities is not None:
            assert self.multiplier_sensitivities.shape[1] == self.yg.shape

        if self.u is not None:
            assert (
                self.u.shape == self.yg.shape
            ), "Grid power and battery power must have the same shape."

        assert (
            self.flex_type in ALLOWED_FLEX_TYPES
        ), f"flex_type (is: '{self.flex_type}') should be in {ALLOWED_FLEX_TYPES}"


@dataclasses.dataclass
class GridDescription:
    """This class is used to describe the grid for optimization."""

    # First guess model is just a large grid with one constraint
    p_lim: float


@dataclasses.dataclass
class OptParameters:
    """This data class encodes parameters for optimization solvers"""

    solver_name: str

    # coefficient for step size in ADMM and second order
    rho: float
    # Coefficient for constraint slack in second order algorithm
    mu: float

    # Coefficient for step size in first oder methods
    alpha: float

    horizon: int

    fc_type: str = "perfect"

    # Penalty for violating a temperature bound of the heating system.
    slack_penalty_thermal = 500.0
    # Penalty for missing the target SoC of a charging process in Eur / kWh
    slack_penalty_missing_soc = 2.0

    experimental_non_smooth: bool = False

    gurobi_version: str = "100"  # GrECCo tested on gurobi version 100.

    # Save this number of local solvers for casadi OCP reusing
    # If None: Save all
    buffer_n_solvers: int | None = None

    def __post_init__(self):
        """Set solver specific variables."""

        if self.solver_name == "gurobi":
            if sim_config := config.get_config():
                os.environ["GRB_LICENSE_FILE"] = sim_config["gurobi_license_file"]
                os.environ["GUROBI_VERSION"] = sim_config["gurobi_version"]
                return

            license_path = os.getenv("GUROBI_LICENSE_FILE")
            if license_path is None:
                msg = (
                    "Please set GUROBI_LICENSE_FILE. In conda you can do so "
                    "using\n conda env config vars set "
                    "GUROBI_LICENSE_FILE=<path/to/gurobi.lic>. \n "
                    "Reactivate your environment afterwards."
                )
                raise ValueError(msg)

            if not Path(license_path).exists():
                msg = f"GUROBI_LICENSE_FILE set incorrectly ({license_path})."
                raise FileNotFoundError(msg)

            os.environ["GUROBI_VERSION"] = self.gurobi_version


@dataclasses.dataclass
class SysPars:
    """Base data class for system description"""

    name: str
    # Simulation time step
    dt_h: float

    # Supply and feed in price
    c_sup: float
    c_feed: float

    @property
    def system(self) -> str:
        """
        Access system name via base class.

        The actual name has to be defined in each inheriting class.
        """
        return self._system  # noqa E1101 self._system must be initialized in child class.


@dataclasses.dataclass
class SysParsLoad(SysPars):
    """Parameter class describing a plain household"""

    _system: str = "load"


@dataclasses.dataclass
class SysParsPV(SysPars):
    """Parameter class describing a household with plain load and PV."""

    _system: str = "pv"


@dataclasses.dataclass
class SysParsPVBat(SysPars):
    """Parameter class describing a Battery system."""

    init_soc: float

    # Battery parameters
    capacity: float
    p_inv: float

    # Arguments with defaults (TBC)
    eff: float = 0.9
    p_lim_dc: float = 10.0
    p_lim_ac: float = 10.0
    on_off: bool = False

    # Bounds for state of charge
    x_lb: float = 0
    x_ub: float = 1.0

    _system: str = "bat"


@dataclasses.dataclass
class SysParsPVBatAging(SysPars):
    """Parameter class describing a PV Battery system."""

    init_soc: float

    init_soh: float

    # Battery parameters
    capacity: float
    p_inv: float

    s_therm: float
    s_dod: float

    a_soc: float
    b_soc: float

    a_char: float
    b_char: float

    # Arguments with defaults
    eff: float = 0.9
    p_lim_dc: float = 10.0
    on_off: bool = False

    # Bounds for state of charge
    x_lb: float = 0.0
    x_ub: float = 1.0

    _system: str = "pv_bat_age"


@dataclasses.dataclass
class SysParsHeatPump(SysPars):
    """Parameter class describing a Heat Pump system."""

    initial_temp: float = 20

    # Household parameters          # TODO: Validate data.  This has to be variable among type of households
    thermal_mass: float = 16500  # in kJ/ K
    heat_rate: float = (
        0.05  # In kW/K  # coefficient determining heat transfer through building hull
    )
    absorbance: float = 0.1  # Percentage of absorbance of solar irradiation
    irradiance_area: float = 10  # in m2

    # Heat pump model can either be "on-off" or "variable-speed"
    heat_pump_model: str = "on-off"

    # Parameters used for the control of the heat pump
    temp_min_heat: float = 18
    temp_max_heat: float = 23
    temp_min_cold: float = 22
    temp_max_cold: float = 26

    heat_pump_type: str = None
    temp_lower_bound: float = 17
    temp_upper_bound: float = 25
    p_max: float = 2.0  # electric nominal power
    cop: float = 2.5

    _system: str = "hp"


@dataclasses.dataclass
class SysParsEV(SysPars):
    """Parameter class describing a EV system."""

    # Charging parameters
    init_soc: float
    target_soc: float

    # Battery parameters
    capacity: float
    p_inv: float

    # Arguments with defaults (eff from SynPro Data, TBC)
    eff: float = 0.93
    p_lim_dc: float = 10.0
    p_lim_ac: float = 11.0

    # Bounds for state of charge
    x_lb: float = 0.0
    x_ub: float = 1.0

    _system: str = "ev"
