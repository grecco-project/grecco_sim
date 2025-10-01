"""This module provides a model for a heat pump thermal system."""

import abc
from enum import Enum
import pathlib
import warnings

import numpy as np
import pandas as pd

from grecco_sim.util import type_defs
from grecco_sim.sim_models import model_base


class OpMode(Enum):
    """Enum type for modes of heat pump operation (should be somewhat translatable to SGready)"""

    MUST_HEAT = 1
    MUST_COOL = 2
    MUST_OFF = 3
    FREE = 4


class HeatPumpBase(abc.ABC):
    """
    Abstract Base class for heat pump models.

    This class defines an interface for specific heat pump implementations.
    """

    p_in: float = 0  # electric input power
    q_out: float = 0  # thermal output

    @abc.abstractmethod
    def set_operation(self, mode: OpMode, control: float) -> None:
        """Set the operation mode of the heat pump.

        Parameters:
            mode: set mode (mandatory heating/cooling or free operation)
            control: set a power for control (the specific implementation can transform this (ON/OFF))
        """


class HeatPumpOnOff(HeatPumpBase):
    """
    This model class represents a simple on-off heat pump model. So the heatpump can either be off and deliver no heat/cold
    or be on and deliver a constant amount of heat/cold. The heat pump is controlled by a simple on-off controller.

    Parameters: heat_pump_type - select from database.
                To each type, refrigerant, pressure levels and compressor power limits are defined
    """

    def __init__(self, parameters):

        # Input parameters
        self.p_max = parameters.p_max
        self.cop = parameters.cop

        self.p_in = 0.0
        self.q_out = 0

    def set_operation(self, mode: OpMode, control: float) -> None:

        assert isinstance(mode, OpMode), "Clean up old stuff!"

        if mode == OpMode.MUST_HEAT:
            self.p_in = self.p_max
            self.q_out = self.p_max * self.cop
        elif mode == OpMode.MUST_OFF:
            self.p_in = 0.0
            self.q_out = 0.0
        else:
            # Free operation as controlled.
            if control > 0.0:  # heating
                self.p_in = self.p_max
                self.q_out = self.p_max * self.cop
            elif control < 0.0:  # cooling if permittet
                self.p_in = self.p_max
                self.q_out = -self.p_max * self.cop
            else:
                self.p_in = 0.0
                self.q_out = 0.0


class HeatPumpVarSpeed:
    """
    This model class represents a simple on-off heat pump model. So the heatpump can either be off and deliver no heat/cold
    or be on and deliver a constant amount of heat/cold. The heat pump is controlled by a simple on-off controller.

    Parameters: heat_pump_type - select from database.
                To each type, refrigerant, pressure levels and compressor power limits are defined
    """

    def __init__(self, parameters):
        warnings.warn("Update with abstract base class and OpMode.")

        # Input parameters
        self.p_max = parameters.p_max
        self.cop = parameters.cop
        self.refrigerant = parameters.refrigerant
        self.p_cut_off = parameters.p_cut_off
        self.slope = parameters.slope
        self.intercept = parameters.intercept

    def set_power(self, mode, control):
        """
        Set the power input and mode of the heat pump
        Parameters:
        mode - 1 for heating, -1 for cooling and 0 for off
        control - Control Power input in kW.  If control is none, heatpump
        """
        if mode == 0:
            self.p_in = 0
        elif mode != 0:
            if control is None:
                self.p_in = self.p_max
            else:
                if control < self.p_cut_off:
                    self.p_in = self.p_cut_off
                elif control > self.p_max:
                    self.p_in = self.p_max
                else:
                    self.p_in = control

    def get_heat_from_heatpump(self, mode, control=None):
        """
        Get the heat output from the power input

        Parameters:  mode -  1 for cooling, 1 for heating and 0 for off
        Return: q_out - Heat to/into household
        """

        self.set_power(mode, control=control)
        q_out = (self.intercept + (self.slope * self.p_in)) * mode

        return q_out


class ThermalSystem(model_base.Model):
    """
    This class model represents the thermal system of the household and interacts with the heat pump model
    defined in class Heat_Pump_Thermodynamic and the EMS configured in the controllers
    """

    def __init__(
        self,
        sys_id: str,
        horizon: int,
        dt_h: float,
        params: type_defs.SysParsHeatPump,
        ts_in: pd.DataFrame,
    ):
        """
        :param sys_id: unique ID of the system
        :param horizon: horizon of simulation
        :param params: parameters to govern storage behavior in simulation
        """
        super().__init__(sys_id, horizon, dt_h)

        # Required form is pandas dataframe but can be changed
        self.temp_out = np.array(ts_in["Outside Temperature"])
        self.irradiation = np.array(ts_in["Solar Irradiation"])
        self.use_model = False
        if f"{sys_id}_heat_demand" in ts_in.columns:
            self.heat_demand = np.array(ts_in[f"{sys_id}_heat_demand"])
        else:
            self.use_model = True
        self.temp = np.zeros(self.horizon + 1)
        self.mode = np.zeros(self.horizon + 1)
        self.p_in = np.zeros(self.horizon)
        self.q_hp = np.zeros(self.horizon)
        self.u_ext = np.zeros(self.horizon)
        self.losses = np.zeros(self.horizon + 1)

        self.heat_pump: HeatPumpBase

        self._init_model(params)

    def _init_model(self, params: type_defs.SysParsHeatPump):

        # Initializing state parameters
        self.temp[0] = params.initial_temp
        self.losses[0] = 0.0

        self.params = params

        # Adding builiding parameters
        self.absorbance = params.absorbance
        self.irradiance_area = params.irradiance_area

        # Either on-off or variable speed
        self.heat_pump_model = params.heat_pump_model

        # Import the heat pump database
        hp_db_path = pathlib.Path(__file__).parent.absolute()
        hp_db_path = hp_db_path.parent.parent
        hp_db_path = (
            hp_db_path / "data" / "heat_pump_database" / "heat_pump_database_short_version.csv"
        )
        heat_pump_info = pd.read_csv(hp_db_path, sep=";", index_col=0)
        self.heat_pump_type = params.heat_pump_type

        if self.heat_pump_type is not None:  # Assigning a model if known
            # If the model's name does not exist in the heat pump database, retrieve data from the default model
            if self.heat_pump_type in heat_pump_info.index:
                self.heat_pump_params = heat_pump_info.loc[self.heat_pump_type]

            else:
                raise ValueError(
                    f"ERROR: Entered model {self.heat_pump_type} is not in "
                    "the database. Select from {heat_pump_info.index.values}"
                )
            # Define the heat pump model
            if self.heat_pump_model == "on-off":
                self.heat_pump = HeatPumpOnOff(self.heat_pump_params)
            elif self.heat_pump_model == "variable-speed":
                self.heat_pump = HeatPumpVarSpeed(self.heat_pump_params)
            else:
                raise ValueError(
                    f"ERROR: Entered model {self.heat_pump_model} is not valid. Select from 'on-off' or 'variable-speed'"
                )
        else:
            self.heat_pump = HeatPumpOnOff(params)  # Initializing onoff when information is limited

    @property
    def system_type(self):
        return "hp"

    def apply_control(self, control):
        """
        Apply control to the Thermal System
        :return: no return
        """
        self.k += 1  # convention: first: iterate then calculate
        self.u_ext[self.k - 1] = control
        self._evolve(control)

    def _get_mode(self) -> OpMode:
        """Make sure that temeperature bounds cannot be violated."""

        temp = self.temp[self.k - 1]
        if temp < self.params.temp_min_heat:
            return OpMode.MUST_HEAT
        elif temp > self.params.temp_max_heat:
            return OpMode.MUST_OFF
        else:
            return OpMode.FREE

        # Please reintegrate if you want to model cooling
        # But the way it was integrated just didn't use the on/off external control
        # if self.mode[self.k - 1] == 1:
        #     if self.temp[self.k] > self.temp_max_heat:
        #         self.mode[self.k] = 0
        #     else:
        #         self.mode[self.k] = 1
        # elif self.mode[self.k - 1] == -1:
        #     if self.temp[self.k] < self.temp_min_cold:
        #         self.mode[self.k] = 0
        #     else:
        #         self.mode[self.k] = -1
        # elif self.mode[self.k - 1] == 0:
        #     if self.temp[self.k] < self.temp_min_heat:
        #         self.mode[self.k] = 1
        #     elif self.temp[self.k] > self.temp_max_cold:
        #         self.mode[self.k] = -1
        #     else:
        #         self.mode[self.k] = 0

    def _evolve(self, control: float):
        """
        Evolve the thermal system model by one time step using the heat pump model
        If the heat pump model is "variable-speed", the power is calculated in the set_power_and_mode function
        and the adjusted schedule self.p_in, a room temperature profile (self.temp) is calculated
        """
        now = self.k - 1
        after = self.k

        # Determine mode and heat/power of HP
        mode = self._get_mode()
        # if mode != OpMode.FREE or control != 0.:
        # print(control)

        self.heat_pump.set_operation(mode, control)

        # Evolve mode evolution
        self.p_in[now] = self.heat_pump.p_in
        self.q_hp[now] = self.heat_pump.q_out

        if self.use_model:
            heat_transfer_to_outside = self.params.heat_rate * (self.temp_out[now] - self.temp[now])
            heat_from_solar_irradiation = (
                self.irradiation[now] * self.absorbance * self.irradiance_area / 1000
            )  # In kW
            self.losses[after] = heat_transfer_to_outside + heat_from_solar_irradiation
            net_heat = heat_transfer_to_outside + self.heat_pump.q_out + heat_from_solar_irradiation
        else:
            # If the heat demand is given, use it
            thermal_storage_losses = self.params.heat_rate * (
                18 - self.temp[now]
            )  # Harcoded room temperature
            self.losses[after] = thermal_storage_losses
            net_heat = self.heat_pump.q_out + thermal_storage_losses - self.heat_demand[now]
        self.temp[after] = self.temp[now] + (
            (1 / self.params.thermal_mass) * net_heat * 3600 * self.dt_h
        )

    def get_state(self):
        return {
            "temp": self.temp[self.k],
            "mode": self.mode[self.k - 1],
        }

    def get_output(self):
        return {
            "temp": self.temp,
            "mode": self.mode,  # +1 for heating, -1 for cooling, 0 for off
            "p_in": self.p_in,
            "q_hp": self.q_hp,
            "u_ext": self.u_ext,
            "losses": self.losses,
        }

    def get_grid_power_at(self, k: int):
        return self.p_in[k]

    def get_grid_power(self):
        return self.p_in
