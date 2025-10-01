import abc
from typing import Any, Optional
from grecco_sim.sim_data_tools import forecast_provider
from grecco_sim.util import sig_types
from grecco_sim.util import type_defs

# from grecco_sim.util.type_defs import LocalFuture
# from grecco_sim.controller import mycas


class LocalControllerBase(abc.ABC):
    """Interface for local controllers."""

    def __init__(self, model_parameters: dict[str, type_defs.SysPars]):
        # map system key to system type from model_params
        self.system_types = {key: par.system for key, par in model_parameters.items()}

    @abc.abstractmethod
    def get_control(self, signal: sig_types.SignalType, state: dict[str, Any]) -> dict[str, float]:
        """Get control at current time for local system's flexibility."""

    @abc.abstractmethod
    def get_flex_schedule(
        self,
        forecast: type_defs.Forecast,
        state: dict[str, Any],
        signal: sig_types.SignalType,
        previous_signals: Optional[dict[str, sig_types.SignalType]] = None,
    ) -> type_defs.LocalFuture:
        """Provide a local future for coordination."""


class LocalOptimalControlBase(LocalControllerBase):
    """Extend abstract base class with a few checks for optimization."""

    def __init__(self, model_parameters: dict[str, type_defs.SysPars]):
        super().__init__(model_parameters)

        if any(key[0].isdigit() for key in model_parameters):
            raise ValueError("Flexibility names may not start with numbers!")


class LocalBasicControl(LocalControllerBase):
    """This class is a base class for all sorts of basic controllers.

    This can be used for controllers in coordination mechanisms that do not have any coordination.


    It defines the abstract control methods for battery, heat pump, and ev charger that
    must be implemented by the child classes.

    These functions are thought to be independent from each other.
    """

    def __init__(self, model_parameters: dict[str, type_defs.SysPars]):
        super().__init__(model_parameters)

        self.control_mapping = {
            "load": lambda _: 0,
            "pv": lambda _: 0,
            "bat": self._control_bat,
            "ev": self._control_ev,
            "hp": self._control_hp,
        }
        """A dispatch map for system type to control function."""

    def get_control(self, signal: sig_types.SignalType, state: dict[str, Any]) -> dict[str, float]:
        control = {}
        for key, system_type in self.system_types.items():
            if system_type not in self.control_mapping:
                raise ValueError(f"Unsupported system type: {system_type}")
            control[key] = self.control_mapping[system_type](state)
        return control

    @abc.abstractmethod
    def _control_ev(self, state: dict) -> float:
        """Return a control for the EV charger."""

    @abc.abstractmethod
    def _control_hp(self, state: dict) -> float:
        """Return a control for the heat pump."""

    @abc.abstractmethod
    def _control_bat(self, state: dict) -> float:
        """Return a control for the battery."""

    def get_flex_schedule(
        self,
        forecast: type_defs.Forecast,
        state: dict[str, Any],
        signal: sig_types.SignalType,
        previous_signals: dict[str, sig_types.SignalType] | None = None,
    ) -> type_defs.LocalFuture:
        """Provide a schedule for the flexibilities.

        Warning: this function is only accurate for PV and load.
        Therefore, controllers that control EV, HP or bat, and that are used in a
        coordination mechanism that uses the returned Futures,
        should override this function and return a future that considers the flexibilitie's schedule
        """
        future = type_defs.LocalFuture(forecast.fc_res_load)
        return future


class LocalControllerSelfSuff(LocalBasicControl):
    """Local controller that aims for self-sufficiency by matching generation to load."""

    def __init__(self, *args, model_params):
        super().__init__(model_params)

        # extract system parameters from model_params
        for unit_name, unit_type in self.system_types.items():
            # Process HP (heat pump) systems
            if unit_type == "hp":
                self.temp_max_heat = model_params[unit_name].temp_max_heat
                self.temp_min_heat = model_params[unit_name].temp_min_heat
                self.temp_max_cold = model_params[unit_name].temp_max_cold
                self.temp_min_cold = model_params[unit_name].temp_min_cold
                self.p_max = model_params[unit_name].p_max
                self.cop = model_params[unit_name].cop

            # Process EV (electric vehicle) systems
            elif unit_type == "ev":
                self.charger_power = model_params[unit_name].p_lim_ac
                self.capacity = model_params[unit_name].capacity

    def _control_bat(self, state):
        # returns a power
        return state["pv_generation"] - state["load_power"]

    def _control_ev(self, state):
        if state["ev_connected"] and (state["ev_soc"] < state["ev_target_soc"]):
            if "pv_generation" in state.keys():
                # only works with max charge power, otherwise proxy would be needed
                delta_soc_poss = state["ev_remaining_time_h"] * self.charger_power / self.capacity
                delta_soc_need = state["ev_target_soc"] - state["ev_soc"]
                # in the following logic it is possible that charging is delayed as generation
                # does not exceed load and grid demand is greater with delayed charging
                if delta_soc_poss >= delta_soc_need:
                    if state["pv_generation"] > state["load_power"]:
                        # can potentially be set to the difference of pv and load
                        # with _set_power_continuous in EV Charging Process
                        return self.charger_power
                    return 0.0
                return self.charger_power
        return 0.0

    def _control_hp(self, state):
        if "pv_generation" in state.keys():
            if state["pv_generation"] > state["load_power"]:
                if (state["temp"] > self.temp_max_cold) or (state["temp"] > self.temp_max_heat):
                    # cooling
                    return -1
                elif (state["temp"] < self.temp_min_heat) or (state["temp"] < self.temp_min_cold):
                    # heating
                    return 1
        # model decides based on temperature
        return 0


class LocalControllerEVBaseline(LocalBasicControl):
    """Local controller implementing baseline EV charging behavior."""

    def __init__(self, *args, model_params):
        super().__init__(model_params)

        for unit, type in self.system_types.items():
            if type == "ev":
                self.charger_power = model_params[unit].p_lim_ac

    def _control_ev(self, state) -> float:
        # if connected and soc < target_soc charge with max power
        if state["ev_connected"] and state["ev_soc"] < state["ev_target_soc"]:
            return self.charger_power
        else:
            return 0.0

    def _control_hp(self, state) -> float:
        # state should be cool, heat, off (-1, 1, 0)
        # defined in state['mode']
        # signal should be temperature
        return 0.0

    def _control_bat(self, state) -> float:
        # returns a power
        return 0.0


class LocalControllerSelfSuffEV(LocalBasicControl):
    """Local controller for systems without controllable loads."""

    def __init__(self, *args, model_params):
        super().__init__(model_params)

        for unit, type in self.system_types.items():
            if type == "ev":
                self.charger_power = model_params[unit].p_lim_ac
                self.capacity = model_params[unit].capacity

    def _control_ev(self, state):
        # if connected and soc < target_soc charge only if PV available or target
        # cannot be reached otherwise (=always wait if PV is available)
        if state["ev_connected"] and (state["soc"] < state["target_soc"]):
            if "pv_generation" in state.keys():
                # only works with max charge power, otherwise proxy would be needed
                delta_soc_poss = state["remaining_time_h"] * self.charger_power / self.capacity
                delta_soc_need = state["target_soc"] - state["soc"]
                # in the following logic it is possible that charging is delayed as generation
                # does not exceed load and grid demand is greater with delayed charging
                if delta_soc_poss >= delta_soc_need:
                    if state["pv_generation"] > state["load_power"]:
                        # can potentially be set to the difference of pv and load
                        # with _set_power_continuous in EV Charging Process
                        return self.charger_power
                    return 0.0
                return self.charger_power
        return 0.0


# class LocalControllerHeatPumpOnOff(LocalControllerBase):
#     """Local controller for heat pumps with on/off control based on temperature bounds."""
#     def __init__(self, model_params):
#         super().__init__()

#         # Define State based on inner temperature TK
#         # e.g. cooling when Tk > 26°C
#         # Stop cooling when Tk < 22°C
#         # heating when Tk < 19°C
#         # Stop heating when Tk > 23°
#         self.temp_max_heat = model_params.temp_max_heat
#         self.temp_min_heat = model_params.temp_min_heat
#         self.temp_max_cold = model_params.temp_max_cold
#         self.temp_min_cold = model_params.temp_min_cold

#     def get_control(self, signal, state):
#         # state should be cool, heat, off (-1, 1, 0)
#         # defined in state['mode']
#         # signal should be temperature
#         temp_k = state["temp"]
#         if state["mode"] == 1:
#             if temp_k > self.temp_max_heat:
#                 return 0
#             else:
#                 return 1
#         elif state["mode"] == -1:
#             if temp_k < self.temp_min_cold:
#                 return 0
#             else:
#                 return -1
#         elif state["mode"] == 0:
#             if temp_k < self.temp_min_heat:
#                 return 1
#             elif temp_k > self.temp_max_cold:
#                 return -1
#             else:
#                 return 0
#         else:
#             raise ValueError(f"Mode '{state['mode']}' invalid")

# class LocalControllerSelfSuffHP(LocalControllerBase):
#     """Local controller for systems without controllable loads."""

#     def __init__(self, *args, model_params):
#         self.temp_max_heat = model_params.temp_max_heat
#         self.temp_min_heat = model_params.temp_min_heat
#         self.temp_max_cold = model_params.temp_max_cold
#         self.temp_min_cold = model_params.temp_min_cold
#         self.p_max = model_params.p_max
#         self.cop = model_params.cop

#     def get_control(self, signal, state):
#         if "pv_generation" in state.keys():
#             if state["pv_generation"] > state["load_power"]:
#                 if (state["temp"] > self.temp_max_cold) or (state["temp"] > self.temp_max_heat):
#                     # cooling
#                     return -1
#                 elif (state["temp"] < self.temp_min_heat) or (state["temp"] < self.temp_min_cold):
#                     # heating
#                     return 1
#         # model decides based on temperature
#         return 0


class LocalControllerNoControl(LocalBasicControl):
    """Local controller for systems without controllable loads."""

    def __init__(self, *args, model_params):
        super().__init__(model_params)

        # Process HP (heat pump) systems
        for unit, type in self.system_types.items():
            if type == "hp":
                self.temp_max_heat = model_params[unit].temp_max_heat
                self.temp_min_heat = model_params[unit].temp_min_heat
                self.temp_max_cold = model_params[unit].temp_max_cold
                self.temp_min_cold = model_params[unit].temp_min_cold
                self.p_max = model_params[unit].p_max
                self.cop = model_params[unit].cop

            # Process EV (electric vehicle) systems
            elif type == "ev":
                self.charger_power = model_params[unit].p_lim_ac

    def _control_ev(self, state):
        if state["ev_connected"] and state["ev_soc"] < state["ev_target_soc"]:
            return self.charger_power
        return 0.0

    def _control_hp(self, state):
        # state should be cool, heat, off (-1, 1, 0)
        # defined in state['mode']
        # signal should be temperature
        temp_k = state["temp"]
        if state["mode"] == 1:
            if temp_k > self.temp_max_heat:
                return 0
            else:
                return 1
        elif state["mode"] == -1:
            if temp_k < self.temp_min_cold:
                return 0
            else:
                return -1
        elif state["mode"] == 0:
            if temp_k < self.temp_min_heat:
                return 1
            elif temp_k > self.temp_max_cold:
                return -1
            else:
                return 0
        else:
            raise ValueError(f"Mode '{state['mode']}' invalid")

    def _control_bat(self, state):
        # returns a power
        return 0


class LocalControllerPassControl(LocalControllerBase):
    """Controller which just passes the centrally determined control to the assets."""

    def __init__(
        self,
        sys_id,
        forecast_access: forecast_provider.ForecastProvider,
        controller_pars: type_defs.OptParameters,
        model_params: dict[str, type_defs.SysPars],
    ):
        """Initialize"""
        super().__init__(model_params)

        self.system_sim_pars = model_params

    def get_control(self, signal: sig_types.SignalType, state) -> dict[str, float]:
        """Get the control applicable in the current time step."""

        assert isinstance(
            signal, sig_types.DirectControlSignal
        ), "Direct pass control needed for this controller."

        ret = {}
        # Make sure, zeros are in the control for evs if no charging process

        return {
            sys_id: signal.control[sys_id][0] if sys_id in signal.control else None
            for sys_id in self.system_types
        }

    def get_flex_schedule(
        self,
        forecast: type_defs.Forecast,
        state: dict,
        signal: sig_types.SignalType,
        previous_signals: Optional[dict[str, sig_types.SignalType]] = None,
    ):

        return type_defs.LocalFuture(yg=forecast.fc_res_load, flex_type="continuous")
