from grecco_sim import coordinators
from grecco_sim.sim_data_tools import forecast_provider
from grecco_sim.util import sig_types
from grecco_sim.sim_models import models
from grecco_sim.local_control import local_control_basic
from grecco_sim.util import type_defs


class GridNode(object):
    def __init__(
        self,
        sys_id: str,
        run_pars: type_defs.RunParameters,
        model_input: dict,
        opt_params: type_defs.OptParameters,
    ):
        """GridNodes are passive grid elements that pass information.

        Args:
            sys_id: Unique id of node. Attached unit (devices) ids are derived from it.
            run_pars: Run specific parameters are required in various places.
            model_input: Parameters and time-dependent data for all units.
            opt_params: GridNodes create their own local optimizers.
            mulit_flex: If True, GridNode has multiple flexibilities. -> controller is set accordingly.

        """

        self.sys_id = sys_id
        self.model_input = model_input
        self.run_pars = run_pars

        # Container to store all future control signals.
        self.signals = {}

        self.phys_model = models.Household(
            sys_id=sys_id,
            horizon=run_pars.sim_horizon,
            dt_h=run_pars.dt_h,
            params=model_input["params"],
            ts_in=model_input["ts"],
        )

        if opt_params.fc_type == "perfect":
            self.fc_access = forecast_provider.PerfectForesightForecast(self.phys_model)
        else:
            self.fc_access = forecast_provider.ForecastProviderNaive(self.phys_model, 96)

        # Init LocalController
        cm_name = run_pars.coordination_mechanism

        model_configuration = tuple(pars.system for pars in model_input["params"].values())
        model_configuration = tuple(sorted(model_configuration))

        cm = coordinators.ALL_COORDINATORS[cm_name]

        if model_configuration not in cm.controller_local:
            raise ValueError(
                f"Not controller defined for system configuration {model_configuration} "
                f"and coordination mechanism {cm_name}"
            )

        local_controller_class = cm.controller_local[model_configuration]
        self.local_controller: local_control_basic.LocalControllerBase = local_controller_class(
            self.sys_id, self.fc_access, opt_params, model_params=model_input["params"]
        )

    def get_current_future(
        self, time_index: int, horizon: int, signal: sig_types.SignalType
    ) -> type_defs.LocalFuture:
        """Apply control and get respective future states ('Future').

        Args:
            time_index: Current timestep as index of shared index array.
            horizon: Determines how many timesteps into the future the weather
                forecast should be regarded.
            signal: Node's LocalController reacts to this control signal.

        Returns:
            Future states ('Future') after control ('Signal') is applied.

        """

        forecast = self.fc_access.get_fc(time_index, horizon)
        state = self.phys_model.get_state()

        # Update forecast with current measurement
        forecast.fc_res_load[0] = state["load_power"]
        if "pv_generation" in state:
            forecast.fc_res_load[0] -= state["pv_generation"]

        if self.run_pars.use_prev_signals is not None:
            extra_arg = {"previous_signals": self.signals}
        else:
            extra_arg = {}

        future = self.local_controller.get_flex_schedule(
            forecast=forecast, state=state, signal=signal, **extra_arg
        )

        future._meta = {
            "fc": forecast,
            "state": self._get_state(),
            "_model_pars": self.model_input["params"],
        }

        return future

    # TODO this function can be restricted to only transmit necessary parts.
    def _get_state(self):
        """Return the state"""
        return self.phys_model.get_state()

    def get_grid_power_at(self, k: int) -> float:
        """Return the grid power at a certain time step.

        This can represent a smart meter measurement executed by the DSO
        """
        return self.phys_model.get_grid_power_at(k)

    def apply_control(self, signal: sig_types.SignalType):
        """Converts central controller signal to local control and applies it.

        Args:
            signal: As provided by central controller.

        ToDo: Check get_state, get_control and apply_control for multiple
            flexis.
        ToDo: It is a little unintuitive to create a control in a method
            called 'apply_control'.
        """

        sys_state = self.phys_model.get_state()
        self.signals[sys_state["k"]] = signal

        control = self.local_controller.get_control(signal, sys_state)

        self.phys_model.apply_control(control)

    def get_output(self):
        """Pass output of physical model together with respective signals."""

        sys_output = self.phys_model.get_output()
        sys_output["signals"] = self.signals

        return sys_output
