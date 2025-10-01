"""
This module provides controllers for several coordination mechanisms using distributed optimization.

This includes first and second order (dual Newton) dual methods and a primal second order method.
"""

from typing import Optional
import numpy as np

from grecco_sim.util import sig_types
from grecco_sim.sim_data_tools import forecast_provider
from grecco_sim.util import type_defs
from grecco_sim.local_control.solver import (
    local_admm,
    local_dual_newton,
    local_gradient_descent,
)

from grecco_sim.local_control import local_control_basic


class LocalControllerFirstOrder(local_control_basic.LocalOptimalControlBase):
    """Controller for first order dual method.

    This controller will take a price curve (multiplier).
    """

    def __init__(
        self,
        sys_id,
        forecast_access: forecast_provider.ForecastProvider,
        controller_pars: type_defs.OptParameters,
        model_params: dict[str, type_defs.SysPars],
    ):
        super().__init__(model_params)

        self.fc_access = forecast_access
        self.dt_h = 0.25
        self.sys_id = sys_id

        self.system_sim_pars = model_params
        self.opt_pars = controller_pars

    def get_control(self, signal: sig_types.SignalType, state) -> dict[str, float]:
        """Get the control applicable in the current time step."""

        assert isinstance(
            signal, sig_types.FirstOrderSignal
        ), "First order signal needed for this controller."

        fc = self.fc_access.get_fc(state["k"], signal.signal_len)
        future = self.get_flex_schedule(fc, state, signal)
        return {sys_id: future.u[sys_id][0] for sys_id in future.u}

    def get_flex_schedule(
        self,
        forecast: type_defs.Forecast,
        state: dict,
        signal: sig_types.SignalType,
        previous_signals: Optional[dict[str, sig_types.SignalType]] = None,
    ):

        if isinstance(signal, sig_types.NoneSignal):
            signal = sig_types.FirstOrderSignal(np.zeros(forecast.fc_len))
        elif isinstance(signal, sig_types.FirstOrderSignal):
            # Signal type is already correct.
            pass
        else:
            raise ValueError("First order signal needed for this controller.")

        # To pickle use:
        # obj = forecast, self.sys_id, self.system_sim_pars, self.opt_pars, state, signal
        solver = local_gradient_descent.get_solver(
            forecast.fc_len, self.sys_id, self.system_sim_pars, self.opt_pars
        )
        solver.solve(state, forecast, signal)
        # if solver.get_u()[0] != 0.:
        #     # for debugging HP controller
        #     u_vec = solver.get_u()
        #     print(solver.get_u())

        return type_defs.LocalFuture(
            u=solver.get_u(),
            yg=solver.get_yg(),
        )


class LocalControllerSecondOrder(local_control_basic.LocalOptimalControlBase):
    """Class providing a controller for a local agent to take part in an ADMM or ALADIN scheme."""

    def __init__(
        self,
        sys_id: str,
        forecast_access: forecast_provider.ForecastProvider,
        controller_pars: type_defs.OptParameters,
        model_params: dict[str, type_defs.SysPars],
    ):

        super().__init__(model_params)
        self.fc_access = forecast_access
        self.sys_id = sys_id

        self.system_sim_pars = model_params
        self.controller_pars = controller_pars

    def get_control(self, signal: sig_types.SignalType, state) -> dict[str, float]:
        """Get the control applicable in the current time step."""

        assert isinstance(signal, sig_types.SecondOrderSignal), "Please pass a second order signal!"

        fc_len = len(signal.res_power_set)

        fc = self.fc_access.get_fc(state["k"], fc_len)
        future = self.get_flex_schedule(fc, state, signal)
        return {sys_id: future.u[sys_id][0] for sys_id in future.u}

    def get_flex_schedule(
        self,
        forecast: type_defs.Forecast,
        state,
        signal: sig_types.SignalType,
        previous_signals: Optional[dict[str, sig_types.SignalType]] = None,
    ):

        if isinstance(signal, sig_types.NoneSignal):
            signal = sig_types.SecondOrderSignal(np.zeros(forecast.fc_len), forecast.fc_res_load)
        elif isinstance(signal, sig_types.SecondOrderSignal):
            # Signal type is already correct.
            pass
        else:
            raise ValueError("Second order or None signal needed for this controller.")

        if previous_signals is None:
            previous_signals = {}

        time_index = state["k"]
        cropped_signals = [
            sig_types.SecondOrderSignal(
                signal.res_power_set[time_index - k :],
                signal.mul_lambda[time_index - k :],
            )
            for k, signal in previous_signals.items()
            if k + signal.signal_len > time_index
        ]

        signal_lengths = [signal.signal_len for signal in cropped_signals]

        solver = local_admm.get_solver(
            signal.signal_len,
            signal_lengths,
            self.sys_id,
            self.system_sim_pars,
            self.controller_pars,
        )
        solver.solve(state, forecast, signal, previous_signals=cropped_signals)

        return type_defs.LocalFuture(
            u=solver.get_u(),
            yg=solver.get_yg(),
            # Returning the gradient is only necessary when using in a real second order
            # context (ALADIN). Can be set to arbitrary value in plain ADMM.
            # grads=solver.get_local_gradients(),
            # jacobian=solver.get_active_constraint_jacobian(),
            flex_type="continuous",
        )


class LocalControllerDualNewton(local_control_basic.LocalOptimalControlBase):
    """This controller takes part in a Dual Newton coordination scheme.

    This scheme is implemented following Reinhardt et al. 2024.
    """

    def __init__(
        self,
        sys_id,
        forecast_access,
        controller_pars: type_defs.OptParameters,
        model_params: dict[str, type_defs.SysPars],
    ):

        super().__init__(model_params)
        self.fc_access = forecast_access
        self.sys_id = sys_id

        self.system_sim_pars = model_params
        self.controller_pars = controller_pars

    def get_control(self, signal: sig_types.SignalType, state):

        assert isinstance(signal, sig_types.FirstOrderSignal), "Please pass a first order signal!"

        fc_len = len(signal.mul_lambda)

        fc = self.fc_access.get_fc(state["k"], fc_len)
        future = self.get_flex_schedule(fc, state, signal)
        return future.u[0]

    def get_flex_schedule(
        self,
        forecast,
        state,
        signal: sig_types.FirstOrderSignal,
        previous_signals: Optional[dict[int, sig_types.SecondOrderSignal]] = None,
    ):

        if isinstance(signal, sig_types.NoneSignal):
            signal = sig_types.FirstOrderSignal(np.zeros(forecast.fc_len))
        elif isinstance(signal, sig_types.FirstOrderSignal):
            # Signal type is already correct.
            pass
        else:
            raise ValueError("First order or None signal needed for this controller.")

        if previous_signals is None:
            previous_signals = {}

        time_index = state["k"]
        cropped_signals = [
            sig_types.FirstOrderSignal(signal.mul_lambda[time_index - k :])
            for k, signal in previous_signals.items()
            if k + signal.signal_len > time_index
        ]

        signal_lengths = [signal.signal_len for signal in cropped_signals]

        solver = local_dual_newton.get_solver(
            signal.signal_len,
            signal_lengths,
            self.sys_id,
            self.system_sim_pars,
            self.controller_pars,
        )
        solver.solve(state, forecast, signal)

        return type_defs.LocalFuture(
            u=solver.get_u(),
            yg=solver.get_yg(),
            multiplier_sensitivities=solver.get_sensitivities(),
            flex_type="continuous",
        )


if __name__ == "__main__":
    pass
