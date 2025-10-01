import warnings
from grecco_sim.sim_models import heat_pump
from grecco_sim.util import type_defs
from grecco_sim.local_control import local_control_basic
import pandas as pd
import numpy as np


def test_heat_pump_mode():
    # Testing correct mode. Heating : 1, Cooling : -1, HeatPumpOff : 0
    model_params = {"hp": type_defs.SysParsHeatPump("name", 0.25, 0.3, 0.1)}
    expected_mode = [1, 1, 0, 0, -1, 0]
    temperature_inside = [
        -20,
        model_params["hp"].temp_min_heat - 2,
        model_params["hp"].temp_max_heat + 2,
        model_params["hp"].temp_max_heat - 2,
        model_params["hp"].temp_max_cold + 2,
        model_params["hp"].temp_min_cold - 2,
    ]

    # mode_before indicates the mode of the heat pump before the current timestep, hence temperatur
    # is needed as an input for the controller
    mode_before = [0, 1, 1, 0, 0, -1]
    controller = local_control_basic.LocalControllerSelfSuff(
        None, None, None, model_params=model_params
    )

    # Array to save control signals
    control_signals = []

    # Test the controller
    for i, temp in enumerate(temperature_inside):
        mode = mode_before[i]
        state = {"temp": temp, "mode": mode}
        control = controller.get_control(None, state)
        control_signals.append(control["hp"])
    assert np.array_equal(control_signals, expected_mode), (
        f"Output of HP sim {control_signals}",
        f" differs from expected result: {expected_mode}.",
    )


def test_heat_pump_variable_speed():
    # This test checks if a given control signal is processed as expected
    # A household with a huge thermal mass is simulated for
    # getting the desired control response
    temperature_cold = [-5, -5, -5, -5]
    solar_irradiation = [0, 0, 0, 0]
    weather_info = pd.DataFrame(
        {
            "Outside Temperature": temperature_cold,
            "Solar Irradiation": solar_irradiation,
        }
    )
    big_household = heat_pump.ThermalSystem(
        "test",
        4,
        0.25,
        type_defs.SysParsHeatPump(
            name="test",
            dt_h=0.25,
            c_sup=0.30,
            c_feed=0.1,
            heat_pump_model="variable-speed",
            heat_pump_type="F370 1x230",
        ),
        weather_info,
    )

    # General testing control signals
    second_point = (big_household.heat_pump.p_cut_off) - 1
    third_point = (big_household.heat_pump.p_cut_off + big_household.heat_pump.p_max) / 2
    fourth_point = (big_household.heat_pump.p_max) + 1
    control_signal = [-2, second_point, third_point, fourth_point]

    # Expected feedback
    expected_feedback = [
        0,
        big_household.heat_pump.p_cut_off,
        third_point,
        big_household.heat_pump.p_max,
    ]

    # Array to save output of the set_power function
    output = []

    # Modes for the heat pump
    modes = [0, 1, 1, 1]

    big_household.c_r = 10000000000000000  # Huge thermal mass

    for i, control in enumerate(control_signal):
        big_household.k += 1
        big_household.heat_pump.set_power(modes[i], control)
        output.append(big_household.heat_pump.p_in)

    assert np.array_equal(
        output, expected_feedback
    ), f"Output of HP sim {output} differs from expected result: {expected_feedback}."


def test_heat_pump_on_off():
    # This test checks if the heat pump is turned on and off correctly
    # A household with a small thermal mass is simulated for
    # getting the desired control response. It should return a signal for cooling
    temperature = [23, 23, 23, 23, 23, 23]
    solar_irradiation = [1000, 1000, 1000, 1000, 1000, 1000]
    weather_info = pd.DataFrame(
        {"Outside Temperature": temperature, "Solar Irradiation": solar_irradiation}
    )
    thermal_system = heat_pump.ThermalSystem(
        "test", 6, 0.25, type_defs.SysParsHeatPump, weather_info
    )
    p_max = thermal_system.heat_pump.p_max
    cop = thermal_system.heat_pump.cop

    control_signals = [0, -1, 0, 1, 0, -1]

    expected_feedback = [0, -p_max * cop, 0, p_max * cop, 0, -p_max * cop]

    for i, control in enumerate(control_signals):
        thermal_system.mode[i] = control
        thermal_system.apply_control(control)

    assert np.array_equal(
        thermal_system.q_hp, expected_feedback
    ), f"Output of HP sim {thermal_system.q_hp} differs from expected result: {expected_feedback}."


if __name__ == "__main__":

    test_heat_pump_mode()
    # test_heat_pump_variable_speed()
    # test_heat_pump_on_off()
