"""
This module provides an interface and a few simple implementations to a forecast provider.

The golar of such a class is to provide a forecast using a certain method.
A simple implementation would e.g. be the PerfectForesightForecast.
Here, the get_fc method will just return the actual power time series.
"""

import abc
import numpy as np

from grecco_sim.sim_models import heat_pump, models
from grecco_sim.util import type_defs


class ForecastProvider(abc.ABC):
    """Abstract interface for forecast provider."""

    def __init__(self, household: models.Household):
        """Initialize object.

        The forecast ist always specific to a certain household.
        """
        self.hh = household

    @abc.abstractmethod
    def get_fc(self, sim_time_step: int, fc_horizon: int) -> type_defs.Forecast:
        """
        Return forecast.

        The returned forecast format is specified in the type definitions.
        All forecasts include a time series for residual load.

        Other time series are optional depending on the system type

        E.g.
        `Forecast(
            fc_res_load=[x, y, z, ...],
            add_fc={"temp_outside": [a, b, c, ...]}
        )`
        """


class PerfectForesightForecast(ForecastProvider):
    """Implementation for perfect foresight forecast."""

    def __init__(self, household: models.Household):
        """Initializing parent class is sufficient."""
        super().__init__(household)

        # Flag if HP related fc needs to be passed
        self.has_hp = False
        self.fc_res_load = np.zeros(self.hh.horizon)

        for unit in self.hh.units.values():
            match unit.system_type:
                case "load":
                    self.fc_res_load += unit.load.copy()
                case "pv":
                    self.fc_res_load -= unit.pvs.copy()
                case "hp":
                    self.has_hp = True
                    self.temp_outside = unit.temp_out.copy()
                    self.solar_heat_gain = unit.irradiation.copy()
                case _:
                    continue

    def get_fc(self, sim_time_step: int, fc_horizon: int) -> type_defs.Forecast:
        """Return perfect forecast i.e. the realized time series."""
        fc_range = slice(sim_time_step, sim_time_step + fc_horizon)

        fc_res_load = self.fc_res_load[fc_range]

        fc_ret = {
            "c_feed": self.hh.c_feed_list[fc_range],
            "c_sup": self.hh.c_sup_list[fc_range],
        }
        if self.has_hp:
            fc_ret["temp_outside"] = self.temp_outside[fc_range]
            fc_ret["solar_heat_gain"] = self.solar_heat_gain[fc_range]

        return type_defs.Forecast(fc_res_load, add_fc=fc_ret)


class ForecastProviderNaive(ForecastProvider):
    """
    This class generates a naive seasonal forecast by shifting the time series by a certain
    number of values.
    """

    def __init__(self, household: models.Household, shift_by: int) -> None:

        super().__init__(household)

        self.res_load = household.subsystems["load"].load
        if "pv" in household.subsystems:
            self.res_load -= household.subsystems["pv"].pvs

        self.shifted = np.concatenate(
            [
                self.res_load[-shift_by:],  # takes the last shift_by values
                self.res_load[:-shift_by],  # takes the complete array but the last shift_by values
            ]
        )

        if "hp" in self.hh.subsystems:
            self.shifted_temp_outside = np.concatenate(
                [
                    self.hh.subsystems["hp"].temp_out[-shift_by:],
                    self.hh.subsystems["hp"].temp_out[:-shift_by],
                ]
            )
            self.shifted_solar_heat_gain = np.concatenate(
                [
                    self.hh.subsystems["hp"].irradiation[-shift_by:],
                    self.hh.subsystems["hp"].irradiation[:-shift_by],
                ]
            )

    def get_fc(self, sim_time_step: int, fc_horizon: int) -> type_defs.Forecast:
        """
        Return Naive seasonal forecast.

        This is hacky in some way. The current value should be input to the forecast.
        """

        fc_range = slice(sim_time_step, sim_time_step + fc_horizon)

        fc_res_load = self.shifted[fc_range]
        fc_res_load[0] = self.res_load[sim_time_step]

        fc_ret = {
            "c_feed": self.hh.c_feed_list[fc_range],
            "c_sup": self.hh.c_sup_list[fc_range],
        }
        if "hp" in self.hh.subsystems:
            fc_ret["temp_outside"] = self.shifted_temp_outside[fc_range]
            fc_ret["temp_outside"][0] = self.hh.subsystems["hp"].temp_out[sim_time_step]

            fc_ret["solar_heat_gain"] = self.shifted_solar_heat_gain[fc_range]
            fc_ret["solar_heat_gain"][0] = self.hh.subsystems["hp"].irradiation[sim_time_step]

        return type_defs.Forecast(fc_res_load, add_fc=fc_ret)
