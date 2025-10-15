from typing import Type
import numpy as np
import pandas as pd

from grecco_sim.sim_models import battery, ev_charger, heat_pump
from grecco_sim.sim_models import model_base
from grecco_sim.util import type_defs


class PV(model_base.Model):
    """Class for plain PV system."""

    def __init__(
        self,
        sys_id: str,
        horizon: int,
        dt_h: float,
        params: type_defs.SysPars,
        ts_in: pd.DataFrame,
    ):
        """
        Initialize system.

        Pass input time series with column named '{sys_id}_p_ac'
        """
        super().__init__(sys_id, horizon, dt_h)

        assert len(ts_in) == self.horizon, (
            f"Length of given time series ({len(ts_in)})"
            f" does not match horizon length ({self.horizon})"
        )
        self.pvs = ts_in[f"{sys_id}_p_ac"].values

    @property
    def system_type(self):
        return "pv"

    def apply_control(self, control):
        self.k += 1

    def get_state(self):
        return {"pv_generation": self.pvs[self.k]}

    def get_output(self):
        return {"p_ac": self.pvs}

    def get_grid_power(self):
        """Return AC power with correct sign convention to sum up household."""
        return -self.pvs

    def get_grid_power_at(self, k: int):
        """Return AC power with correct sign convention to sum up household."""
        return -self.pvs[k]


class Load(model_base.Model):
    def __init__(
        self, sys_id: str, horizon: int, dt_h: float, params: type_defs.SysPars, ts_in: pd.DataFrame
    ):
        super().__init__(sys_id, horizon, dt_h)

        assert len(ts_in) == self.horizon, (
            f"Length of given time series ({len(ts_in)})"
            f" does not match horizon length ({self.horizon})"
        )
        self.load = ts_in[f"{sys_id}_p_load"].values
        assert (self.load >= 0.0).all(), "Sign convention is load >! 0"

    @property
    def system_type(self):
        return "load"

    def apply_control(self, control):
        self.k += 1

    def get_state(self):
        return {"load_power": self.load[self.k]}

    def get_output(self):
        return {"p_load": self.load}

    def get_grid_power(self):
        """Return AC power with correct sign convention to sum up household."""
        return self.load

    def get_grid_power_at(self, k: int):
        """Return AC power with correct sign convention to sum up household."""
        return self.load[k]


class Household(model_base.Model):
    """
    A household is a grid connection point!

    That means, the household is the point the EMS can control.

    For each household, all unit models are initialized according to the given parameters.

    """

    UNIT_MAPPING: dict[str, Type[model_base.Model]] = {
        "load": Load,
        "pv": PV,
        "bat": battery.Storage,
        "bat_age": battery.StorageAging,
        "ev": ev_charger.EVCharger,
        "hp": heat_pump.ThermalSystem,
    }

    def __init__(
        self,
        sys_id: str,
        horizon: int,
        dt_h: float,
        params: dict[str, type_defs.SysPars],
        ts_in: pd.DataFrame,
    ):
        super().__init__(sys_id, horizon, dt_h)
        # print(params)

        # This dict will hold the units with their ID as a key.
        self.units: dict[str, model_base.Model] = {}
        available_heat_demand = False
        for col in ts_in.columns:
            if "heat_demand" in col:
                available_heat_demand = True
                break
        for unit_id, unit_pars in params.items():
            match unit_pars.system:
                case "load":
                    col_list = [unit_id + "_p_load"]
                case "pv":
                    col_list = [unit_id + "_p_ac"]
                case "bat":
                    col_list = []
                case "ev":
                    col_list = [
                        unit_id + "_cp",
                        unit_id + "_initial_soc",
                        unit_id + "_target_soc",
                        unit_id + "_until_departure",
                    ]
                case "hp":
                    col_list = ["Outside Temperature", "Solar Irradiation"]
                    if available_heat_demand:
                        col_list.append(f"{unit_id}_heat_demand")
                case _:
                    raise ValueError(f"Unknown system type{unit_pars.system}")

            sub_ts = ts_in.loc[:, col_list]
            self.units[unit_id] = self.UNIT_MAPPING[unit_pars.system](
                unit_id, horizon, dt_h, unit_pars, sub_ts
            )

        # Normal initialization
        self.c_sup_list = np.ones(ts_in.index.shape) * params[sys_id].c_sup
        self.c_feed_list = np.ones(ts_in.index.shape) * params[sys_id].c_feed

        if "c_sup" in ts_in.columns:
            self.c_sup_list = list(ts_in["c_sup"])
        if "c_feed" in ts_in.columns:
            self.c_feed = list(ts_in["c_feed"])

    @property
    def system_type(self):
        return "household"

    def apply_control(self, control: dict[str, float]):
        for sub_name, sub in self.units.items():
            sub.apply_control(control[sub_name])
        self.k += 1

    def get_state(self) -> dict[str, float]:
        ret = {"k": self.k}
        for sub in self.units.values():
            ret.update(sub.get_state())
        return ret

    def get_grid_power_at(self, k: int):
        grid = 0.0
        for unit in self.units.values():
            grid += unit.get_grid_power_at(k)

        return grid

    def get_grid_power(self):
        grid = np.zeros(self.horizon)
        for unit in self.units.values():
            grid += unit.get_grid_power()

        return grid

    def get_output(self):
        # This generates output as it used to: Maybe this won't work properly for multiple EVs
        subs = {unit.system_type: unit.get_output() for unit in self.units.values()}

        grid = np.zeros(self.horizon)
        for unit in self.units.values():
            grid += unit.get_grid_power()

        res = {
            "grid": grid,
            "c_sup": self.c_sup_list,
            "c_feed": self.c_feed_list,
            **subs,
        }

        return res
