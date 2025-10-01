"""This module provides classes to load simulation data."""

import abc
import os
import pathlib
import pickle
from typing import List, Dict
import datetime
import pandas as pd
import numpy as np

from grecco_sim.sim_models import grid
from grecco_sim.util import config, type_defs, data_io


class InputDataLoader(abc.ABC):
    """
    Interface definition for Input data loading.
    Use different implementations for different kinds of input data.
    E.g. dummy data, data with grid information and without.
    """

    @abc.abstractmethod
    def get_input_data(self, sys_id: str, scenario: Dict) -> pd.DataFrame:
        """Return dataframe with necessary input data columns."""

    @abc.abstractmethod
    def get_parameters(self, sys_id: str) -> dict:
        """Return list of parameterizations of agents in scenario."""

    @abc.abstractmethod
    def get_sys_ids(self) -> List[str]:
        "Return a list of unique ids of agents in the scenario."


class DummyInputDataLoader(InputDataLoader):
    """
    This class creates dummy data by randomly drawing power profiles.

    Enhance or write an alternative to read actual simulation data.
    """

    def __init__(self, time_index, scenario: Dict, pv_kwp=5.0, load_kw=8.0):
        self.time_index = time_index

        self.sys_ids = [f"dummy_{i:02d}" for i in range(scenario["n_agents"])]

        self.load_kw = load_kw
        self.pv_kwp = pv_kwp
        self.multi_flex_bool = False

    def _get_load(self):
        noon = datetime.datetime(1, 1, 1, 12, 0, 0)
        diff = np.array(
            [
                (datetime.datetime.combine(noon.date(), idx.time()) - noon).total_seconds() / 3600.0
                for idx in self.time_index
            ]
        )
        diff_sq = (diff**2 - 36) ** 2
        data = np.exp(-diff_sq * np.random.random(len(self.time_index)) / 150)
        return self.load_kw * data

    def _get_pv(self):
        noon = datetime.datetime(1, 1, 1, 12, 0, 0)
        diff = np.array(
            [
                (datetime.datetime.combine(noon.date(), idx.time()) - noon).total_seconds() / 3600.0
                for idx in self.time_index
            ]
        )
        pv_ts = self.pv_kwp * np.exp(-((diff / 6.0) ** 2))
        pv_ts += np.random.random(len(pv_ts))
        return pv_ts

    def _get_ev(self):
        ts_cp = np.array([np.nan] * len(self.time_index), dtype=object)
        ts_cp[::20] = {"parking_time_h": 3, "capacity": 40.0, "init_soc": 0.7, "target_soc": 0.8}
        return ts_cp

    def get_input_data(self, sys_id: str, scenario: Dict) -> pd.DataFrame:
        data = {f"{sys_id}_load_p_load": self._get_load(), f"{sys_id}_pv_p_ac": self._get_pv()}

        ret = pd.DataFrame(index=self.time_index, data=data)
        return ret

    def get_parameters(self, sys_id: str) -> dict:
        assert sys_id in self.sys_ids, f"Requested sys_id '{sys_id}' not in available Ids"
        params = dict()
        if sys_id == "dummy_00":
            params = type_defs.SysParsPVBat(
                sys_id,
                c_sup=0.3,
                c_feed=0.1,
                eff=0.9,
                capacity=10.0,
                init_soc=0.0,
                dt_h=0.25,
                p_inv=5.0,
            )
        else:
            params = type_defs.SysParsLoad(sys_id, 0.25, 0.3, 0.1, "load")

        return params

        # ev_params = dict(system="ev", eff=0.9, init_soc=0.5, p_lim_dc=11., p_lim_ac=11.)

    def get_sys_ids(self):
        return self.sys_ids


class SampleInputDataLoader(InputDataLoader):
    """
    Loads Sample data from example data input.
    """

    def __init__(self, time_index: pd.DatetimeIndex, scenario: Dict):
        self.time_index = time_index

        if "data_path" in scenario:
            self.data_path = pathlib.Path(scenario["data_path"])
        else:
            # get the path of current file
            path = pathlib.Path(__file__).parent.absolute()
            # go two levels up
            self.data_path = path.parent.parent / "data" / "sample_scenario"

        self.scenario = scenario
        self.sys_ids = []
        self.multi_flex_bool = False
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise ValueError(f"Given data path for scenario input {self.data_path} is non-existant")

        self._data = {}

        for comp in ["load", "pv"]:
            data = pd.read_csv(self.data_path / f"{comp}_data.csv", index_col=0)

            # Compare data length with time_index length
            # If Data is longer, then takes the first len(time_index) elements
            # If Data is shorter, raise exception
            if len(data) > len(self.time_index):
                data = data[: len(self.time_index)]
            elif len(data) < len(self.time_index):
                raise ValueError(
                    f"Input data for {comp} in scenario '{self.scenario['name']}' "
                    f"has shorter length ({len(data)}) then requested simulation horizon ({len(self.time_index)})"
                )
            self.time_index = self.time_index[: len(data)]

            self._data[comp] = data.reset_index(drop=True).set_index(self.time_index)

        # =================================================================
        # This could be a little more usefully selecting. But works for now I guess
        self.sys_ids = self._data["pv"].columns[: self.scenario["n_agents"]]
        # =================================================================

        if self.scenario["focus"] == "hp":
            data = pd.read_csv(self.data_path / "weather_data.csv", index_col=0)

            if len(data) > len(self.time_index):
                data = data[: len(self.time_index)]
            elif len(data) < len(self.time_index):
                raise ValueError(
                    f"Input data for weather in scenario '{self.scenario['name']}' "
                    f"has shorter length ({len(data)}) then requested simulation horizon ({len(self.time_index)})"
                )

            self._data["weather"] = data.reset_index(drop=True).set_index(self.time_index)

        # TODO: add check that agents (= _data.columns) are identical over all data sets.

    def get_input_data(self, sys_id: str, scenario: Dict) -> pd.DataFrame:
        data = {f"{sys_id}_p_load": self._data["load"][sys_id].values}
        if sys_id in self._data["pv"]:
            data[f"{sys_id}_pv_p_ac"] = self._data["pv"][sys_id].values
        else:
            data[f"{sys_id}_pv_p_ac"] = [0.0] * len(self.time_index)

        if self.scenario["focus"] == "hp":
            data.update(
                {_col: self._data["weather"][_col].values for _col in self._data["weather"]}
            )

        ret = pd.DataFrame(index=self.time_index, data=data)
        return ret

    def get_parameters(self, sys_id: str) -> dict:
        assert sys_id in self.sys_ids, f"Requested sys_id '{sys_id}' not in available Ids"
        params = dict()
        params[sys_id] = type_defs.SysParsLoad(sys_id, c_sup=0.3, c_feed=0.1, dt_h=0.25)
        if self.scenario["focus"] == "pv_bat":
            params[sys_id + "_pv"] = type_defs.SysParsPVBat(
                sys_id,
                c_sup=0.3,
                c_feed=0.1,
                eff=0.9,
                capacity=10.0,
                init_soc=0.0,
                dt_h=0.25,
                p_inv=5.0,
            )
        elif self.scenario["focus"] == "hp":
            params[sys_id + "_hp"] = type_defs.SysParsHeatPump(
                sys_id, dt_h=0.25, c_sup=0.3, c_feed=0.1
            )
        else:
            raise ValueError(f"Unknown focus of sample data {self.scenario['focus']}")
        return params

    def get_sys_ids(self):
        return self.sys_ids


class PyPsaGridInputLoader(InputDataLoader):
    """
    This class is a data container for the input data of the simulation, to be used by the simulator,
    and avoid reloading the data for each agent.
    The data is loaded from the paths specified in the constructor, and stored in the class attributes.
    If no path is provided, the framework checks if the default path contains
    the required files. If not, dummy data is used.
    """

    MWH_TO_KWH = 1000.0

    def __init__(self, time_index: pd.DatetimeIndex, scenario: Dict, dt_h: datetime.timedelta):

        self.dt_h = dt_h
        self.time_index = time_index

        # TODO: Discuss: Would be handy to have a GrECCo config class that
        #   handles data verification and default values?
        if "grid_data_path" and "weather_data_path" in scenario:
            network_data_path = pathlib.Path(scenario["grid_data_path"])
            weather_data_path = pathlib.Path(scenario["weather_data_path"])
        else:
            current_file = pathlib.Path(__file__).parents[0].absolute()
            data_root = current_file.parents[1] / "data" / "opfingen"
            network_data_path = data_root / "grid"
            weather_data_path = data_root / "weather_data.csv"

        if "heat_demand_data_path" in scenario:
            heat_data_path = pathlib.Path(scenario["heat_demand_data_path"])
            delimiter = data_io.get_csv_delimiter(heat_data_path)
            heat_demand = pd.read_csv(heat_data_path, sep=delimiter, index_col=0)
            heat_demand.index = pd.to_datetime(heat_demand.index, utc=True)
            self.heat_demand = data_io.set_tz_index_to_utc(heat_demand)
        else:
            heat_data_path = None
            self.heat_demand = pd.DataFrame(index=self.time_index, data={})  # empty DataFrame

        if not network_data_path.exists():
            raise FileNotFoundError(f"Network data at: {network_data_path}.")
        if (
            not weather_data_path.exists()
        ):  # TODO: Weather data is optional if heat demand is given for all agents
            raise FileNotFoundError(f"Weather data: {weather_data_path}.")

        delimiter = data_io.get_csv_delimiter(weather_data_path)
        weather_data = pd.read_csv(
            weather_data_path, index_col=0, date_format="%Y-%m-%d %H:%M:%S", sep=delimiter
        )
        try:
            weather_data.index = pd.to_datetime(weather_data.index, utc=True, unit="s")
        except ValueError:
            weather_data.index = pd.to_datetime(weather_data.index, utc=True)
            # weather_data.index = weather_data.index.tz_convert("Europe/Berlin")
        weather_data = weather_data.resample("15min").interpolate()

        # In case the weather data has ERA5 format
        if all(
            col not in weather_data.columns for col in ["Outside Temperature", "Solar Irradiation"]
        ):
            weather_data.rename(
                columns={"t": "Outside Temperature", "G": "Solar Irradiation"}, inplace=True
            )
        self.weather_data = data_io.set_tz_index_to_utc(weather_data)
        self.weather_data.index -= datetime.timedelta(days=365)

        # check if data on ev capacity is available
        ev_capacity_data = None
        if "ev_capacity_data_path" in scenario and scenario["ev"]:
            ev_capacity_data = pd.read_csv(
                scenario["ev_capacity_data_path"], index_col=0, date_format="%Y-%m-%d %H:%M:%S"
            )

        # TODO: These variable names do not look like bools.
        self._model_hps = scenario["hp"]
        self._model_bats = scenario["bat"]
        self._model_evs = scenario["ev"]
        # For debugging: be able to scale pv generators.
        self.pv_scale = scenario.get("pv_scale") or 1.0

        # if two of the three are true, set multi_bool to True
        self.multi_flex_bool = sum([self._model_hps, self._model_bats, self._model_evs]) > 1

        pickled_grid_path = config.data_path() / "tmp" / f"pickled_grid_{scenario['name']}.pkl"
        if os.path.exists(pickled_grid_path):
            print("Loading pickled grid (Delete pickle file if reload is necessary).")
            print(f"'rm {pickled_grid_path}'")
            with open(pickled_grid_path, "rb") as grid_tmp:
                self.grid = pickle.load(grid_tmp)
        else:
            self.grid = grid.Grid(network_data_path, self.dt_h, ev_capacity_data)
            print(f"Caching Grid object to {pickled_grid_path}")
            with open(pickled_grid_path, "wb") as grid_tmp:
                pickle.dump(self.grid, grid_tmp)

    def get_sys_ids(self):
        return self.grid.sys_ids

    def get_input_data(self, sys_id: str, scenario: Dict) -> pd.DataFrame:
        """
        Return dataframe with time series information.
        If the system has a flexibility, only weather ts data is needed for the heat pumps.
        If the system has an unused flexibility, load ts data.
        """

        assert sys_id in self.grid.sys_ids, f"Agent {sys_id} not known in grid data"
        load_ts = self.grid.get_load_ts()
        data = {
            f"{sys_id}_p_load": load_ts.loc[self.time_index, sys_id],
        }

        # check for PV data
        if any(sys_id in col for col in self.grid.pv_p.columns):
            unit_id = next((col for col in self.grid.pv_p.columns if sys_id in col), None)
            data[f"{unit_id}_p_ac"] = (
                self.pv_scale * self.grid.get_pv_ts().loc[self.time_index, unit_id]
            )

        # check for HP data
        if any(sys_id in col for col in self.grid.hp_p.columns) and self._model_hps:
            unit_id = next((col for col in self.grid.hp_p.columns if sys_id in col), None)
            data[f"{unit_id}_p"] = self.grid.get_hp_ts().loc[self.time_index, unit_id]
            for col_name in self.weather_data:
                data[col_name] = self.weather_data.loc[self.time_index, col_name]
        # check for heat demand data
        if any(col in sys_id for col in self.heat_demand.columns):
            # if the system has a heat demand, add it to the data
            # this is only used for the heat pump
            bus_name = next((col for col in self.heat_demand.columns if col in sys_id), None)
            data[f"{sys_id}_hp_heat_demand"] = self.heat_demand.loc[self.time_index, bus_name]

        # check for storage data
        if any(sys_id in key for key in self.grid.batteries.index) and self._model_bats:
            unit_id = next((key for key in self.grid.batteries.index if sys_id in key), None)
            data[f"{unit_id}_p"] = self.grid.get_bat_ts().loc[
                self.time_index, unit_id + "_p"
            ]  # load
            data[f"{unit_id}_soc"] = self.grid.get_bat_soc_ts().loc[
                self.time_index, unit_id + "_soc"
            ]  # soc

        # check for EV data
        if any(sys_id in key for key in self.grid.evs.index) and self._model_evs:
            unit_id = next((key for key in self.grid.evs.index if sys_id in key), None)
            # gets processed ev data
            data_dict = self.grid.get_ev_soc_ts().loc[self.time_index, unit_id].to_dict("series")
            data.update(data_dict)

            # Concatenate the new columns along axis=1 (columns)
            # ev_data = pd.concat([ev_data] + charging_data_list, axis=1)

        try:
            ts_data = pd.DataFrame()
            for key, value in data.items():
                ts_data[key] = value
        except ValueError as e:
            print(f"Error in {sys_id}")
            print(data)
            raise e

        # TODO @Rebecca, if you want to use time varying prices. Find a way to read your input here.
        # This example generates a time varying price which increases from 0.1 to 0.3
        # over the horizon
        # ts_data["c_sup"] = 0.2 * np.arange(len(ts_data.index)) / len(ts_data.index) + 0.1

        return ts_data

    def get_parameters(self, sys_id: str) -> dict:
        """Returns modelling parameters"""
        assert sys_id in self.grid.sys_ids, f"Requested sys_id '{sys_id}' not in available Ids"
        # create parameters for each unit in a system, store in a data frame

        params = dict()
        params[sys_id] = type_defs.SysParsLoad(sys_id, c_sup=0.3, c_feed=0.1, dt_h=0.25)

        # check if system has PV
        if any(sys_id in col for col in self.grid.pv_p.columns):
            unit_id = next((col for col in self.grid.pv_p.columns if sys_id in col), None)

            params[unit_id] = type_defs.SysParsPV(sys_id, c_sup=0.3, c_feed=0.1, dt_h=0.25)

        # check if system has heat pump
        if any(sys_id in col for col in self.grid.hp_p.columns) and self._model_hps:
            # print("hp: " + sys_id)
            # HP initialization can only include on-off heatpumps since information from synthetic
            # profiles is limited
            unit_id = next((col for col in self.grid.hp_p.columns if sys_id in col), None)
            heat_pump_size = self.grid.heat_pumps.loc[unit_id, "p_set"]
            if any(col in sys_id for col in self.heat_demand.columns):  # Heat demand available
                bus_name = next((col for col in self.heat_demand.columns if col in sys_id), None)
                heat_demand_ts = self.heat_demand[bus_name]
                thermal_mass, heat_rate = self.get_thermal_parameters_from_water_tank(
                    heat_demand_ts
                )
                params[unit_id] = type_defs.SysParsHeatPump(
                    sys_id,
                    c_sup=0.3,
                    c_feed=0.1,
                    dt_h=0.25,
                    p_max=heat_pump_size,
                    initial_temp=75,
                    temp_min_heat=70,
                    temp_max_heat=80,
                    temp_lower_bound=70,
                    temp_upper_bound=80,
                    thermal_mass=thermal_mass,  # Use water storage sizing model
                    heat_rate=heat_rate,
                )
            else:
                params[unit_id] = type_defs.SysParsHeatPump(
                    sys_id, c_sup=0.3, c_feed=0.1, dt_h=0.25, p_max=heat_pump_size
                )

        # check if system has storage
        if any(sys_id in col for col in self.grid.batteries.index) and self._model_bats:
            # print("bat: " + sys_id)
            # HP initialization can only include on-off heatpumps since information from synthetic
            # profiles is limited
            unit_id = next((col for col in self.grid.batteries.index if sys_id in col), None)
            p_bat = self.grid.batteries.loc[unit_id, "p_nom"]
            params[unit_id] = type_defs.SysParsPVBat(
                sys_id, c_sup=0.3, c_feed=0.1, dt_h=0.25, capacity=5, init_soc=0, p_inv=p_bat
            )

        # check if system has ev
        if any(sys_id in col for col in self.grid.evs.index) and self._model_evs:
            # HP initialization can only include on-off heatpumps since information from synthetic
            # profiles is limited
            unit_id = next((col for col in self.grid.evs.index if sys_id in col), None)
            p_bat = self.grid.evs.loc[unit_id, "p_nom"]
            capacity = self.grid.evs.loc[unit_id, "capacity"]
            params[unit_id] = type_defs.SysParsEV(
                sys_id,
                c_sup=0.3,
                c_feed=0.1,
                dt_h=0.25,
                capacity=capacity,
                init_soc=0.5,
                target_soc=1,
                p_inv=p_bat,
            )
        return params
        # ev_params = dict(system="ev", eff=0.9, init_soc=0.5, p_lim_dc=11., p_lim_ac=11.)

    def get_thermal_parameters_from_water_tank(
        self, heat_demand_ts: pd.Series, time_buffer=5, temp_decrease=5
    ) -> float:
        """
        Size the water storage based on the heat demand time series.
        The size is determined by the percentile 80 of the heat demand, considering a 5C temperature decrease with
        a buffer time of 3 hours
        :param heat_demand_ts: Time series of heat demand in kW
        :return tuple: thermal_mass in kJ/K, heat_transfer_rate in kW/K m2
        """
        # # Hardcoded parameters - These values shouldn't be so distant from reality
        # Heat transfer rate is a rough value from:
        # https://www.spiraxsarco.com/learn-about-steam/steam-engineering-principles-and-heat-transfer/energy-consumption-of-tanks-and-vats
        heat_transfer_coefficient = 11.5 / 1000 * 0.1  # kW/K m2
        water_cp = 4.19355  # kJ/kg K at 75°C - This value can be found anywhere
        water_density = 977.76  # kg/m³ at 75°C - This value can be found anywhere

        # Getting thermal mass
        time_buffer_seconds = time_buffer * 3600
        sizing_heat_demand = heat_demand_ts.max()
        if sizing_heat_demand <= 0:
            raise ValueError("Sizing heat demand must be positive.")
        thermal_mass = sizing_heat_demand * time_buffer_seconds / temp_decrease

        # Getting tank dimensions
        water_volume = thermal_mass / (water_cp * water_density)
        length_diameter_ratio = 1.0  # TODO: Harcoded
        tank_diameter = (water_volume * 4 / (np.pi * length_diameter_ratio)) ** (1 / 3)
        tank_length = length_diameter_ratio * tank_diameter
        if tank_length > 2:  # Harcoded: A very large tank wont fit in a room
            tank_length = 2
            tank_diameter = (4 * water_volume / (np.pi * length_diameter_ratio)) ** (1 / 2)
        tank_surface_area = np.pi * tank_diameter * (tank_length + tank_diameter / 2)

        # Getting heat transfer rate
        heat_transfer_rate = heat_transfer_coefficient * tank_surface_area

        return thermal_mass, heat_transfer_rate
