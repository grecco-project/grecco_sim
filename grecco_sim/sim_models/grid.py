import warnings
import datetime as dt
from typing import Optional

from grecco_sim.util import data_io
import pandas as pd
import numpy as np
import pypsa
import pathlib
import difflib
import logging

logger = logging.getLogger(__name__)


class Grid(object):
    def __init__(
        self,
        grid_path: pathlib.Path,
        dt_h: dt.timedelta,
        ev_extra_data: Optional[pd.DataFrame] = None,
    ):

        self.network = pypsa.Network()
        self.ev_extra_data = ev_extra_data
        self.dt_h = dt_h

        logger.info("Reading pypsa files ...")
        with warnings.catch_warnings(action="ignore"):
            self.network.import_from_csv_folder(grid_path)
        logger.info("... Done.")

        self._init_variables()
        logger.info("Grid object created")

    @staticmethod
    def _sys_id(system: str, bus: int):
        return f"bus_{bus}_load_{system}"

    def _init_variables(self):
        """
        Decision:
            Loads are aggregated on a bus level.
            Later distinction between several loads at each bus is possible.
        """

        _grid: pypsa.Network = self.network

        _loads = _grid.loads.loc[_grid.loads.carrier != "heat_pump", :]
        self._ind_load = _loads.index.to_list()

        # Take unique buses
        self._buses = list(set(_loads.bus))

        _hps = _grid.loads.loc[_grid.loads.carrier == "heat_pump", :]
        self._ind_hps = _hps.index.to_list()

        _gens = _grid.generators[_grid.generators.carrier == "solar"]
        self._ind_gens = _gens.index.to_list()

        _bats = _grid.storage_units[_grid.storage_units.type == "h0_battery"]
        self._ind_bats = _bats.index.to_list()

        _evs = _grid.storage_units[_grid.storage_units.type != "h0_battery"]

        self._ind_evs = _evs.index.to_list()

        # Take unique buses
        self._buses = list(set(_loads.bus))

        # Assert generators and storages are connected to a bus where also a load is connected
        assert set(_loads.bus) == set(_loads.bus) | set(_hps.bus) | set(_gens.bus) | set(
            _evs.bus
        ) | set(_bats.bus)

        # A system is identified by a building containing a load
        # Generate << a >> bus to load mapping
        self.bus_to_load = _loads.reset_index().groupby("bus").first()["Load"]
        self.sys_ids = [self._sys_id(self.bus_to_load.loc[b], b) for b in self._buses]

        # TODO: the following steps should probably be put in functions
        # rename load ts with sys_id and scale to kW
        self.load_p = (
            self.network.loads_t["p_set"]
            .loc[:, self._ind_load]
            .groupby(self.network.loads["bus"], axis=1)
            .sum()
        )
        self.load_p = self.load_p.rename(
            columns={b: self._sys_id(self.bus_to_load.loc[b], b) for b in self.load_p.columns}
        )
        self.load_p *= 1000.0  # Transform to kW instead of MW

        # rename pvs with sys_id_pv and scale PV input data
        self.pv_p = self.network.generators_t["p_set"].loc[:, self._ind_gens]
        self.pv_p = self.pv_p.rename(
            columns={
                i: self._sys_id(self.bus_to_load.loc[_gens.bus[i]] + "_pv", _gens.bus[i])
                for i in _gens.index
            }
        )
        self.pv_p *= 1000.0  # Transform to kW instead of MW

        self.generators = _gens.rename(
            index={
                i: self._sys_id(self.bus_to_load.loc[_gens.bus[i]] + "_pv", _gens.bus[i])
                for i in _gens.index
            }
        )
        self.generators["p_set"] *= 1000.0
        self.generators = _gens.rename(
            index={
                i: self._sys_id(self.bus_to_load.loc[_gens.bus[i]] + "_pv", _gens.bus[i])
                for i in _gens.index
            }
        )

        # Rename and scale heat pumps
        self.heat_pumps = _hps.rename(
            index={
                i: self._sys_id(self.bus_to_load.loc[_hps.bus[i]] + "_hp", _hps.bus[i])
                for i in _hps.index
            }
        )
        self.heat_pumps["p_set"] *= 1000.0
        if "s_nom" in self.heat_pumps.columns:
            self.heat_pumps["s_nom"] *= 1000.0

        self.hp_p = self.network.loads_t["p_set"][self._ind_hps]

        self.hp_p = self.hp_p.rename(
            columns={
                i: self._sys_id(
                    self.bus_to_load.loc[_hps.bus[i]] + "_hp", self.network.loads.loc[i, "bus"]
                )
                for i in self._ind_hps
            }
        )
        self.hp_p *= 1000.0

        # TODO: check if soc is necessary for bats - it shouldnt be necessary
        # Rename and scale batteries
        self.batteries = _bats.rename(
            index={
                i: self._sys_id(self.bus_to_load.loc[_bats.bus[i]] + "_bat", _bats.bus[i])
                for i in _bats.index
            }
        )
        self.batteries["p_nom"] *= 1000.0

        self.bat_p = self.network.storage_units_t["p_set"][self._ind_bats]
        self.bat_p = self.bat_p.rename(
            columns={
                i: self._sys_id(self.bus_to_load.loc[_bats.bus[i]] + "_bat_p", _bats.bus[i])
                for i in self._ind_bats
            }
        )
        self.bat_soc = self.network.storage_units_t["state_of_charge"][self._ind_bats]
        self.bat_soc = self.bat_soc.rename(
            columns={
                i: self._sys_id(self.bus_to_load.loc[_bats.bus[i]] + "_bat_soc", _bats.bus[i])
                for i in self._ind_bats
            }
        )

        self.bat_p *= 1000.0  # Transform to kW instead of MW

        # Rename and scale EVs
        self.evs = self.get_ev_params(_evs)
        self.evs["p_nom"] *= 1000.0
        self.evs["counts"] = self.evs.groupby("bus").cumcount()

        self.evs = _evs.rename(
            index={
                i: self._sys_id(
                    self.bus_to_load.loc[_evs.bus[i]] + "_" + str(_evs.loc[i, "counts"]) + "_ev",
                    _evs.bus[i],
                )
                for i in _evs.index
            }
        )

        self.ev_p = self.network.storage_units_t["p_set"].loc[:, self._ind_evs]
        self.ev_p = self.ev_p.rename(
            columns={
                i: self._sys_id(
                    self.bus_to_load.loc[_evs.bus[i]] + "_" + str(_evs.loc[i, "counts"]) + "_ev_p",
                    _evs.bus[i],
                )
                for i in self._ind_evs
            }
        )
        self.ev_p *= 1000.0  # Transform to kW instead of MW

        self.ev_soc = self.network.storage_units_t["state_of_charge"][self._ind_evs]
        self.ev_soc = self.ev_soc.rename(
            columns={
                i: self._sys_id(
                    self.bus_to_load.loc[_evs.bus[i]]
                    + "_"
                    + str(_evs.loc[i, "counts"])
                    + "_ev_soc",
                    _evs.bus[i],
                )
                for i in self._ind_evs
            }
        )

        # ts data for EVs
        self.ev_data = self.get_ev_data(_evs)
        self.ev_data = self.ev_data.rename(
            columns={
                i: self._sys_id(
                    self.bus_to_load.loc[_evs.bus[i]] + "_" + str(_evs.loc[i, "counts"]) + "_ev",
                    _evs.bus[i],
                )
                for i in self._ind_evs
            }
        )
        self.ev_data.columns = pd.MultiIndex.from_tuples(
            [(left, f"{left}_{right}") for left, right in self.ev_data.columns]
        )

    def get_load_ts(self) -> pd.DataFrame:
        return self.load_p

    def get_pv_ts(self) -> pd.DataFrame:
        return self.pv_p

    def get_hp_ts(self) -> pd.DataFrame:
        return self.hp_p

    def get_bat_ts(self) -> pd.DataFrame:
        return self.bat_p

    def get_ev_ts(self) -> pd.DataFrame:
        return self.ev_p

    def get_bat_soc_ts(self) -> pd.DataFrame:
        return self.bat_soc

    def get_ev_ts(self) -> pd.DataFrame:
        return self.ev_data

    def get_pv_data(self):
        installed_pv = self.network.generators[self.network.generators.carrier == "solar"].p_set
        pv_generation = self.network.generators_t.p_set.loc[:, installed_pv.index]
        # To calculate the capacity factor as time series we have to divide the generation over installed_pv
        capacity_factor = pv_generation / installed_pv
        return {
            "installed_pv": installed_pv,
            "pv_generation": pv_generation,
            "capacity_factor": capacity_factor,
        }

    def get_ev_params(self, _evs):
        # TODO ? put this at a more suitable place
        _evs.loc[:, "capacity"] = 60.0  # Default value
        # Check if extra data has capacity information.
        if self.ev_extra_data is not None:
            for ev in _evs.index:
                try:
                    capacity = self.ev_extra_data.loc[ev, "capacity"]
                except KeyError:
                    name = difflib.get_close_matches(ev, self.ev_extra_data.index, 1)
                    capacity = self.ev_extra_data.loc[name, "capacity_kWh"].values

                _evs.loc[ev, "capacity"] = capacity

        return _evs

    def get_ev_data(self, _evs):
        plugged_in = self.network.storage_units_t.plugged_in[self._ind_evs].fillna(0)
        plugged_in = plugged_in.rename(
            columns={
                i: self._sys_id(
                    self.bus_to_load.loc[_evs.bus[i]] + "_" + str(_evs.loc[i, "counts"]) + "_ev",
                    _evs.bus[i],
                )
                for i in self._ind_evs
            }
        )
        soc = self.ev_soc.copy()
        soc.columns = soc.columns.str.replace("_soc", "", regex=False)
        ev_data_in = pd.concat(
            {"cp": plugged_in, "soc": soc},
            axis=1,
        )

        # Swap MultiIndex levels
        ev_data_in.columns = ev_data_in.columns.swaplevel(0, 1)
        # Sort index to group 'A' and 'B' together
        ev_data = ev_data_in.sort_index(axis=1)
        if ev_data.index.tz is None:
            ev_data = ev_data.tz_localize("UTC")
            # ev_data = ev_data.tz_convert("Europe/Berlin")

        # interpolate soc data blockiwse while plugged in and get parking duration
        # Compute and add "soc" and "until_departure" for each sys_id
        # Store computed charging data in a list
        charging_data_list = []

        # Iterate over each system ID (level 0 column)
        for sys_id in ev_data.columns.levels[0]:
            try:
                charging_data = data_io.get_charging_data(ev_data[sys_id], self.dt_h)
            except:
                print(f"faulty data for {sys_id}")

            charging_data.columns = pd.MultiIndex.from_product(
                [[sys_id], charging_data.columns]
            )  # Ensure correct MultiIndex
            charging_data_list.append(charging_data)

        # Concatenate the new columns along axis=1 (columns)
        ev_data = pd.concat([ev_data] + charging_data_list, axis=1)
        # remove tz information
        ev_data.index = ev_data.index.tz_localize(None)
        return ev_data

    def rename_duplicates(self, type: str) -> pd.DataFrame:
        evs = self.evs.copy()
        evs["counts"] = evs.groupby("bus").cumcount()  # 0,1,2,... within each bus
        is_dup = evs["bus"].duplicated(keep=False)  # True for all rows in dup groups

        # map is_dup false index to load via bus_to_load
        bus_to_load = self.bus_to_load
        load_indices = self.evs.loc[is_dup].values
        # append _2 for second instance, _3 for third instance, etc.
        load_indices = [f"{name}_{i+2}" for i, name in enumerate(load_indices)]

        new_names = np.where(is_dup, "_" + (counts + 1).astype(str), "") + "_" + type
        ev_map = {
            i: self._sys_id(name, bus) for i, name, bus in zip(_evs.index, evs_unique, _evs["bus"])
        }

        return ev_map

    def get_load_data(self):
        load_index = self.network.loads[self.network.loads.carrier != "heat_pump"].index
        load_buses = self.network.loads.loc[load_index, "bus"]
        load_active_power = self.network.loads_t.p_set.loc[:, load_index]
        try:
            load_reactive_power = self.network.loads_t.q_set.loc[:, load_index]
        except KeyError:
            # Since no reactive power data is available, we assume that the reactive power is 0
            # This has to be changed in the future to include realistic reactive power data otherwise
            # the power flow calculation might not converge
            # within iterations limit, assuming all buses are PV buses would be incorrect since this
            # won't allow us to see voltage drops (Voltage control assumption)
            load_reactive_power = load_active_power * 0

        return {
            "load_index": load_index,
            "load_bus_map": load_buses,
            "load_active_power": load_active_power,
            "load_reactive_power": load_reactive_power,
        }

    def get_heatpump_data(self):
        heatpump_index = self.network.loads[self.network.loads.carrier == "heat_pump"].index
        heatpump_buses = self.network.loads.loc[heatpump_index, "bus"]
        heatpump_rated_power = self.network.loads.loc[heatpump_index, ["p_set", "q_set"]]
        heatpump_active_power_profile = self.network.loads_t.p_set.loc[:, heatpump_index]
        return {
            "heatpump_index": heatpump_index,
            "heatpump_buses": heatpump_buses,
            "heatpump_rated_power": heatpump_rated_power,
            "heatpump_active_power_profile": heatpump_active_power_profile,
        }

    def calculate_timeseries_powerflow(self, n_timesteps: int = 672, path=None, export_csv=False):
        """
        Function calculates power flow and generates csv files (time series data).

        Required to check congestion, visualize the bus voltage levels, and avoid
        repetitive power flow calculations (which can be time-consuming).
        """

        # Non-linear power flow
        self.network.pf(self.network.snapshots[:n_timesteps])

        # Export the results
        if export_csv:
            self.network.export_to_csv_folder(path)

        return self.network

    def _set_active_load(self, load_set: pd.Series):
        """
        This methods is used to set the power of each load in the grid.
        Only the loads that are in the load_set will be set. Other loads will remain unchanged.
        This allows the user to set Heatpump and EV loads separately.

        :param load_set: DataFrame with the columns as load name and the index as the snapshot range

        return: None
        """

    def _set_reactive_load(self, load_set: pd.Series):
        """
        This methods is used to set the reactive power of each load in the grid.
        Only the loads that are in the load_set will be set. Other loads will remain unchanged.
        This allows the user to set Heatpump and EV loads separately.

        :param load_set: DataFrame with the columns 'q_mvar' and the index as the load id

        return: None
        """
        load_set.columns = ["q_mvar"]

        self.net.load.q_mvar.loc[load_set.index, :] = load_set

    def _set_active_sgen(self, sgen_set: pd.Series):
        """
        This methods is used to set the power of each sgen in the grid.
        Only the sgens that are in the sgen_set will be set. Other sgens will remain unchanged.
        This allows the user to set PV generation separately.

        :param sgen_set: DataFrame with the columns 'p_mw' and the index as the sgen id

        return: None
        """
        sgen_set.columns = ["p_mw"]

        self.net.sgen.p_mw.loc[sgen_set.index, :] = sgen_set

    def _set_reactive_sgen(self, sgen_set: pd.Series):
        """
        This methods is used to set the reactive power of each sgen in the grid.
        Only the sgens that are in the sgen_set will be set. Other sgens will remain unchanged.
        This allows the user to set PV generation separately.

        :param sgen_set: DataFrame with the columns 'q_mvar' and the index as the sgen id

        return: None
        """
        sgen_set.columns = ["q_mvar"]

        self.net.sgen.q_mvar.loc[sgen_set.index, :] = sgen_set

    def check_congestion(self):
        """
        Function for evaluating the state of the grid regarding the congestion.

        Return:
        Return:
            congestion_results: dict
                Dictionary containing the KPIs of the congestion calculations.
        """

        # Check if time series data is available (e.g. for lines)
        count = 0
        for key, df in self.network.lines_t.items():
            if df.empty:
                count += 1

        if count == len(self.network.lines_t):
            raise Exception(
                "Calculate the power flow using the 'calculate_timeseries_powerflow' function, \n"
                "or import the results of the power flow calculations if they are available."
            )
        else:

            # Evaluating transformer loading

            # Evaluating transformer loading
            # Trafo_loading= (Power_transfered(at time t)/ Nominal capacity)*100
            trafo_nominal_capacity = self.network.transformers["s_nom"]
            trafo_power_transferred = np.sqrt(
                self.network.transformers_t["p1"] ** 2 + self.network.transformers_t["q1"] ** 2
            )
            # trafo_power_transferred = self.network.transformers_t["p1"]
            trafo_loading_perc = (trafo_power_transferred / trafo_nominal_capacity) * 100

            # Events of congestion in the transformer
            # Events_of_congestion= 1 if Trafo_loading>80% and 0 if Trafo_loading<=80%
            trafo_events_of_congestion = (trafo_loading_perc > 80).astype(int)
            trafo_total_events_of_congestion = trafo_events_of_congestion.sum()

            # Evaluating line loading
            line_nominal_capacity = self.network.lines["s_nom"]
            line_power_transferred = np.sqrt(
                self.network.lines_t["p1"] ** 2 + self.network.lines_t["q1"] ** 2
            )
            # line_power_transferred = self.network.lines_t["p1"]
            line_power_transferred = np.sqrt(
                self.network.lines_t["p1"] ** 2 + self.network.lines_t["q1"] ** 2
            )
            # line_power_transferred = self.network.lines_t["p1"]
            line_loading_perc = (line_power_transferred / line_nominal_capacity) * 100

            # CLI = CL / G (congestion line index)
            line_events_of_congestion = (line_loading_perc > 80).astype(int)
            line_total_events_of_congestion = line_events_of_congestion.sum()
            count_congested_lines = (line_total_events_of_congestion > 0).sum()
            congestion_line_index = count_congested_lines / len(self.network.lines)

            # CVI = sum of power transferred / installed line capacity
            # (congestion volume index) - this considers only one timestep
            # CVI = sum of power transferred / installed line capacity
            # (congestion volume index) - this considers only one timestep
            congestion_volume_index = (line_power_transferred.iloc[0] / line_nominal_capacity).sum()

            # Weighted average is performed to consider the importance of line capacity;
            # Weighted average is performed to consider the importance of line capacity;
            # large lines are more important than the smaller ones
            # Weighted average = weight * max_power_flow in line / line capacity
            # weight = line capacity / sum of all line capacities
            max_power_flow_line = line_power_transferred.max()
            # weight = line capacity / sum of all line capacities
            max_power_flow_line = line_power_transferred.max()
            weight = line_nominal_capacity / line_nominal_capacity.sum()
            weighted_average = 0
            weighted_average = 0
            for i in range(len(max_power_flow_line)):
                weighted_average += weight[i] * max_power_flow_line[i] / line_nominal_capacity[i]

            congestion_results = {
                "trafo_loading_perc": trafo_loading_perc,
                "trafo_total_events_of_congestion": trafo_total_events_of_congestion,
                "line_loading_perc": line_loading_perc,
                "line_total_events_of_congestion": line_total_events_of_congestion,
                "congestion_line_index": congestion_line_index,
                "congestion_volume_index": congestion_volume_index,
                "weighted_average": weighted_average,
            }

            return congestion_results

    def estimate_power_flow(self):
        # return ptdf matrix for the network
        # pp.rundcpp(self.net)
        # _, ppci = _pd2ppc(self.net)

        # ptdf_sparse = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
        #                     using_sparse_solver=True)
        # return ptdf_sparse
        pass
