"""This script was written by Nilklas Maertens in his Bachelor thesis on battery ageing.

There's probably some reactivating to be done when trying to recover the results.
The goal was to investigate a ODE model for battery ageing.
"""

import os
import pandas as pd
import numpy as np

from grecco_sim.sim_models import models
from grecco_sim.simulator import household_simulator
from grecco_sim.util import type_defs


# actual data import
class DataImporter:

    def __init__(self, sys_id: str, horizon: int, dt_h: float):
        self.sys_id = sys_id
        self.horizon = horizon
        self.dt_h = dt_h

    def get_data_small(self, datatype: str) -> pd.DataFrame:
        path = os.path.join("data", "sample_scenario", f"{datatype}_data.csv")
        csv_file = pd.read_csv(path, index_col=0)
        data = pd.DataFrame(csv_file)
        # ONLY FOR SINGLE HOUSEHOLDS
        data = pd.DataFrame(data["ag_66"])

        for column in data:
            if datatype == "pv":
                data.rename(columns={column: f"{self.sys_id}_{datatype}_p_ac"}, inplace=True)
            elif datatype == "load":
                data.rename(columns={column: f"{self.sys_id}_{datatype}_p_load"}, inplace=True)

        if self.horizon != len(data):
            raise ValueError(f"Horizon length should be set to {len(data)}")
        return data

    def get_data_big(self, data_path, datatype, sys_num):
        csv_file = pd.read_csv(data_path, index_col=0)
        data = pd.DataFrame(csv_file)

        data = pd.DataFrame(data[str(sys_num)])

        for column in data:
            if datatype == "pv":
                data.rename(columns={column: f"{self.sys_id}_{datatype}_p_ac"}, inplace=True)
            elif datatype == "load":
                data.rename(columns={column: f"{self.sys_id}_{datatype}_p_load"}, inplace=True)

        # if self.horizon != len(data):
        #     raise ValueError(f"Horizon length should be set to {len(data)}")
        return data

    def merge_data_big(self, data_paths: list[str], datatypes: list[str], sys_num) -> pd.DataFrame:
        full_data = pd.DataFrame()
        for i in range(len(data_paths)):
            dataset = self.get_data_big(data_paths[i], datatypes[i], sys_num)
            full_data = pd.concat([full_data, dataset], axis=1)
        return full_data

    def merge_data_small(self, datatypes):
        full_data = pd.DataFrame()
        for i in datatypes:
            dataset = self.get_data_small(i)
            full_data = pd.concat([full_data, dataset], axis=1)
        return full_data

    def cyclic_aging_test(self, num_steps_per_cyc):
        data = {
            f"{self.sys_id}_pv_p_ac": np.zeros(self.horizon),
            f"{self.sys_id}_load_p_load": np.zeros(self.horizon),
        }

        p_plus = np.zeros(self.horizon)

        p_minus = np.zeros(self.horizon)

        for i in range(self.horizon):
            if i % num_steps_per_cyc in np.linspace(
                0, int(num_steps_per_cyc / 2 - 1), int(num_steps_per_cyc / 2)
            ):
                p_plus[i] = (5 / 12) / 0.9
                p_minus[i] = 0
            elif i % num_steps_per_cyc in np.linspace(
                int(num_steps_per_cyc / 2), num_steps_per_cyc - 1, int(num_steps_per_cyc / 2)
            ):
                p_plus[i] = 0
                p_minus[i] = (5 / 12) * 10 / 11

        data = {f"{self.sys_id}_pv_p_ac": p_plus, f"{self.sys_id}_load_p_load": p_minus}

        ts_in_test = pd.DataFrame(data)
        return ts_in_test


class Run_Aging:
    def __init__(self, sim, sys_num, plot):
        self.sim = sim
        self.sys_num = sys_num
        self.plot = plot

        self.sys_id_test = "sys_id_test"

        self.dod_test = 0.5
        self.init_soc_test = 0.25
        self.init_soh_test = 0.999
        self.capacity_test = 1  # kWh
        self.p_inv_test = 15  # kW

        self.c_sup_test = 0.27  # €/kWh
        self.c_feed_test = 0.08  # €/kWh

        self.temp_test = 40 + 273.15

        self.a_soc_test = 2.8575
        self.b_soc_test = 0.60225

        self.a_char_test = 0.063
        self.b_char_test = 0.0971

        self.s_therm_test = (
            1.2 * 10 ** (-5) * np.exp(-((17126) / (8.314)) * (1 / self.temp_test - 1 / 298.15))
        )

        self.s_dod_test = 4.0253 * (self.dod_test - 0.6) ** 3 + 1.0923

        if sim == "cyc_test":
            self.num_days = 1
            self.num_FECS = 1
            self.num_cycs = int(self.num_FECS / self.dod_test)
            self.num_steps_per_cyc = 10

            self.horizon_test = self.num_cycs * self.num_steps_per_cyc
            self.d_th_test = (self.num_days * 24) / self.horizon_test

            self.time = np.linspace(0, self.num_days, self.horizon_test)

            data_importer = data_importer = DataImporter(
                sys_id=self.sys_id_test, horizon=self.horizon_test, dt_h=self.d_th_test
            )
            self.ts_in_test = data_importer.cyclic_aging_test(
                num_steps_per_cyc=self.num_steps_per_cyc
            )

        elif "big" in sim:
            self.d_th_test = 0.25
            self.horizon_test = 35040
            self.num_days = (self.horizon_test * self.d_th_test) / 24

            pv_data_path = os.path.join(
                "data",
                "Opfingen_Profiles_2023",
                "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
                "generators-p_set.csv",
            )
            load_data_path = os.path.join(
                "data",
                "Opfingen_Profiles_2023",
                "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
                "loads-p_set.csv",
            )

            self.time = np.linspace(0, self.num_days, self.horizon_test)

            data_importer = DataImporter(
                sys_id=self.sys_id_test, horizon=self.horizon_test, dt_h=self.d_th_test
            )
            data = (
                data_importer.merge_data_big(
                    [pv_data_path, load_data_path], ["pv", "load"], self.sys_num
                )
                * 1000
            )
            self.ts_in_test = data

            if sim == "big5":
                self.horizon_test = 35040 * 5
                self.time = np.linspace(0, self.num_days * 5, self.horizon_test)
                for i in range(4):
                    self.ts_in_test = pd.concat([self.ts_in_test, data], axis=0, ignore_index=True)

        elif sim == "age_to_80":
            self.d_th_test = 0.25
            self.horizon_test = 35040
            self.num_days = (self.horizon_test * self.d_th_test) / 24

            pv_data_path = os.path.join(
                "data",
                "Opfingen_Profiles_2023",
                "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
                "generators-p_set.csv",
            )
            load_data_path = os.path.join(
                "data",
                "Opfingen_Profiles_2023",
                "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all",
                "loads-p_set.csv",
            )

            self.time = np.linspace(0, self.num_days, self.horizon_test)

            data_importer = DataImporter(
                sys_id=self.sys_id_test, horizon=self.horizon_test, dt_h=self.d_th_test
            )
            data = (
                data_importer.merge_data_big(
                    [pv_data_path, load_data_path], ["pv", "load"], self.sys_num
                )
                * 1000
            )
            self.ts_in_test = data
            self.horizon_test = 35040 * 10
            self.time = np.linspace(0, self.num_days * 10, self.horizon_test)
            for i in range(9):
                self.ts_in_test = pd.concat([self.ts_in_test, data], axis=0, ignore_index=True)

        elif sim == "small":
            self.num_days = 1
            self.horizon_test = 97
            self.d_th_test = 0.25  # timestep in hours

            self.time = np.linspace(0, self.num_days, self.horizon_test)

            data_importer = DataImporter(sys_id=self.sys_id_test, horizon=97, dt_h=self.d_th_test)
            data = data_importer.merge_data_small(["pv", "load"])  # length  == horizon
            self.ts_in_test = data

        else:
            self.d_th_test = 0.25
            self.horizon_test = 35040
            self.num_days = 365
            self.ts_in_test = pd.DataFrame(
                pd.DataFrame(np.zeros((365, 2)), columns=["p_pv", "p_load"])
            )

            self.time = np.linspace(0, self.num_days, self.horizon_test)

        self.pv_bat_sys_test = type_defs.SysParsPVBatAging(
            name="test_system",
            dt_h=self.d_th_test,
            c_sup=self.c_sup_test,
            c_feed=self.c_feed_test,
            init_soc=self.init_soc_test,
            init_soh=self.init_soh_test,
            capacity=self.capacity_test,
            p_inv=self.p_inv_test,
            s_therm=self.s_therm_test,
            s_dod=self.s_dod_test,
            a_soc=self.a_soc_test,
            b_soc=self.b_soc_test,
            a_char=self.a_char_test,
            b_char=self.b_char_test,
        )

        self.household_test = models.HouseholdAging(
            sys_id=self.sys_id_test,
            horizon=self.horizon_test,
            dt_h=self.d_th_test,
            params=self.pv_bat_sys_test,
            ts_in=self.ts_in_test,
        )

        self.Household_Simulation = household_simulator.HouseholdSimulator(
            sys_id=self.sys_id_test,
            horizon=self.horizon_test,
            dt_h=self.d_th_test,
            household=self.household_test,
            ts_in=self.ts_in_test,
        )

    def run_aging(self):

        res = self.Household_Simulation.simulate(goal="age_to_80")

        return res

    def plot_results(self, res):

        self.Household_Simulation.plot_results(
            res, what_plot=self.plot, num_days=self.num_days, grid=True
        )

    def aging_ratios(self, res: dict):

        ratio = (1 - res["bat"]["soh_cyc"][-1]) / (
            (1 - res["bat"]["soh_cal"][-1]) + (1 - res["bat"]["soh_cyc"][-1])
        )

        return ratio

    def what_age_when(self, res):
        pass

    def financials(self, res: dict, bat_price):

        net_money = 0

        for p in res["grid"]:
            if p < 0:
                net_money -= p * self.c_feed_test * self.d_th_test
            else:
                net_money -= p * self.c_sup_test * self.d_th_test

        net_money += res["bat"]["soc"][-1] * self.capacity_test * self.c_feed_test

        bat_deg = 1 - (5 * res["bat"]["soh_sep"][-1] - 4)

        bat_loss = bat_deg * bat_price

        net_money -= bat_loss

        return net_money


if __name__ == "__main__":
    aging_sys = Run_Aging("age_to_80", 1, "soh_bat")
    res = aging_sys.run_aging()
    aging_sys.plot_results(res)
    print(aging_sys.financials(res, 500))
    # print(aging_sys.aging_ratios(res))
    # print(f'It took {len(res["grid"])/(4*24)} days, or {len(res["grid"])/(4*24*365)} years to age the battery to 80% capacity')
    # for i in range(20):
    #     print(run_aging(i))
