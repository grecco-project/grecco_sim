import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from grecco_sim.sim_models import models


class HouseholdSimulator:
    """Used to simulate and plot the results of a single household."""

    def __init__(
        self,
        sys_id: str,
        horizon: int,
        dt_h: float,
        household: models.Household,
        ts_in: pd.DataFrame,
    ):
        self.sys_id = sys_id
        self.horizon = horizon
        self.dt_h = dt_h
        self.household = household

    def simulate(self, goal: str):
        """
        Simulate a single household.

        Different goals are used for different simulations:
        goal = "simple" -> difference in pv-power and load-power is put directly into the battery,
          battery is discharged until empty, grid tekes the rest

        no other goals yet
        """
        if goal == "simple":
            for time in range(self.horizon):  # Simulation
                control = (
                    self.household.get_state()["pv_generation"]
                    - self.household.get_state()["load_power"]
                )
                self.household.apply_control(control)
            res = self.household.get_output()
            return res
        elif goal == "age_to_80":
            for time in range(self.horizon):  # Simulation
                control = (
                    self.household.get_state()["pv_generation"]
                    - self.household.get_state()["load_power"]
                )
                self.household.apply_control(control)
            res = self.household.get_output()
            battery_died = np.where(res["bat"]["soh_sep"] > 0.8)[0]
            if len(battery_died) > 0:
                for sub in res:
                    if isinstance(res[sub], dict):
                        for array in res[sub]:
                            res[sub][array] = res[sub][array][: battery_died[-1]]
                    else:
                        res[sub] = res[sub][: battery_died[-1]]
                return res
            else:
                return res
        else:  ## space for other (future) goals
            raise ValueError(f"Goal '{goal}' not implemented yet.")

    def plot_results(self, res: dict, what_plot: str, num_days, grid: bool):
        """
        Used to plot the results of the simulation.
        what_plot takes one string with all the things to plot:

        "p_pv"   -> plots PV-generation
        "p_load" -> plots load consumption
        "p_net"  -> plots charging power for the battery
        "p_grid" -> plots grid supply/feed-in
        "soc_bat"-> plots the state of charge of the battery
        "soh_bat"-> plots the state of health of the battery
        "soh_cal"
        "soh_cyc"

        "all" -> plots all above
        """
        pv_generation = res["pv"]["p_ac"]
        load_consumption = res["load"]["p_load"]
        charging_power = res["bat"]["p_net"]
        grid_interaction = res["grid"]

        battery_soc = res["bat"]["soc"]

        battery_soh = res["bat"]["soh_sep"]
        battery_soh_tgth = res["bat"]["soh_tgth"]
        battery_soh_cal = res["bat"]["soh_cal"]
        battery_soh_cyc = res["bat"]["soh_cyc"]

        aging_rate = res["bat"]["aging_rate"]
        aging_rate = aging_rate[1:]

        time = np.linspace(0, num_days, len(res["grid"]) - 1)

        fig, ax1 = plt.subplots()

        # ax1.plot(time, aging_rate, color="blue", label= "aging_rate")

        if what_plot == "all":
            what_plot = "p_pv-p_load-p_net-p_grid-soc_bat-soh_bat-bat_age_ratio"

        if "p" in what_plot:
            if "p_pv" in what_plot:
                ax1.plot(time, pv_generation, color="red", label="PV-Gerneration")

            if "p_load" in what_plot:
                ax1.plot(time, load_consumption, color="blue", label="load consumption")

            if "p_net" in what_plot:
                ax1.plot(time, charging_power, color="orange", label="battery charging")

            if "p_grid" in what_plot:
                ax1.plot(time, grid_interaction, color="green", label="grid supply/feed-in")

            ax1.set_xlabel("time in days")
            ax1.set_ylabel("p in kW")
            ax1.set_xlim(0, time[-1])
            plt.legend()

            if "bat" in what_plot:
                ax2 = ax1.twinx()

                if "soc_bat" in what_plot:
                    battery_soc = battery_soc[1:]
                    ax2.plot(time, battery_soc, label="battery soc")
                    ax2.set_ylabel("SoC in %/100")

                if "soh_bat" in what_plot:
                    battery_soh = battery_soh[1:]
                    battery_soh_tgth = battery_soh_tgth[:-1]
                    ax2.plot(time, battery_soh, label="battery soh_sep")
                    # ax2.plot(time, battery_soh_tgth, label= "battery soh_tgth")
                    if ax2.get_ylabel() == "SoC in %/100":
                        ax2.set_ylabel("SoH and SoC in %/100")
                    else:
                        ax2.set_ylabel("SoH in %/100")

                if "bat_age_ratio" in what_plot:
                    battery_soh_cal = battery_soh_cal[1:]
                    battery_soh_cyc = battery_soh_cyc[1:]
                    ratio = battery_soh_cyc / (battery_soh_cyc + battery_soh_cal)
                    ax2.plot(time, ratio, label="age-ratio")

        else:
            if "soc_bat" in what_plot:
                battery_soc = battery_soc[:-1]
                ax1.plot(time, battery_soc, label="battery soc")
                ax1.set_ylabel("SoC in %/100")

            if "soh_bat" in what_plot:
                battery_soh = battery_soh[:-1]
                battery_soh_tgth = battery_soh_tgth[:-1]
                ax1.plot(time, battery_soh, label="battery soh")
                # ax1.plot(time, battery_soh_tgth, label= "battery soh_tgth")
                if ax1.get_ylabel() == "SoC in %/100":
                    ax1.set_ylabel("SoH and SoC in %/100")
                else:
                    ax1.set_ylabel("SoH in %/100")

            if "bat_age_ratio" in what_plot:
                battery_soh_cal = battery_soh_cal[1:]
                battery_soh_cyc = battery_soh_cyc[1:]
                ratio = (1 - battery_soh_cyc) / ((1 - battery_soh_cyc) + (1 - battery_soh_cal))
                ax1.plot(time, ratio, label="age-ratio")

        fig.tight_layout()
        plt.xlabel("time in days")
        plt.legend()
        if grid:
            plt.grid()
        plt.show()
