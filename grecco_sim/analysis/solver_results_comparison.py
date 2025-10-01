import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import itertools


class SolverComparer:
    """A class to compare results from different optimization solvers (here run1 and run2) in a simulation.
    It's not a very smart code as it reads out historical data from the simulation results. The user therefor needs to know the names of the runs to compare.
    It reads data from  CSV files, computes energy summaries, and generates plots
    for flexible devices like electric vehicles (EVs) and heat pumps (HPs).
    """

    def __init__(self, data_to_compare):
        self.runs = list(data_to_compare.keys())
        data = list(data_to_compare.values())
        self.run1 = data[0]  # run1
        self.run2 = data[1]  # run2
        self.none_run = data_to_compare["none"]  # none
        self.base_path = pathlib.Path(__file__).parent.parent.parent
        self.results_path = self.base_path / "results" / "default"
        self.charging_processes = pd.DataFrame()
        self.loss_data = {}
        self.diff = pd.DataFrame()
        self.rmse = pd.Series(dtype=float)

        self.dir = pathlib.Path.cwd() / "results" / "comparisons"
        self.dir.mkdir(exist_ok=True)

    def read_data(self):
        def read_solver_data(path, file_nodes, file_flex, file_signals):
            return (
                pd.read_csv(path / file_nodes, sep=";", index_col=0),
                pd.read_csv(path / file_flex, sep=";", index_col=0),
                pd.read_csv(path / file_signals, sep=";", index_col=0) if file_signals else None,
            )

        run1_path, run2_path, none_path = [
            self.results_path / r for r in (self.run1, self.run2, self.none_run)
        ]
        # collect dataframes
        dfs = []

        if self.runs[0] == "self_suff":
            self.run1_nodes_data, self.srun1_flex_data, self.run1_signal_data = read_solver_data(
                run1_path, "nodal_load_local_self_suff.csv", "flex_power_local_self_suff.csv", None
            )
            dfs.extend(self.run1_nodes_data)

        elif self.runs[0] in ["gurobi", "bonmin", "osqp", "plain_grid_fee"]:
            self.run1_nodes_data, self.run1_flex_data, self.run1_signal_data = read_solver_data(
                run1_path,
                "nodal_load_plain_grid_fee.csv",
                "flex_power_plain_grid_fee.csv",
                "signals_costs_plain_grid_fee.csv",
            )
            dfs.extend([self.run1_nodes_data, self.run1_signal_data])

        if self.runs[1] == "self_suff":
            self.run2_nodes_data, self.run2_flex_data, self.run2_signal_data = read_solver_data(
                run2_path, "nodal_load_local_self_suff.csv", "flex_power_local_self_suff.csv", None
            )
            dfs.append(self.run2_nodes_data)

        elif self.runs[1] in ["gurobi", "bonmin", "osqp", "plain_grid_fee"]:
            self.run2_nodes_data, self.run2_flex_data, self.run2_signal_data = read_solver_data(
                run2_path,
                "nodal_load_plain_grid_fee.csv",
                "flex_power_plain_grid_fee.csv",
                "signals_costs_plain_grid_fee.csv",
            )
            dfs.extend([self.run2_nodes_data, self.run2_signal_data])

        self.none_nodes_data, self.none_flex_data, self.none_signal_data = read_solver_data(
            none_path, "nodal_load_none.csv", "flex_power_none.csv", None
        )
        dfs.append(self.none_nodes_data)

        self.charging_processes = pd.read_csv(
            self.base_path / "data" / "tmp" / "charging_data_2024.csv",
            sep=",",
            index_col=0,
            parse_dates=True,
            header=[0, 1],
        )

        self.time_index = dfs[0].index
        self.charging_processes = self.charging_processes.loc[self.time_index]

        for df in dfs:
            df.drop(columns="unixtimestamp", inplace=True)
            df.sort_index(axis=1, inplace=True)
            df["Trafo"] = df.sum(axis=1)
            if "iterations" in df.columns:
                df.drop(columns="iterations", inplace=True)

    def compute_energy_summary(self):
        def calc_demand_feed(df):
            demand = df[df["Trafo"] > 0]["Trafo"].sum() * 0.25
            feed_in = df[df["Trafo"] < 0]["Trafo"].sum() * 0.25
            return demand, feed_in

        for label, df in zip(
            [self.runs[0], self.runs[1], "none"],
            [self.run1_nodes_data, self.run2_nodes_data, self.none_nodes_data],
        ):
            demand, feed_in = calc_demand_feed(df)
            print(f"Demand at trafo for {label}: {demand:.2f} kWh")
            print(f"Feed in at trafo for {label}: {feed_in:.2f} kWh\n")

    def get_flexible_device_columns(self):
        cols = self.run1_flex_data.columns
        p_hps = cols[cols.str.contains("hp_p")]
        soc_evs = cols[cols.str.contains("ev_soc")]
        hps = [c.replace("_hp_p_in", "") for c in p_hps]
        evs = [c.replace("_soc", "") for c in soc_evs]

        return hps, evs

    def plot_ev_socs(self):
        _, evs = self.get_flexible_device_columns()
        colors = ["red", "green", "blue", "orange"]
        for clr, ev in enumerate(evs):
            mask = self.charging_processes[ev, ev + "_until_departure"].notna()
            in_region = False
            start = None
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(len(self.charging_processes)):
                if mask.iloc[i] and not in_region:
                    start = self.charging_processes.index[i]
                    in_region = True
                elif not mask.iloc[i] and in_region:
                    ax.axvspan(
                        start, self.charging_processes.index[i], color=colors[clr], alpha=0.3
                    )
                    in_region = False
            if in_region:
                ax.axvspan(start, self.charging_processes.index[-1], color=colors[clr], alpha=0.3)

            ax.plot(
                x,
                self.charging_processes[ev, ev + "_target_soc"],
                color="grey",
                linewidth=1,
                linestyle="-",
                alpha=0.8,
            )
            ax.plot(
                x,
                self.charging_processes[ev, ev + "_initial_soc"],
                color="grey",
                linewidth=1,
                linestyle="-",
                alpha=0.8,
            )
            ax.plot(
                x,
                self.none_flex_data[ev + "_soc"],
                label="none",
                color="black",
                linewidth=1,
            )
            ax.plot(
                x,
                self.run1_flex_data[ev + "_soc"],
                label=self.runs[0],
                color=colors[clr],
                linewidth=1,
                linestyle="--",
            )
            ax.plot(
                x,
                self.run2_flex_data[ev + "_soc"],
                label=self.runs[1],
                color=colors[clr],
                linewidth=1,
                linestyle="-.",
            )
            ax.plot(
                x,
                self.run1_signal_data["signal"] * 4,
                label="signal",
                color="limegreen",
                linewidth=1,
            )
            ax.set_title(f"SOC Comparison for EV {ev}")
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.dir / f"_{ev}.png")
        plt.show()

    def plot_hp_behavior(self):
        hps, _ = self.get_flexible_device_columns()
        colors = plt.cm.viridis.colors
        color_dict = {k: c for k, c in zip(hps, itertools.cycle(colors))}

        for hp in hps:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                self.time_index,
                self.run1_flex_data[hp + "_hp_p_in"],
                label=self.runs[0],
                color=color_dict[hp],
                linestyle="--",
            )
            ax.plot(
                self.time_index,
                self.run2_flex_data[hp + "_hp_p_in"],
                label=self.runs[1],
                color=color_dict[hp],
                linestyle="-.",
            )
            ax.plot(
                self.time_index,
                self.none_flex_data[hp + "_hp_p_in"],
                label="none",
                color=color_dict[hp],
                linestyle="-",
            )
            ax.set_title(f"HP Power Input: {hp}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Power in kW")
            ax.grid(True, axis="y", linestyle="-", linewidth=1.2, color="gray", alpha=0.2)
            ax.legend()
            plt.tight_layout()
        plt.show()

    def compare_node_loads(self):
        self.diff = self.run1_nodes_data - self.run2_nodes_data
        self.rmse = (self.diff**2).mean()

        # calculate signal induced costs (sum)
        hps, evs = self.get_flexible_device_columns()

        # total costs
        if "self_suff" in self.runs or "none" in self.runs:
            print("No two runs with signals to compare.")

        else:
            costs_run2 = self.run2_signal_data.sum()
            costs_run1 = self.run1_signal_data.sum()

            costs_run2.drop(columns="signal", inplace=True)
            costs_run1.drop(columns="signal", inplace=True)

            over = 0
            under = 0
            same = 0

            for col in costs_run2.index:
                excess_costs = costs_run1[col] - costs_run2[col]
                if excess_costs > 0:
                    over += 1
                    print(
                        f"Excess costs for {col} in {self.runs[0]} compared to {self.runs[1]}: {excess_costs:.2f}."
                    )
                elif excess_costs < 0:
                    under += 1
                    print(
                        f"Excess costs for {col} in {self.runs[0]} compared to {self.runs[1]}: {-excess_costs:.2f}."
                    )
                else:
                    same += 1
                    print(f"Equal costs for {col}.")
            print(
                f"{over} columns have excess costs in {self.runs[0]} compared to {self.runs[1]}, {under} in {self.runs[1]}, and {same} are equal."
            )
        # costs for hps
        # costs for evs

    def plot_node_loads(self):
        for col in self.diff.columns:
            if col == "Trafo":
                enlarge = 3000
            else:
                enlarge = 100
            if self.rmse[col] > 1:
                print(f"Column {col} has an RMSE > 1.")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x, self.run1_nodes_data[col], label=self.runs[0], color="blue")
                ax.plot(x, self.run2_nodes_data[col], label=self.runs[1], color="orange")
                ax.plot(x, self.none_nodes_data[col], label="none", color="green")
                ax.set_title(col)
                ax.set_xlabel("Time")
                ax.set_ylabel("Load in kW")
                ax.grid(True, axis="y", linestyle="-", linewidth=1.2, color="gray", alpha=0.2)
                ax.legend()
                plt.tight_layout()
                ax.plot(
                    x,
                    self.run1_signal_data["signal"] * enlarge,
                    label="signal",
                    color="black",
                    linewidth=1,
                )
        plt.show()

    def load_hp_losses(self):

        hps, _ = self.get_flexible_device_columns()
        path = self.base_path / "results" / "default"
        agent_files = "agents"
        for hp in hps:
            bus = hp + ".csv"
            loss_data = {
                self.runs[0]: pd.read_csv(
                    path / self.run1 / agent_files / bus, sep=";", index_col=0
                ),
                self.runs[1]: pd.read_csv(
                    path / self.run2 / agent_files / bus, sep=";", index_col=0
                ),
                "none": pd.read_csv(path / self.none_run / agent_files / bus, sep=";", index_col=0),
            }

            self.loss_data[hp] = loss_data

            for df in self.loss_data[hp].values():
                df["cum_losses"] = df["hp_losses"].cumsum()

            print(
                f"Total losses for {hp} in {self.runs[0]}: {loss_data[self.runs[0]]['cum_losses'].iloc[-1]} kWh. This is {loss_data[self.runs[0]]['cum_losses'].iloc[-1] / loss_data['none']['cum_losses'].iloc[-1] * 100:.2f}% of the none losses."
            )
            print(
                f"Total losses for {hp} in {self.runs[1]}: {loss_data[self.runs[1]]['cum_losses'].iloc[-1]} kWh. This is {loss_data[self.runs[1]]['cum_losses'].iloc[-1] / loss_data['none']['cum_losses'].iloc[-1] * 100:.2f}% of the none losses."
            )
            print(f"Total losses for {hp} in none: {loss_data['none']['cum_losses'].iloc[-1]} kWh.")

    def plot_hp_losses(self):
        for hp, data in self.loss_data.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax2 = ax.twinx()

            """ax.plot(
                self.time_index,
                data[self.runs[0]]["hp_losses"],
                label=self.runs[0] + " losses",
                color="lightgreen",
                linestyle=":",
            )"""
            ax.plot(
                self.time_index,
                data[self.runs[0]]["hp_p_in"],
                label=self.runs[0] + " hp power",
                color="green",
                linestyle="--",
            )
            """ax2.plot(
                self.time_index,
                data[self.runs[0]]["hp_temp"],
                label=self.runs[0] + " temp",
                color="darkgreen",
                linestyle="-",
            )

            ax.plot(
                self.time_index,
                data[self.runs[1]]["hp_losses"],
                label=self.runs[1] + " losses",
                color="deepskyblue",
                linestyle=":",
            )"""
            ax.plot(
                self.time_index,
                data[self.runs[1]]["hp_p_in"],
                label=self.runs[1] + " hp power",
                color="blue",
                linestyle="--",
            )
            """ax2.plot(
                self.time_index,
                data[self.runs[1]]["hp_temp"],
                label=self.runs[1] + " temp",
                color="midnightblue",
                linestyle="-",
            )

            ax.plot(
                self.time_index,
                data["none"]["hp_losses"],
                label="none losses",
                color="lightcoral",
                linestyle=":",
            )"""
            ax.plot(
                self.time_index,
                data["none"]["hp_p_in"],
                label="none hp power",
                color="orange",
                linestyle="--",
            )
            """ax2.plot(
                self.time_index,
                data["none"]["hp_temp"],
                label="none temp",
                color="darkorange",
                linestyle="-",
            )"""
            ax.plot(
                self.time_index,
                self.run1_signal_data["signal"] * 15,
                label="signal",
                color="red",
                linewidth=1.5,
            )

            ax.set_xlabel("Time")
            ax.set_ylabel("Losses and Power kW")
            ax2.set_ylabel("Temperature in °C")
            ax.set_title(f"{hp} heat pump")
            ax.grid(True, axis="y", linestyle="-", linewidth=1.2, color="gray", alpha=0.2)
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data_to_compare = {
        "plain_grid_fee": "250917_0820",
        "self_suff": "250917_0819",
        "none": "250917_0818",
    }

    comparer = SolverComparer(data_to_compare)
    comparer.read_data()
    comparer.compute_energy_summary()
    comparer.get_flexible_device_columns()
    x = pd.DatetimeIndex(comparer.time_index).tz_convert("UTC").tz_localize(None)
    comparer.plot_ev_socs()
    # comparer.plot_hp_behavior()
    comparer.compare_node_loads()
    comparer.plot_node_loads()
    # comparer.load_hp_losses()
    # comparer.plot_hp_losses()
