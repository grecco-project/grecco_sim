from curses import meta
from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from grecco_sim.simulator import results
from grecco_sim.util import style, type_defs


@dataclass
class PlottingMetaInf:
    """Dataclass for meta information needed for plotting."""

    n_agents: int
    sys_ids: List[str]
    sim_tag: str
    color_list = [xkcd_color[1] for xkcd_color in list(mcolors.XKCD_COLORS.items())]


def make_plots(
    sim_result: results.SimulationResult,
):
    """Make plots for visual analysis."""
    if not sim_result.run_pars.plot:
        # Nothing to do
        return

    meta_inf = PlottingMetaInf(
        sys_ids=list(sim_result.ts_grid.columns.values),
        n_agents=len(sim_result.ts_grid.columns),
        sim_tag=sim_result.run_pars.sim_tag,
    )
    os.makedirs(sim_result.run_pars.output_file_dir / "plots", exist_ok=True)
    # Plot grid everytime
    _plot_grid(sim_result.ts_grid, meta_inf, sim_result.run_pars)

    if True:
        # Plot these 'analysis' plots only when showing
        if (
            sim_result.run_pars.scenario["bat"] == True
            or sim_result.run_pars.scenario["ev"] == True
        ):
            _plot_batteries(sim_result.flex_ts, meta_inf, sim_result.run_pars)
            _plot_soc(sim_result.agents_ts, meta_inf, sim_result.run_pars)
        if sim_result.run_pars.scenario["hp"] == True:
            _plot_hps(
                sim_result.flex_ts,
                sim_result.agents_ts,
                sim_result.assigned_grid_fees,
                meta_inf,
                sim_result.run_pars,
            )
        if sim_result.run_pars.coordination_mechanism != "local_self_suff":
            _plot_signals(sim_result.assigned_grid_fees, meta_inf)
        _plot_flex_nodes(
            sim_result.agents_ts,
            sim_result.flex_ts,
            sim_result.assigned_grid_fees,
            meta_inf,
            sim_result.run_pars,
        )
    if sim_result.run_pars.plot:
        # plt.show()
        pass


def _plot_grid(
    ts_grid: pd.DataFrame, meta_inf: PlottingMetaInf, run_pars: type_defs.RunParameters
):
    """Plot the grid time series with combined and individual agents."""
    fig, ax = style.styled_plot(
        xlabel="Time",
        ylabel="Load at transformer / kW",
        figsize="landscape",
        # ylim=(0.0, 140.0),
        title="Trafo Load",
    )

    grid_max = ts_grid.loc[:, meta_inf.sys_ids].sum(axis=1).max()
    grid_min = ts_grid.loc[:, meta_inf.sys_ids].sum(axis=1).min()
    print(f"Grid max: {grid_max}, min: {grid_min}")

    if (ts_grid.index[-1] - ts_grid.index[0]).total_seconds() < 3600 * 24 * 10:
        # Only plot individual agents if time range is sufficiently small.
        ts_grid["_prev_sys"] = 0

        for i, sys_id in enumerate(meta_inf.sys_ids):
            ax.fill_between(
                ts_grid.index,
                ts_grid["_prev_sys"],
                ts_grid[sys_id] + ts_grid["_prev_sys"],
                color=meta_inf.color_list[i],
                alpha=0.2,
                step="post",
            )
            ax.plot(
                ts_grid.index,
                ts_grid[sys_id] + ts_grid["_prev_sys"],
                label=None,
                drawstyle="steps-post",
                color=meta_inf.color_list[i],
            )
            ts_grid["_prev_sys"] += ts_grid[sys_id]

    ax.plot(
        ts_grid.loc[:, meta_inf.sys_ids].sum(axis=1),
        label="Sum",
        # linestyle="dashed",
        color="black",
        drawstyle="steps-post",
    )

    # ax.legend(title=meta_inf.sim_tag)
    ax.legend(title=run_pars.sim_tag)
    ax.y_lim = (grid_min * 1.1, grid_max * 1.1)
    ax.grid()
    fig.tight_layout()
    fig.savefig(run_pars.output_file_dir / f"plot_grid_{run_pars.sim_tag}.pdf")
    fig.savefig(run_pars.output_file_dir / "plots" / "Trafo_Load.png")


def _plot_hps(
    ts_hp: pd.DataFrame,
    agents_ts: dict[str, pd.DataFrame],
    ts_signals: pd.DataFrame,
    meta_inf: PlottingMetaInf,
    run_pars: type_defs.RunParameters,
):

    fig, ax = style.styled_plot(
        xlabel="Time", ylabel="Power Heat Pumps", figsize="landscape", title="Heat Pumps"
    )

    sum_hp = np.zeros(len(ts_hp.index))
    _label = "Agents"
    for i, col_name in enumerate(ts_hp.columns.values):
        if "hp_p_in" in col_name:
            ax.plot(
                ts_hp[col_name],
                label=_label,
                drawstyle="steps-post",
                color=meta_inf.color_list[i],
                linestyle="dashed",
            )
            sum_hp += ts_hp[col_name].values
            if _label == "Agents":
                _label = None
            # ax.plot(ts_hp.index, sum_hp, label="All heat pumps", drawstyle="steps-post", color="black")
    if meta_inf.sim_tag not in ["local_self_suff", "none"]:
        ax.plot(
            ts_signals.index,
            ts_signals["signal"] * 10,
            label="Signal",
            drawstyle="steps-post",
            color="black",
        )

    ax.legend()
    fig.tight_layout()
    fig.savefig(run_pars.output_file_dir / "plots" / "hp_p.png")

    fig_temp, ax_temp = style.styled_plot(
        xlabel="Time", ylabel="Temperature / C", figsize=(8, 6), title="Heat Pump Temperatures"
    )
    for ag_name, ts_ag in agents_ts.items():
        if "hp_temp" in ts_ag:
            ax_temp.plot(ts_ag["hp_temp"], label=ag_name, drawstyle="steps-post")
    fig_temp.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig_temp.tight_layout(rect=[0, 0.1, 1, 1])
    fig_temp.savefig(run_pars.output_file_dir / "plots" / "temperature.png")


def _plot_batteries(
    ts_bat: pd.DataFrame, meta_inf: PlottingMetaInf, run_pars: type_defs.RunParameters
):

    fig, ax = style.styled_plot(
        xlabel="Time", ylabel="Power", figsize="landscape", title="Battery/EV Power"
    )

    sum_bat = np.zeros(len(ts_bat.index))
    _label = "Agents"
    for i, col_name in enumerate(ts_bat.columns.values):
        if "p_ac" in col_name:
            ax.plot(
                ts_bat.index,
                ts_bat[col_name],
                label=_label,
                drawstyle="steps-post",
                color=meta_inf.color_list[i],
                linestyle="dashed",
            )
            sum_bat += ts_bat[col_name].values
            if _label == "Agents":
                _label = None

    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(run_pars.output_file_dir / "plots" / "Bat_EV_p.png")


def _plot_signals(ts_signals: pd.DataFrame, meta_inf: PlottingMetaInf):
    fig, ax = style.styled_plot(
        xlabel="Time", ylabel="Costs / Euro", figsize="landscape", title="Signal incurred costs"
    )

    for i, sys_id in enumerate(meta_inf.sys_ids):
        ax.plot(
            ts_signals.index,
            ts_signals[f"fee_{sys_id}"],
            label=sys_id,
            drawstyle="steps-post",
            color=meta_inf.color_list[i],
        )

    if meta_inf.sim_tag not in ["local_self_suff", "none"]:
        ax.plot(
            ts_signals.index,
            ts_signals["signal"],
            label="Signal",
            drawstyle="steps-post",
            color="black",
        )

    if len(ts_signals) < 13:
        ax.legend()

    fig.tight_layout()


def _plot_soc(
    ts_agents: dict[str, pd.DataFrame],
    meta_inf: PlottingMetaInf,
    run_pars: type_defs.RunParameters,
):
    fig, ax = style.styled_plot(
        xlabel="Time", ylabel="SoC", figsize="landscape", title="SoC of Storages/EVs"
    )

    for i, (ag_name, ts_ag) in enumerate(ts_agents.items()):
        if "bat_soc" in ts_ag:
            ax.plot(
                ts_ag.index,
                ts_ag["bat_soc"],
                label=ag_name,
                drawstyle="steps-post",
                color=meta_inf.color_list[i],
            )
        if "ev_soc" in ts_ag:
            ax.plot(
                ts_ag.index,
                ts_ag["ev_soc"],
                label=ag_name,
                drawstyle="steps-post",
                color=meta_inf.color_list[i],
            )
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(run_pars.output_file_dir / "plots" / "Bat_EV_soc.png")


def _plot_flex_nodes(
    ts_agents: dict[str, pd.DataFrame],
    ts_flex: pd.DataFrame,
    ts_signals: pd.DataFrame,
    meta_inf: PlottingMetaInf,
    run_pars: type_defs.RunParameters,
):
    fig, ax = style.styled_plot(
        xlabel="Time", ylabel="Power in kW", figsize="landscape", title="Nodes with Flexibility"
    )
    # plot only the nodes that have PV and Flexibility
    # iterate over ts_flex columns and assign integer to fetch color from color_list
    pvs = {}

    for i, unit in enumerate(ts_flex.columns):
        for suffix, linestyle in {
            "_ev_p_ac": "dashdot",
            "_hp_p_in": "dotted",
            "_bat_p_ac": "dashed",
        }.items():
            if unit.endswith(suffix):
                bus = unit.replace(suffix, "")
                if "p_el_pv" in ts_agents[bus].columns:
                    # Always plot unit if PV exists
                    ax.plot(
                        ts_flex.index,
                        ts_flex[unit],
                        drawstyle="steps-post",
                        linestyle=linestyle,
                        color=meta_inf.color_list[i],
                    )
                    # Add bus to pvs only the first time for PV plotting later
                    if bus not in pvs:
                        pvs[bus] = i

    # Plot PV data only once per bus
    for bus, i in pvs.items():
        ax.plot(
            ts_flex.index,
            ts_agents[bus]["p_el_pv"],
            label=bus,
            drawstyle="steps-post",
            color=meta_inf.color_list[i],
        )
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="small", frameon=False
        )
    if meta_inf.sim_tag not in ["local_self_suff", "none"]:
        ax.plot(
            ts_signals.index,
            ts_signals["signal"] * 10,
            label="Signal",
            drawstyle="steps-post",
            color="black",
        )

    fig.tight_layout()
    fig.savefig(run_pars.output_file_dir / "plots" / "Flex.png")


def make_agent_plots(kpis_ag: pd.DataFrame, run_pars: results.SimulationResult, agents="all"):
    """Make plots from agent evaluation."""
    if not run_pars.plot:
        return
    agents_opt = {
        "flex": kpis_ag[
            kpis_ag[["battery_energy", "ev_energy", "hp_energy_el"]].sum(axis=1) > 0
        ].index.to_list(),
        "all": kpis_ag.index.tolist(),
    }

    _make_general_plot(kpis_ag, agents_opt[agents], run_pars)
    _make_economics_plot(kpis_ag, agents_opt[agents])
    _make_pv_plot(kpis_ag, agents_opt[agents])
    _make_battery_plot(kpis_ag, agents_opt[agents])
    _make_ev_plot(kpis_ag, agents_opt[agents])
    _make_hp_plot(kpis_ag, agents_opt[agents])
    plt.show()


def _make_general_plot(df: pd.DataFrame, agents: List[str], run_pars: type_defs.RunParameters):
    """Make general plots for agents."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle("General KPIs of Agents")

    axs[0].boxplot(df.loc[agents, ["grid_demand"]].values, labels=["Grid Demand"])
    axs[0].set_ylabel("Energy [kWh]")
    axs[0].set_title("Total Demand")

    axs[1].boxplot(df.loc[agents, ["max_grid_demand"]].values, labels=["Max Demand from Grid"])
    axs[1].set_ylabel("Power [kW]")
    axs[1].set_title("Peak Demand")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(run_pars.output_file_dir / "plots" / "Agents_P_E.png")


def _make_economics_plot(df: pd.DataFrame, agents: List[str]):
    """Make economics plots for agents."""
    costs = df.loc[agents, "costs"]
    profit = df.loc[agents, "profit"]
    revenue = df.loc[agents, "revenue"]

    # filter nan values: either drop or fill with 0
    profit_all = profit.fillna(0)
    profit = profit.dropna()
    revenue = -costs[profit_all.values < 0] - profit
    revenue_all = -costs - profit_all

    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    fig.suptitle("Economic KPIs of Agents")

    ax[0].boxplot(
        costs.values,
        labels=["Costs"],
    )
    ax[0].set_ylabel("EUR")
    ax[0].set_title("Costs")

    ax[1].boxplot(
        [profit, profit_all],
        labels=["Profit (has PV)", "Profit (all)"],
    )
    ax[1].set_ylabel("EUR")
    ax[1].set_title("Profit")

    ax[2].boxplot(
        [revenue, revenue_all],
        labels=["Revenue (has PV)", "Revenue (all)"],
    )
    # ax[2].set_yscale("symlog")
    ax[2].set_ylabel("EUR")
    ax[2].set_title("Revenue")

    plt.tight_layout()


def _make_pv_plot(df: pd.DataFrame, agents: List[str]):
    """Make PV plots for agents."""
    df = df[["total_feed", "self-consumption", "max_feed"]].copy()
    mask = df["total_feed"] < 0
    df = abs(df[mask])

    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    fig.suptitle("PV KPIs of Agents")

    axs[0].boxplot(
        df["total_feed"].values,
        labels=["Total Feed-in"],
    )
    axs[0].set_ylabel("Energy [kWh]")
    axs[0].set_title("PV feed-in")

    axs[1].boxplot(
        df["self-consumption"].values,
        labels=["Self-consumption"],
    )
    axs[1].set_ylabel("Energy [kWh]")
    axs[1].set_title("PV self consumption")

    axs[2].boxplot(df[["max_feed"]].values, labels=["Max Feed-in Power"])
    axs[2].set_ylabel("Power [kW]")
    axs[2].set_title("PV Feed-in Peak")

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def _make_battery_plot(df: pd.DataFrame, agents: List[str]):
    """Make battery plots for agents."""
    df = df[
        [
            "battery_energy",
            "battery_energy_from_grid",
            "battery_energy_to_grid",
            "battery_max_from_grid",
            "battery_max_to_grid",
            "charging_cycle_equivalents",
            "overcharge_bat",
            "undercharge_bat",
        ]
    ]
    # df.dropna(inplace=True)
    # mask = df["battery_energy"] > 0
    # df = df[mask]
    # df.fillna(0, inplace=True)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Battery KPIs of Agents")

    # Energy-related metrics
    axs[0].boxplot(
        df.loc[agents, ["battery_energy", "battery_energy_from_grid", "battery_energy_to_grid"]]
        .dropna()
        .values,
        labels=["Total", "From Grid", "To Grid"],
    )
    axs[0].set_ylabel("Energy [kWh]")
    axs[0].set_title("Battery Energy Flows")

    # Power and violations
    x1 = pd.to_numeric(df["battery_max_from_grid"], errors="coerce").dropna().values
    x2 = pd.to_numeric(df["battery_max_to_grid"], errors="coerce").dropna().values

    axs[1].boxplot([x1, x2], labels=["P max Grid", "P max Feed"])
    axs[1].set_ylabel("Power [kW]")
    axs[1].set_title("Battery Power")

    axs[2].boxplot(
        df[["charging_cycle_equivalents", "overcharge_bat", "undercharge_bat"]].dropna().values,
        labels=["Charging Cycle Equivalents", "Overcharge", "Undercharge"],
    )
    axs[2].set_ylabel("Count")
    axs[2].set_title("Battery SOC status")

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def _make_ev_plot(df: pd.DataFrame, agents: List[str]):
    """Make EV plots for agents."""
    df = df[["ev_energy", "ev_energy_from_grid", "overcharge_ev", "undercharge_ev"]].copy()
    mask = df["ev_energy"] > 0
    df = df[mask]
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle("EV KPIs of Agents")

    axs[0].boxplot(
        df[["ev_energy", "ev_energy_from_grid"]].values,
        labels=["EV Energy", "From Grid"],
    )
    axs[0].set_ylabel("Energy [kWh]")
    axs[0].set_title("EV Charging")

    axs[1].boxplot(
        df[["overcharge_ev", "undercharge_ev"]].values,
        labels=["Overcharge", "Undercharge"],
    )
    axs[1].set_ylabel("Count")
    axs[1].set_title("EV SOC Violations")

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def _make_hp_plot(df: pd.DataFrame, agents: List[str]):
    """Make HP plots for agents."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle("Heat KPIs of Agents")

    axs[0].boxplot(df.loc[agents, ["hp_energy_el"]].dropna().values, labels=["Electric Demand"])
    axs[0].set_ylabel("Energy [kWh]")
    axs[0].set_title("Electric Demand")

    axs[1].boxplot(df.loc[agents, ["mean_temp"]].dropna().values, labels=["Mean Temperature"])
    axs[1].set_ylabel("Temperature [°C]")
    axs[1].set_title("Mean Temperature")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
