from dataclasses import dataclass
import dataclasses
import os
from typing import Any
import numpy as np
import pandas as pd
import pathlib

from grecco_sim.sim_models import battery
from grecco_sim.util import sig_types
from grecco_sim.simulator import results
from grecco_sim.util import type_defs
from grecco_sim.util import style


def _get_aggregated_ts_result(sim_results: results.SimulationResult):
    """
    Calculate the KPIs in DOCUMENTED_KPIS from the aggregated grid power profile:
    """
    # Define a set to keep track of the returned elements of the dict.
    DOCUMENTED_KPIS = {
        "dt_h",  # time step of profile in hours
        "max_load",  # maximum load (positive) of aggregated profile
        "max_feed",  # minimum of aggregated profile (feed-back)
        "agg_consumption",
        "agg_feed_back",
        "agg_signal",
        "congested_times",  # number of time steps where the grid was congested
    }

    res_load = sim_results.ts_grid.sum(axis=1)
    dt_h = (res_load.index[1] - res_load.index[0]).total_seconds() / 3600.0

    res = {}

    res["dt_h"] = dt_h

    res["max_load"] = max(0, res_load.max())
    res["max_feed"] = max(0, -res_load.min())

    res["agg_consumption"] = res_load.loc[res_load > 0].sum() * dt_h
    res["agg_feed_back"] = -res_load.loc[res_load < 0].sum() * dt_h

    # Signals
    sys_ids = sim_results.sys_ids
    res["agg_signal"] = (
        sim_results.assigned_grid_fees.loc[:, [f"fee_{sys_id}" for sys_id in sys_ids]].sum().sum()
    )

    # Congestions
    res["congested_times"] = len(res_load[res_load > sim_results.grid_pars.p_lim]) + len(
        res_load[res_load < -sim_results.grid_pars.p_lim]
    )

    windows = pd.DataFrame(columns=["time", "max_load", "max_feed"])
    for i, window in enumerate(res_load.rolling(10)):
        windows.loc[i, :] = [window.index[0], window.max(), -window.min()]

    assert set(res.keys()) == DOCUMENTED_KPIS, "Make sure that the return is correctly documented!"

    return res


# TODO bring the agent analysis result to the set approach from above. It seems more straightforward.
@dataclass
class AgentAnalysisResult:
    # Combined individual costs (owed to utility, no coordination penalties)
    costs_all: float
    # time series of summed energy in storages
    en_in_bat_ts: np.ndarray
    # Net charged energy over sim horizon
    charged_energy: float
    # Combined capacity of all individual storages
    combined_capacity: float
    # Summed up losses through battery operation
    losses_bat: float
    # feed in aggregated over all agents
    agg_feed: float
    # feed in from battery aggregated over all agents
    agg_bat_to_grid: float


def flex_analysis(
    agent_ts: dict[str, pd.DataFrame], sizing: dict[str, type_defs.SysParsPVBat], dt_h: float
):
    """Do some analysis of time series of individual agents."""
    costs_all = 0
    en_in_bat_ts = None

    combined_cap = 0.0
    en_loss = 0.0

    agg_feed = 0.0
    agg_bat_to_grid = 0.0

    for sys_id, df in agent_ts.items():

        pv = (
            df["p_el_pv"]
            if "p_el_pv" in df.columns
            else pd.Series(index=df.index, data=np.zeros(len(df.index)))
        )

        grid = df["grid"]
        supp = grid[grid > 0]
        feed = grid[grid < 0]

        costs_supp = (supp * df["c_sup"][grid > 0]).sum() * dt_h
        costs_feed = (feed * df["c_feed"][grid < 0]).sum() * dt_h

        costs_all += costs_supp + costs_feed

        bat_to_grid = grid + pv
        bat_to_grid = bat_to_grid[bat_to_grid < 0]
        agg_bat_to_grid += bat_to_grid.sum() * dt_h
        agg_feed += feed.sum() * dt_h

        if isinstance(sizing[sys_id], type_defs.SysParsPVBat):
            if en_in_bat_ts is None:
                en_in_bat_ts = df["bat_soc"].values * sizing[sys_id].capacity
            else:
                en_in_bat_ts += df["bat_soc"].values * sizing[sys_id].capacity

            combined_cap += sizing[sys_id].capacity

            en_loss += (df["bat_p_ac"] - df["bat_p_net"]).sum() * sizing[sys_id].dt_h

        charged_energy = 0.0 if en_in_bat_ts is None else en_in_bat_ts[-1] - en_in_bat_ts[0]

    return AgentAnalysisResult(
        costs_all, en_in_bat_ts, charged_energy, combined_cap, en_loss, agg_feed, agg_bat_to_grid
    )


def signal_analysis(run_params: type_defs.RunParameters, raw_output: dict[str, dict]):

    if not run_params.plot:
        return

    if run_params.coordination_mechanism not in ["admm", "second_order"]:
        print(
            f"No analysis to be made for '{run_params.coordination_mechanism}' coordination mechanism."
        )
        return

    fig_sig, ax_sig = style.styled_plot(title="Signal over time", ylabel="Signal", figsize=(16, 8))
    fig_ref, ax_ref = style.styled_plot(
        title="Reference power over time", ylabel="Power / kW", figsize=(16, 8)
    )

    for k, signal in list(raw_output.values())[0]["signals"].items():
        # Show change of signal over simulation horizon for one agent.
        if not isinstance(signal, sig_types.SecondOrderSignal):
            print(f"Signal not Second order signal but {type(signal)}")
            return

        sig = ax_sig.plot(
            np.arange(start=k, stop=k + signal.signal_len),
            signal.mul_lambda,
            label=f"k = {k}",
            drawstyle="steps-post",
        )
        ax_sig.plot(k, signal.mul_lambda[0], marker="s", color=sig[0].get_color(), label=None)
        ref = ax_ref.plot(
            np.arange(start=k, stop=k + signal.signal_len),
            signal.res_power_set,
            label=f"k = {k}",
            drawstyle="steps-post",
        )
        ax_ref.plot(k, signal.res_power_set[0], marker="s", color=ref[0].get_color(), label=None)

    ax_sig.legend()
    ax_ref.legend()

    fig_sig.tight_layout()
    fig_ref.tight_layout()


import pandas as pd
import numpy as np


def agent_analysis(agent_ts: dict[str, pd.DataFrame], dt_h: float) -> pd.DataFrame:
    """
    KPIs for individual agents
    - Costs, Profit, Revenue
    - Consumption (max, total, battery, ev), Feed in (max, total, PV, battery), Self consumption
    - Charging cycles of battery, battery metrics, violations
    """

    DOCUMENTED_KPIS = {
        "grid_demand",  # Total demand from grid (kWh)
        "energy_demand",  # Total energy demand (kWh)
        "max_grid_demand",  # Maximum demand from grid (kW)
        "total_feed",  # Total feed in to grid (kWh)
        "max_feed",  # Maximum feed in to grid (kW)
        "self-consumption",  # Self consumption (kWh)
        "self-sufficiency",  # Self-consumtion / (self consumption + total _demand)
        "charging_cycle_equivalents",  # Charging cycles of battery
        "battery_energy",  # Battery energy (kWh)
        "battery_energy_from_grid",  # Battery energy from grid (kWh)
        "battery_max_from_grid",  # Battery max from grid (kW)
        "battery_energy_to_grid",  # Battery energy to grid (kWh)
        "battery_max_to_grid",  # Battery max to grid (kW)
        "costs",
        "profit",
        "revenue",
        "overcharge_bat",
        "undercharge_bat",
        "hp_energy_el",
        "hp_p_max",
        "mean_temp",
        "overheating",
        "underheating",
        "above_t",  # temperatures above 23 degrees
        "under_t",  # temperatures below 18 degrees
        "ev_energy",
        "ev_energy_from_grid",
        "ev_max_from_grid",
        "ev_energy_to_grid",
        "overcharge_ev",
        "undercharge_ev",
    }

    energy_map = {
        "hp": "hp_energy_el",
        "ev": "ev_energy",
    }
    agents_res = pd.DataFrame(index=agent_ts.keys(), columns=list(DOCUMENTED_KPIS))

    for ag, df in agent_ts.items():
        systems = []
        res = {k: np.nan for k in DOCUMENTED_KPIS}

        # Basic consumption/cost metrics
        grid = df["grid"].to_numpy()
        c_sup = df["c_sup"].to_numpy()
        c_feed = df["c_feed"].to_numpy()

        demand_mask = grid >= 0
        feed_mask = grid < 0

        res.update(
            {
                "grid_demand": (grid[demand_mask] * dt_h).sum(),
                "max_grid_demand": grid[demand_mask].max(),
                "costs": (grid[demand_mask] * c_sup[demand_mask]).sum(),
            }
        )

        # Battery metrics
        if "bat_soc" in df.columns:
            bat_soc = df["bat_soc"].to_numpy()
            bat_p_ac = df["bat_p_ac"].to_numpy()
            bat_p_net = df["bat_p_net"].to_numpy()

            charge_mask = bat_p_net > 0
            grid_charge_mask = (bat_p_net > 0) & (grid > 0)
            grid_discharge_mask = (bat_p_ac < 0) & (grid < 0)

            try:
                battery_max_from_grid = bat_p_ac[grid_charge_mask].max()
            except ValueError:
                battery_max_from_grid = np.nan

            try:
                battery_max_to_grid = bat_p_ac[grid_discharge_mask].min()
            except ValueError:
                battery_max_to_grid = np.nan

            battery_energy = bat_p_net[charge_mask].sum() * dt_h

            res.update(
                {
                    "battery_energy": battery_energy,
                    "battery_energy_from_grid": bat_p_ac[grid_charge_mask].sum() * dt_h,
                    "battery_max_from_grid": battery_max_from_grid,
                    "battery_energy_to_grid": bat_p_ac[grid_discharge_mask].sum() * dt_h,
                    "battery_max_to_grid": battery_max_to_grid,
                    "overcharge_bat": int(np.count_nonzero(bat_soc > 1)),
                    "undercharge_bat": int(np.count_nonzero(bat_soc < 0)),
                    "charging_cycle_equivalents": calculate_ccs(bat_p_net, dt_h),
                }
            )

        # HP metrics
        #'hp_temp', 'hp_losses', 'hp_p_in', 'hp_q_hp', 'hp_u_ext'
        if "hp_temp" in df.columns:
            systems.append("hp")
            hp_temp = df["hp_temp"].to_numpy()
            hp_p_in = df["hp_p_in"].to_numpy()

            hp_mask = hp_p_in > 0
            upper_temp_mask = hp_temp > 23
            lower_temp_mask = hp_temp < 18

            try:
                hp_p_max = hp_p_in[hp_mask].max()
            except ValueError:
                hp_p_max = np.nan
            res.update(
                {
                    "mean_temp": hp_temp.mean(),
                    "hp_energy_el": hp_p_in[hp_mask].sum() * dt_h,
                    # only relevant if HP has variable power input:
                    "hp_p_max": hp_p_max,
                    "overheating": int(np.count_nonzero(upper_temp_mask)),
                    "underheating": int(np.count_nonzero(lower_temp_mask)),
                    "above_t": len(hp_temp[upper_temp_mask]),
                    "under_t": len(hp_temp[lower_temp_mask]),
                }
            )

        # EV metrics
        #'ev_soc', 'ev_p_ac', 'ev_p_net'
        if "ev_soc" in df.columns:
            systems.append("ev")
            ev_soc = df["ev_soc"].to_numpy()
            # Replace this with your actual CPS function if defined
            ev_p_ac = df["ev_p_ac"].to_numpy()
            ev_p_net = df["ev_p_net"].to_numpy()

            charge_mask_ev = ev_p_net > 0
            grid_charge_mask_ev = (ev_p_net > 0) & (grid > 0)
            grid_discharge_mask_ev = (ev_p_ac < 0) & (grid < 0)

            try:
                ev_max_from_grid = ev_p_ac[grid_charge_mask_ev].max()
            except ValueError:
                ev_max_from_grid = np.nan

            res.update(
                {
                    "ev_energy": ev_soc[charge_mask_ev].sum() * dt_h,
                    "ev_energy_from_grid": ev_p_ac[grid_charge_mask_ev].sum() * dt_h,
                    "ev_max_from_grid": ev_max_from_grid,
                    # "ev_energy_to_grid": ev_p_ac[grid_discharge_mask_ev].sum() * dt_h,
                    "overcharge_ev": int(np.count_nonzero(ev_soc > 1)),
                    "undercharge_ev": int(np.count_nonzero(ev_soc < 0)),
                }
            )

        # PV metrics
        if "p_el_pv" in df.columns:
            systems.append("pv")
            pv = df["p_el_pv"].to_numpy()
            profit = (grid[feed_mask] * c_feed[feed_mask]).sum()
            self_consumption = pv[demand_mask].sum() * dt_h
            self_consumption += (pv[feed_mask] + grid[feed_mask]).sum() * dt_h
            try:
                max_feed = grid[feed_mask].min()
            except ValueError:
                max_feed = np.nan
            res.update(
                {
                    "total_feed": grid[feed_mask].sum(),
                    "max_feed": max_feed,
                    "profit": profit,
                    "self-consumption": self_consumption,
                }
            )

        # Overall energy demand
        total_demand = (
            sum(res[energy_map[k]] for k in systems if k in energy_map) + df["p_el_load"]
        ).sum() * dt_h
        res["energy_demand"] = total_demand

        # self-consumption
        if "pv" in systems:
            if total_demand > 0:
                res["self-sufficiency"] = res["self-consumption"] / total_demand
            else:
                res["self-sufficiency"] = np.nan

        # Revenue calculation
        res["revenue"] = abs(res.get("profit", 0)) - res.get("costs", 0)

        agents_res.loc[ag] = res

    print("Agent analysis results")
    print(agents_res)

    agent_stats = {
        "mean": agents_res.astype(float).mean(),
        "median": agents_res.astype(float).median(),
        "25_quantile": agents_res.astype(float).quantile(0.25),
        "75_quantile": agents_res.astype(float).quantile(0.75),
        "max": agents_res.astype(float).max(),
        "min": agents_res.astype(float).min(),
    }

    return agents_res, agent_stats


def calculate_ccs(power: np.ndarray, dt_h: float) -> int:
    throughput = np.abs(power).sum() * dt_h
    count = throughput / 60.0

    return count


# ================== Write results to file =============================


def _write_to_files(
    run_params: type_defs.RunParameters,
    eval_res: dict[str, Any],
    file_name: str = "kpis",
):

    # Abbreviate
    out_dir = run_params.output_file_dir

    # check that output directory extists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Handle flat dict: one row
    if all(not isinstance(v, pd.Series) for v in eval_res.values()):
        eval_df = pd.DataFrame(
            {tag: [eval_res[tag]] for tag in eval_res}, index=[run_params.sim_tag]
        )
    # Handle nested dict: multiple rows
    else:
        eval_df = pd.DataFrame.from_dict(eval_res, orient="index")

    file_name = file_name + ".csv"
    eval_file_path = out_dir / file_name
    if os.path.exists(eval_file_path):
        eval_df = pd.concat([pd.read_csv(eval_file_path, index_col="tag"), eval_df])

    eval_df.to_csv(eval_file_path)


def evaluate_sim(sim_result: results.SimulationResult):
    # Evaluate Simulation Results ==========================================
    eval_res = _get_aggregated_ts_result(sim_result)
    eval_res["tag"] = sim_result.run_params.sim_tag

    eval_res.update(
        dataclasses.asdict(flex_analysis(sim_result.agents_ts, sim_result.sizing, eval_res["dt_h"]))
    )
    eval_res.pop("en_in_bat_ts")

    # eval_res.update(agent_analysis(sim_result.agents_ts, eval_res["dt_h"]))

    eval_res["calc_time"] = sim_result.exec_time
    agent_kpis, agent_stats = agent_analysis(sim_result.agents_ts, eval_res["dt_h"])

    print(f"Analysis result for sim {sim_result.run_params.sim_tag}: {pd.Series(eval_res)}")
    _write_to_files(sim_result.run_params, eval_res, "kpis")

    for key in agent_stats:
        pd.concat([agent_stats[key], pd.Series(sim_result.run_params.sim_tag, index=["tag"])])
    print(
        f"Mean agent analysis result for sim {sim_result.run_params.sim_tag}: {agent_stats['mean']}"
    )
    _write_to_files(sim_result.run_params, agent_stats, "agent_stats")

    # Reactivate if needed. However, sim_result must get a signals field
    # signal_analysis(sim_result.run_params, sim_result.agents_ts)

    res = {"general_res": eval_res, "agent_res": agent_kpis, "agent_stats": agent_stats}

    return res


def compare_kpis(kpis_self_suff: dict[str, Any], kpis_none: dict[str, Any]):
    """
    Compare the KPIs of two simulations.
    :param kpis_self_suff: KPIs of the self sufficiency simulation
    :param kpis_none: KPIs of the none simulation
    """
    # consumption
    print(
        "Comparison of KPIs: Consumption self suff:",
        kpis_self_suff["agg_consumption"],
        "Consumption none:",
        kpis_none["agg_consumption"],
    )
    print(f"share {kpis_self_suff['agg_consumption'] / kpis_none['agg_consumption']:.2%} of none")
    print(
        "Maximum load self suff:",
        kpis_self_suff["max_load"],
        "Maximum load none:",
        kpis_none["max_load"],
    )
    print(f"share {kpis_self_suff['max_load'] / kpis_none['max_load']:.2%} of none")

    print(
        "Maximum feed in self suff:",
        kpis_self_suff["max_feed"],
        "Maximum feed in none:",
        kpis_none["max_feed"],
    )
    print(f"share {kpis_self_suff['max_feed'] / kpis_none['max_feed']:.2%} of none")

    print("Costs self suff:", kpis_self_suff["costs_all"], "Costs none:", kpis_none["costs_all"])
    print(f"share {kpis_self_suff['costs_all'] / kpis_none['costs_all']:.2%} of none")


if __name__ == "__main__":
    data_path = pathlib.Path(__file__).parent.parent.parent / "results" / "default"
    folder_names = sorted(
        [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    )

    # Get the last two folder names
    latest_two = folder_names[-2:]
    # Create full paths
    latest_paths = [os.path.join(data_path, f) for f in latest_two]

    self_suff_folder = latest_paths[0]
    none_folder = latest_paths[1]
    eval, flex_bus = evaluate_flex(self_suff_folder, none_folder)
    print("Buses with flexibility:")
    for bus in flex_bus:
        print(bus)
    for key, value in eval.items():
        if not value["self suff superior"] and not value["indefinite"]:
            print(f"{key} performes worse in self sufficiency than none")
            print(f"{key}: {value}")
        if key in flex_bus:
            print(f"{key}: {value}")
    # pass
