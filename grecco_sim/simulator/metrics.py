from typing import Any, Dict, Mapping, Tuple, Union
from pathlib import Path
from typing import Callable
import numpy as np

import pandas as pd

import warnings

from grecco_sim.simulator.result import SimulationResult
from grecco_sim.util import type_defs


# A network KPI is applied to Simulation result.
NETWORK_KPI = Callable[[SimulationResult], Any]
NETWORK_KPIS: dict[str, NETWORK_KPI] = {}

# An agent KPI is applied to the dataframe representing the agents results.
AGENT_KPI = Callable[[pd.DataFrame], Any]
AGENT_KPIS: dict[str, dict[str, KPI]] = {}


def network_kpi(name: str):
    def wrap(fn: KPI):
        NETWORK_KPIS[name] = fn
        return fn
    return wrap


def agent_kpi(name: str):
    def wrap(fn: AGENT_KPI):
        AGENT_KPIS[name] = fn
        return fn
    return wrap


@network_kpi("trafo_load")
def trafo_load(sim_result: SimulationResult) -> pd.DataFrame:
    """ Sum up along agent axis to retrieve transformer load for each step. """
    return sim_result.ts_grid.sum(axis=1)


@network_kpi("max_load")
def max_load(sim_result: SimulationResult) -> float:
    return sim_result.ts_grid.clip(lower=0).max().max()


@network_kpi("dt_h")
def dt_h(sim_result: SimulationResult) -> float:
    warnings.warn("Input parameters should never be a KPI.")
    return sim_result.run_pars.dt_h


@network_kpi("max_feed")
def max_feed(sim_result: SimulationResult) -> float:
    return -(trafo_load(sim_result).clip(upper=0).min())


@network_kpi("MV_demand")
def mv_demand(sim_result: SimulationResult) -> float:
    """ The total demand of power withdrawn from the MV grid. """
    return trafo_load(sim_result).clip(0).sum() * sim_result.run_pars.dt_h


@network_kpi("MV_feed")
def mv_feed(sim_result: SimulationResult) -> float:
    """ The total power fed-in to the MV grid. """
    return (-trafo_load(sim_result)).clip(0).sum() * sim_result.run_pars.dt_h


@network_kpi("agg_signal")
def total_signal_costs(sim_result: SimulationResult) -> float:
    fee_cols = [col for col in sim_result.assigned_grid_fees if "fee" in col]
    try:
        return sim_result.assigned_grid_fees[fee_cols].sum().item()
    except ValueError:
        # Does this Exception occur?
        print("No Signals were found. ")
        return 0.0


@network_kpi("congested_load")
def congested_load(sim_result: SimulationResult) -> float:
    return (trafo_load(sim_result) > sim_result.grid_pars.p_lim).sum()


@network_kpi("congested_pv")
def congested_pv(sim_result: SimulationResult) -> float:
    return (trafo_load(sim_result) < sim_result.grid_pars.p_lim).sum()


@network_kpi("congested_times")
def congested_times(sim_result: SimulationResult) -> float:
    return congested_pv(sim_result) + congested_load(sim_result)


# @network_kpi("costs_all")
# def costs_all(sim_result: SimulationResult) -> float:
#     grid =
#
#     supp = grid.clip(lower=0)
#     feed = grid.clip(upper=0)
#
#     costs_supp = (
#             float((supp.to_numpy() * df["c_sup"].to_numpy())[
#                       supp.to_numpy() > 0].sum()) * dt_h
#     )
#     costs_feed = (
#             float((feed.to_numpy() * df["c_feed"].to_numpy())[
#                       feed.to_numpy() < 0].sum()) * dt_h
#     )
#     costs_all += costs_supp + costs_feed
#
#


@agent_kpi("grid_demand")
def total_demand_kwh(sim_result: SimulationResult) -> dict[str, float]:
    total_demand_kwh = dict()
    for sys_id, df in sim_result.agents_ts.items():
        total_demand_kwh[sys_id] = df["grid"].clip(lower=0).sum()
    return total_demand_kwh


@agent_kpi("max_grid_demand")
def max_demand_kw(sim_result: SimulationResult) -> dict[str, float]:
    max_demand_kw = dict()
    for sys_id, df in sim_result.agents_ts.items():
        max_demand_kw[sys_id] = df["grid"].clip(lower=0).max()
    return max_demand_kw


@agent_kpi("costs")
def capacity_costs(sim_result: SimulationResult) -> dict[str, float]:
    capacity_costs = dict()
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        demand_capacity = df["grid"].clip(lower=0)
        c_sup = df["c_sup"]
        capacity_costs[sys_id] = (demand_capacity * c_sup).sum() * dt_h
    return capacity_costs


@agent_kpi("battery_energy")
def battery_energy(sim_result: SimulationResult) -> dict[str, float]:
    battery_energy = dict()
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        bat_p_net = df["bat_p_net"] if "bat_p_net" in df.columns else 0.0
        battery_energy[sys_id] = bat_p_net.clip(0).sum() * dt_h
    return battery_energy


@agent_kpi("battery_energy_from_grid")
def bss_kwh_from_grid(sim_result: SimulationResult) -> dict[str, float]:
    """ Total charged energy to BSS while household net load is postive. """
    warnings.warn("The math of battery_energy_from_grid does not seem right.")
    bss_kwh_from_grid = dict()
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "bat_p_ac" in df.columns:
            kw_from_grid = df["bat_p_ac"][df["grid"] > 0].clip(0).sum()
            bss_kwh_from_grid[sys_id] = kw_from_grid * dt_h
        else:
            bss_kwh_from_grid[sys_id] = 0.0

    return bss_kwh_from_grid


@agent_kpi("battery_max_from_grid")
def bss_max_kw_from_grid(sim_result: SimulationResult) -> dict[str, float]:
    """ Maxmimal BSS charge load while household net load is postive. """
    warnings.warn("The math of battery_max_from_grid does not seem right.")
    bss_max_kw_from_grid = dict()
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "bat_p_ac" in df.columns:
            kw_from_grid = df["bat_p_ac"][df["grid"] > 0].clip(0)
            bss_max_kw_from_grid[sys_id] = kw_from_grid.max()
        else:
            bss_max_kw_from_grid[sys_id] = float("nan")
    return bss_max_kw_from_grid


@agent_kpi("battery_energy_to_grid")
def battery_energy_to_grid(sim_result: SimulationResult) -> dict[str, float]:
    """ Total discharged energy of BSS while household net load is negative."""
    warnings.warn("The math of battery_energy_to_grid does not seem right.")
    battery_energy_to_grid = dict()
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "bat_p_ac" in df.columns:
            kw_to_grid = (-df["bat_p_ac"])[df["grid"] < 0].clip(0).sum()
            battery_energy_to_grid[sys_id] = kw_to_grid * dt_h
        else:
            battery_energy_to_grid[sys_id] = 0.0
    return battery_energy_to_grid


@agent_kpi("battery_max_to_grid")
def battery_max_to_grid(sim_result: SimulationResult) -> dict[str, float]:
    """ Maxmimal BSS discharge load while household net load is negative. """
    warnings.warn("The math of battery_max_to_grid does not seem right.")
    bss_max_kw_to_grid = dict()
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "bat_p_ac" in df.columns:
            kw_to_grid = (-df["bat_p_ac"])[df["grid"] > 0].clip(0)
            bss_max_kw_to_grid[sys_id] = kw_to_grid.max()
        else:
            bss_max_kw_to_grid[sys_id] = float("nan")
    return bss_max_kw_to_grid

@agent_kpi("overcharge_bat")
def overcharge_bat(sim_result: SimulationResult) -> dict[str, float]:
    overcharge_bat = dict()
    for sys_id, df in sim_result.agents_ts.items():
        if "bat_soc" in df.columns:
            overcharge_bat[sys_id] = (df["bat_soc"] > 1.0).sum()
        else:
            overcharge_bat[sys_id] = 0.0
    return overcharge_bat


@agent_kpi("undercharge_bat")
def undercharge_bat(sim_result: SimulationResult) -> dict[str, float]:
    undercharge_bat = dict()
    for sys_id, df in sim_result.agents_ts.items():
        if "bat_soc" in df.columns:
            undercharge_bat[sys_id] = float((df["bat_soc"] < 0.0).sum())
        else:
            undercharge_bat[sys_id] = 0.0
    return undercharge_bat


@agent_kpi("charging_cycle_equivalents")
def equivalent_full_charge_cycles(sim_result: SimulationResult) -> dict[str, float]:
    """ Here some explanations would be really nice. """
    full_charge_cycles = dict()
    for sys_id, df in sim_result.agents_ts.items():
        if "bat_soc" in df.columns and df["bat_soc"].size >= 2:
            diff = np.diff(df["bat_soc"])
            full_charge_cycles[sys_id] = float(diff[diff > 0].sum())
        else:
            full_charge_cycles[sys_id] = float("nan")
    return full_charge_cycles


@agent_kpi("mean_temp")
def mean_temperature(sim_result: SimulationResult) -> dict[str, float]:
    """Average indoor temperature of the building."""
    mean_temperature = {}
    for sys_id, df in sim_result.agents_ts.items():
        if "hp_temp" in df.columns:
            mean_temperature[sys_id] = float(df["hp_temp"].mean())
        else:
            mean_temperature[sys_id] = float("nan")
    return mean_temperature


@agent_kpi("hp_energy_el")
def hp_energy_el(sim_result: SimulationResult) -> dict[str, float]:
    """Total electrical energy consumed by the heat pump."""
    hp_energy_el = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "hp_p_in" in df.columns:
            hp_energy_el[sys_id] = df["hp_p_in"].sum() * dt_h
        else:
            hp_energy_el[sys_id] = 0.0
    return hp_energy_el


@agent_kpi("hp_p_max")
def hp_p_max(sim_result: SimulationResult) -> dict[str, float]:
    """Maximum electrical input power of the heat pump."""
    hp_p_max = {}
    for sys_id, df in sim_result.agents_ts.items():
        if "hp_p_in" in df.columns > 0:
            hp_p_max[sys_id] = df["hp_p_in"].max()
        else:
            hp_p_max[sys_id] = float("nan")
    return hp_p_max


@agent_kpi("overheating")
def overheating(sim_result: SimulationResult) -> dict[str, float]:
    """Number of timesteps where the heat pump temperature exceeded 23°C."""
    overheating = {}
    for sys_id, df in sim_result.agents_ts.items():
        if "hp_temp" in df.columns:
            overheating[sys_id] = float((df["hp_temp"] > 23).sum())
        else:
            overheating[sys_id] = 0.0
    return overheating


@agent_kpi("underheating")
def underheating(sim_result: SimulationResult) -> dict[str, float]:
    """Number of timesteps where the heat pump temperature was below 18°C."""
    underheating = {}
    for sys_id, df in sim_result.agents_ts.items():
        if "hp_temp" in df.columns:
            underheating[sys_id] = float((df["hp_temp"] < 18).sum())
        else:
            underheating[sys_id] = 0.0
    return underheating


@agent_kpi("above_t")
def above_t(sim_result: SimulationResult) -> dict[str, float]:
    """Alias metric for overheating (timesteps above temperature threshold)."""
    return overheating(sim_result)


@agent_kpi("under_t")
def under_t(sim_result: SimulationResult) -> dict[str, float]:
    """Alias metric for underheating (timesteps below temperature threshold)."""
    return underheating(sim_result)


@agent_kpi("ev_energy")
def ev_energy(sim_result: SimulationResult) -> dict[str, float]:
    """Total electrical energy charged into the EV (from all sources)."""
    ev_energy = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "ev_p_net" in df.columns:
            ev_energy[sys_id] = float(df["ev_p_net"].clip(lower=0).sum() * dt_h)
        else:
            ev_energy[sys_id] = 0.0
    return ev_energy


@agent_kpi("ev_energy_from_grid")
def ev_energy_from_grid(sim_result: SimulationResult) -> dict[str, float]:
    """Energy charged into the EV that originated from the grid."""
    ev_energy_from_grid = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "ev_p_ac" in df.columns:
            from_grid = (df["ev_p_net"] > 0) & (df["grid"] > 0)
            df[from_grid, "ev_p_ac"].sum() * dt_h
            ev_energy_from_grid[sys_id] = df[from_grid, "ev_p_ac"].sum() * dt_h
        else:
            ev_energy_from_grid[sys_id] = 0.0
    return ev_energy_from_grid


@agent_kpi("ev_max_from_grid")
def ev_max_from_grid(sim_result: SimulationResult) -> dict[str, float]:
    """Maximum EV charging power drawn from the grid."""
    ev_max_from_grid = {}
    for sys_id, df in sim_result.agents_ts.items():
        if "ev_p_ac" in df.columns:
            from_grid = (df["ev_p_net"] > 0) & (df["grid"] > 0)
            if from_grid.any():
                ev_max_from_grid[sys_id] = df[from_grid, "ev_p_ac"].max()
            else:
                ev_max_from_grid[sys_id] = float("nan")
        else:
            ev_max_from_grid[sys_id] = float("nan")

    return ev_max_from_grid


@agent_kpi("ev_energy_to_grid")
def ev_energy_to_grid(sim_result: SimulationResult) -> dict[str, float]:
    """Energy discharged from the EV back to the grid (V2G)."""
    ev_energy_to_grid = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "ev_p_ac" in df.columns:
            from_grid = (-df["ev_p_net"] > 0) & (-df["grid"] > 0)
            df[from_grid, "ev_p_ac"].sum() * dt_h
            ev_energy_to_grid[sys_id] = -df[from_grid, "ev_p_ac"].sum() * dt_h
        else:
            ev_energy_to_grid[sys_id] = 0.0
    return ev_energy_to_grid


@agent_kpi("overcharge_ev")
def overcharge_ev(sim_result: SimulationResult) -> dict[str, float]:
    """Number of timesteps where the EV SoC exceeded 100%."""
    overcharge_ev = {}
    for sys_id, df in sim_result.agents_ts.items():
        if "ev_soc" in df.columns:
            overcharge_ev[sys_id] = float((df["ev_soc"] > 1.0).sum())
        else:
            overcharge_ev[sys_id] = 0.0
    return overcharge_ev


@agent_kpi("undercharge_ev")
def undercharge_ev(sim_result: SimulationResult) -> dict[str, float]:
    """Number of timesteps where the EV SoC dropped below 0%."""
    undercharge_ev = {}
    for sys_id, df in sim_result.agents_ts.items():
        if "ev_soc" in df.columns:
            undercharge_ev[sys_id] = float((df["ev_soc"] < 0.0).sum())
        else:
            undercharge_ev[sys_id] = 0.0
    return undercharge_ev


@agent_kpi("total_import")
def total_import(sim_result: SimulationResult) -> dict[str, float]:
    """Total electrical energy imported from the grid [kWh]."""
    total_import = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        total_import[sys_id] = df["grid"].clip(0).sum() * dt_h
    return total_import


@agent_kpi("max_import")
def max_import(sim_result: SimulationResult) -> dict[str, float]:
    """Maximum grid import power [kW]."""
    max_import = {}
    for sys_id, df in sim_result.agents_ts.items():
        max_import[sys_id] = df["grid"][df["grid"] > 0].max()

    return max_import


@agent_kpi("total_feed")
def total_feed(sim_result: SimulationResult) -> dict[str, float]:
    """Total energy exported to the grid [kWh]."""
    total_feed = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        total_feed[sys_id] = (-df["grid"]).clip(0).sum() * dt_h
    return total_feed


@agent_kpi("max_feed")
def max_feed(sim_result: SimulationResult) -> dict[str, float]:
    """ Maximum export power magnitude [kW]."""
    max_feed = {}
    for sys_id, df in sim_result.agents_ts.items():
        max_feed[sys_id] = (-df["grid"])[df["grid"] < 0].max()

    return max_feed


@agent_kpi("energy_consumption")
def energy_consumption(sim_result: SimulationResult) -> dict[str, float]:
    """Total energy consumed by all systems and electric load [kWh]."""
    energy_consumption = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        base_load = df["p_el_load"].sum() if "p_el_load" in df.columns else 0.0
        hp = df["hp_p_in"].clip(0).sum() if "hp_p_in" in df.columns else 0.0
        ev = df["ev_p_net"].clip(0).sum() if "ev_p_net" in df.columns else 0.0
        energy_consumption[sys_id] = base_load + hp + ev * dt_h
    return energy_consumption


@agent_kpi("self_consumption")
def self_consumption(sim_result: SimulationResult) -> dict[str, float]:
    """PV generation used on-site [kWh]."""
    self_consumption = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "p_el_pv" in df.columns:
            pv_gen = float(df.loc[df["p_el_pv"] > 0, "p_el_pv"].sum() * dt_h)
            export = float((-df.loc[df["grid"] < 0, "grid"]).sum() * dt_h)
            self_consumption[sys_id] = max(0.0, pv_gen - export)
        else:
            self_consumption[sys_id] = 0.0
    return self_consumption


@agent_kpi("self_sufficiency")
def self_sufficiency(sim_result: SimulationResult) -> dict[str, float]:
    """ Ratio of total demand covered by local PV generation."""
    self_sufficiency = {}
    # Here only inflexible load was considered - why?
    consumption = energy_consumption(sim_result)
    for sys_id, df in sim_result.agents_ts.items():
        self_sufficiency[sys_id] = self_consumption[sys_id] / consumption[sys_id]
    return self_sufficiency


@agent_kpi("costs")
def costs(sim_result: SimulationResult) -> dict[str, float]:
    """Total cost of electricity purchased from the grid."""
    results = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if {"grid", "c_sup"}.issubset(df.columns):
            values = (df["grid"] * df["c_sup"]).where(df["grid"] >= 0, 0.0)
            results[sys_id] = float(values.sum() * dt_h)
        else:
            results[sys_id] = 0.0
    return results


@agent_kpi("revenue")
def revenue(sim_result: SimulationResult) -> dict[str, float]:
    """ Revenue from energy exported to the grid."""
    revenue = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "c_feed" in df.columns:
            revenues = (-df["grid"]).clip(0) * df["c_feed"]
            revenue[sys_id] = revenues.sum() * dt_h
        else:
            revenue[sys_id] = 0.0
    return revenue


@agent_kpi("profit")
def profit(sim_result: SimulationResult) -> dict[str, float]:
    """Net profit from energy trading: revenue minus costs [€]."""
    profit = {}
    dt_h = sim_result.dt_h
    for sys_id, df in sim_result.agents_ts.items():
        if "c_feed" in df.columns:
            profit[sys_id] = revenue[sys_id] - costs[sys_id]
        else:
            profit[sys_id] = 0.0
    return profit


@agent_kpi("net_grid_energy")
def net_grid_energy(sim_result: SimulationResult) -> dict[str, float]:
    """Net energy balance with the grid: imports minus exports [kWh]."""
    net_grid_energy = {}
    for sys_id, df in sim_result.agents_ts.items():
        net_grid_energy[sys_id] = total_import[sys_id] - total_feed[sys_id]
    return net_grid_energy


AGENT_KPI_FIELDS = {
    "grid_demand",
    "energy_consumption",
    "max_grid_demand",
    "total_feed",
    "max_feed",
    "self-consumption",
    "self-sufficiency",
    "charging_cycle_equivalents",
    "battery_energy",
    "battery_energy_from_grid",
    "battery_max_from_grid",
    "battery_energy_to_grid",
    "battery_max_to_grid",
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
    "above_t",
    "under_t",
    "ev_energy",
    "ev_energy_from_grid",
    "ev_max_from_grid",
    "ev_energy_to_grid",
    "overcharge_ev",
    "undercharge_ev",
}


def _write_to_files(
    run_pars: type_defs.RunParameters,
    eval_res: Union[Mapping[str, Any], pd.Series, pd.DataFrame],
    file_name: str = "kpis",
) -> None:
    """Persist evaluation results to CSV in `run_pars.output_file_dir`.

    - If `eval_res` is a DataFrame, append/union columns and write.
    - If `eval_res` is a Series, convert to one-row DataFrame.
    - If it's a mapping, convert to one-row DataFrame (index = tag if present).
    - If it's a dict of Series, we use `DataFrame.from_dict(..., orient='index')`.
    """
    out_dir: Path = run_pars.output_file_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build DataFrame to append
    if isinstance(eval_res, pd.DataFrame):
        eval_df = eval_res
    elif isinstance(eval_res, pd.Series):
        eval_df = eval_res.to_frame().T
    elif isinstance(eval_res, Mapping) and all(
        not isinstance(v, pd.Series) for v in eval_res.values()
    ):
        eval_df = pd.DataFrame([dict(eval_res)])
    else:
        eval_df = pd.DataFrame.from_dict(eval_res, orient="index")

    eval_file_path = out_dir / f"{file_name}.csv"
    if eval_file_path.exists():
        try:
            existing = pd.read_csv(eval_file_path, index_col=0)
            combined = pd.concat([existing, eval_df], axis=0, sort=False)
            eval_df = combined
        except Exception:
            pass

    eval_df.to_csv(eval_file_path)


def evaluate_kpis(sim_result: SimulationResult) -> (
        Tuple[pd.Series, Dict[str, Dict[str, float]]]):

    network_kpis = {}
    # Add sim_tag as id.
    network_kpis["sim_tag"] = sim_result.tag

    for kpi_name in list(NETWORK_KPIS.keys()):
        try:
            network_kpis[kpi_name] = NETWORK_KPIS[kpi_name](sim_result)
        except Exception as e:
            network_kpis[kpi_name] = f"ERROR: {e}"

    # Agent KPIs: Dict[<KPI_NAME>, Dict[<SYS_ID>, <KPI_VALUE>]
    agent_kpis = {}
    dt_h = sim_result.dt_h

    for kpi_name in list(AGENT_KPIS.keys()):
        agent_kpis[kpi_name] = dict()

        for sys_id, df in sim_result.agents_ts.items():
            try:
                agent_kpis[kpi_name][sys_id] = AGENT_KPIS[kpi_name](df, dt_h)
            except Exception as e:
                agent_kpis[kpi_name][sys_id] = f"ERROR: {e}"

    return pd.Series(network_kpis), agent_kpis


# def agent_analysis(
#     agent_ts: Mapping[str, pd.DataFrame], dt_h: float
# ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
#     """Compute KPIs for each agent.
#
#     Definitions / Notes
#     -------------------
#     - All energy metrics integrate power with `dt_h` to kWh.
#     - `total_feed` is export energy to grid (kWh, positive magnitude).
#     - `max_feed` is the maximum export power magnitude (kW, positive value).
#     - EV energy is integrated from `ev_p_net` (kW), not from SoC.
#     - Self-consumption is PV generation used on-site: PV_gen - feed_to_grid, bounded below by 0.
#     - Profit = revenue - costs, where revenue is positive cash inflow from exports.
#     """
#     energy_map = {"hp": "hp_energy_el", "ev": "ev_energy"}
#     agents_res = pd.DataFrame(
#         index=list(agent_ts.keys()), columns=sorted(AGENT_KPI_FIELDS), dtype=float
#     )
#
#     # Aggregate statistics
#     numeric_agents = agents_res.astype(float)
#     agent_stats: Dict[str, pd.Series] = {
#         "mean": numeric_agents.mean(numeric_only=True),
#         "median": numeric_agents.median(numeric_only=True),
#         "25_quantile": numeric_agents.quantile(0.25, numeric_only=True),
#         "75_quantile": numeric_agents.quantile(0.75, numeric_only=True),
#         "max": numeric_agents.max(numeric_only=True),
#         "min": numeric_agents.min(numeric_only=True),
#     }
#
#     return agents_res, agent_stats

