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
AGENT_KPIS: dict[str, dict[str, AGENT_KPI]] = {}


def network_kpi(name: str):
    def wrap(fn: NETWORK_KPI):
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
    return float(sim_result.ts_grid.sum(axis=1))


@network_kpi("max_load")
def max_load(sim_result: SimulationResult) -> float:
    """ ToDo: Rename peak_load """
    return sim_result.ts_grid.clip(lower=0).max().max()


@network_kpi("dt_h")
def dt_h(sim_result: SimulationResult) -> float:
    warnings.warn("Input parameters should never be a KPI.")
    return sim_result.run_pars.dt_h


@network_kpi("max_feed")
def max_feed(sim_result: SimulationResult) -> float:
    """ ToDo: Rename feed_in_peak """
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
    """ Rename: signal_sum"""
    fee_cols = [col for col in sim_result.assigned_grid_fees if "fee" in col]
    try:
        return sim_result.assigned_grid_fees[fee_cols].sum().item()
    except ValueError:
        # Does this Exception occur?
        print("No Signals were found. ")
        return 0.0


@network_kpi("congested_load")
def congested_load(sim_result: SimulationResult) -> float:
    """ Rename: load_(based)_congestion_time"""
    return (trafo_load(sim_result) > sim_result.grid_pars.p_lim).sum()


@network_kpi("congested_pv")
def congested_pv(sim_result: SimulationResult) -> float:
    """ Rename: pv_(based)_congestion_time"""
    return (trafo_load(sim_result) < sim_result.grid_pars.p_lim).sum()


@network_kpi("congested_times")
def congested_times(sim_result: SimulationResult) -> float:
    """ Rename: congestion_time"""
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
def total_demand_kwh(df: pd.DataFrame, dt_h: float) -> float:
    """Total grid demand in kWh."""
    return float(df["grid"].clip(lower=0).sum() * dt_h)


@agent_kpi("max_grid_demand")
def max_demand_kw(df: pd.DataFrame, dt_h: float) -> float:
    """Maximum grid demand in kW."""
    return df["grid"].clip(lower=0).max()


@agent_kpi("costs")
def capacity_costs(df: pd.DataFrame, dt_h: float) -> float:
    """Total cost based on grid demand and supply cost (c_sup)."""
    return (df["grid"].clip(lower=0) * df["c_sup"]).sum() * dt_h


@agent_kpi("battery_energy")
def battery_energy(df: pd.DataFrame, dt_h: float) -> float:
    """Total charged battery energy (kWh)."""
    if "bat_p_net" in df.columns:
        return df["bat_p_net"].clip(lower=0).sum() * dt_h
    return 0.0


@agent_kpi("battery_energy_from_grid")
def bss_kwh_from_grid(df: pd.DataFrame, dt_h: float) -> float:
    """Total charged energy to BSS while household net load is positive."""
    warnings.warn("battery_energy_from_grid is only heuristical.")
    if "bat_p_ac" in df.columns:
        kw_from_grid = df.loc[df["grid"] > 0, "bat_p_ac"].clip(lower=0).sum()
        return kw_from_grid * dt_h
    return 0.0


@agent_kpi("battery_max_from_grid")
def bss_max_kw_from_grid(df: pd.DataFrame, dt_h: float) -> float:
    """Maximal BSS charge load while household net load is positive."""
    warnings.warn("battery_max_from_grid is only heuristical.")
    if "bat_p_ac" in df.columns:
        kw_from_grid = df.loc[df["grid"] > 0, "bat_p_ac"].clip(lower=0)
        return kw_from_grid.max()
    return float("nan")


@agent_kpi("battery_energy_to_grid")
def battery_energy_to_grid(df: pd.DataFrame, dt_h: float) -> float:
    """Total discharged energy of BSS while household net load is negative."""
    warnings.warn("battery_energy_to_grid is only heuristical.")
    if "bat_p_ac" in df.columns:
        kw_to_grid = (-df["bat_p_ac"])[df["grid"] < 0].clip(lower=0).sum()
        return kw_to_grid * dt_h
    return 0.0


@agent_kpi("battery_max_to_grid")
def battery_max_to_grid(df: pd.DataFrame, dt_h: float) -> float:
    """Maximal BSS discharge load while household net load is negative."""
    warnings.warn("battery_max_to_grid is only heuristical.")
    if "bat_p_ac" in df.columns:
        kw_to_grid = (-df["bat_p_ac"])[df["grid"] < 0].clip(lower=0)
        return kw_to_grid.max()
    return float("nan")


@agent_kpi("overcharge_bat")
def overcharge_bat(df: pd.DataFrame, dt_h: float) -> float:
    """Count of time steps where battery SOC exceeds 1.0."""
    if "bat_soc" in df.columns:
        return float((df["bat_soc"] > 1.0).sum())
    return 0.0


@agent_kpi("undercharge_bat")
def undercharge_bat(df: pd.DataFrame, dt_h: float) -> float:
    """Count of time steps where battery SOC drops below 0.0."""
    if "bat_soc" in df.columns:
        return float((df["bat_soc"] < 0.0).sum())
    return 0.0


@agent_kpi("charging_cycle_equivalents")
def equivalent_full_charge_cycles(df: pd.DataFrame, dt_h: float) -> float:
    """Approximate equivalent full charge cycles based on SOC increases."""
    if "bat_soc" in df.columns and len(df["bat_soc"]) >= 2:
        diff = np.diff(df["bat_soc"])
        # Sum of all positive SOC changes represents charged fraction (in full cycles)
        return float(diff[diff > 0].sum())
    return float("nan")


@agent_kpi("mean_temp")
def mean_temperature(df: pd.DataFrame, dt_h: float) -> float:
    """Average indoor temperature of the building."""
    if "hp_temp" in df.columns:
        return float(df["hp_temp"].mean())
    return float("nan")


@agent_kpi("hp_energy_el")
def hp_energy_el(df: pd.DataFrame, dt_h: float) -> float:
    """Total electrical energy consumed by the heat pump (kWh)."""
    if "hp_p_in" in df.columns:
        return float(df["hp_p_in"].sum() * dt_h)
    return 0.0


@agent_kpi("hp_p_max")
def hp_p_max(df: pd.DataFrame, dt_h: float) -> float:
    """Maximum electrical input power of the heat pump (kW)."""
    if "hp_p_in" in df.columns:
        return float(df["hp_p_in"].max())
    return float("nan")


@agent_kpi("overheating")
def overheating(df: pd.DataFrame, dt_h: float) -> float:
    """Number of timesteps where the heat storage temperature exceeded 75°C."""
    if "hp_temp" in df.columns:
        return float((df["hp_temp"] > 75).sum())
    return 0.0


@agent_kpi("underheating")
def underheating(df: pd.DataFrame, dt_h: float) -> float:
    """Number of timesteps where heat storage temperature was below 60°C."""
    if "hp_temp" in df.columns:
        return float((df["hp_temp"] < 60).sum())
    return 0.0


@agent_kpi("above_t")
def above_t(df: pd.DataFrame, dt_h: float) -> float:
    """Alias for overheating."""
    return overheating(df, dt_h)


@agent_kpi("under_t")
def under_t(df: pd.DataFrame, dt_h: float) -> float:
    """Alias for underheating."""
    return underheating(df, dt_h)


@agent_kpi("ev_energy")
def ev_energy(df: pd.DataFrame, dt_h: float) -> float:
    """Total electrical energy charged into the EV (from all sources)."""
    if "ev_p_net" in df.columns:
        return float(df["ev_p_net"].clip(lower=0).sum() * dt_h)
    return 0.0


@agent_kpi("ev_energy_from_grid")
def ev_energy_from_grid(df: pd.DataFrame, dt_h: float) -> float:
    """Energy charged into the EV that originated from the grid."""
    if "ev_p_ac" in df.columns:
        mask = (df["ev_p_net"] > 0) & (df["grid"] > 0)
        return float(df.loc[mask, "ev_p_ac"].clip(lower=0).sum() * dt_h)
    return 0.0


@agent_kpi("ev_max_from_grid")
def ev_max_from_grid(df: pd.DataFrame, dt_h: float) -> float:
    """Maximum EV charging power drawn from the grid."""
    if "ev_p_ac" in df.columns:
        mask = (df["ev_p_net"] > 0) & (df["grid"] > 0)
        return float(df.loc[mask, "ev_p_ac"].clip(lower=0).max())
    return float("nan")


@agent_kpi("ev_energy_to_grid")
def ev_energy_to_grid(df: pd.DataFrame, dt_h: float) -> float:
    """Energy discharged from the EV back to the grid (V2G)."""
    if "ev_p_ac" in df.columns:
        mask = (df["ev_p_net"] < 0) & (df["grid"] < 0)
        return float((-df.loc[mask, "ev_p_ac"]).clip(lower=0).sum() * dt_h)
    return 0.0


@agent_kpi("overcharge_ev")
def overcharge_ev(df: pd.DataFrame, dt_h: float) -> float:
    """Number of timesteps where the EV SoC exceeded 100%."""
    if "ev_soc" in df.columns:
        return float((df["ev_soc"] > 1.0).sum())
    return 0.0


@agent_kpi("undercharge_ev")
def undercharge_ev(df: pd.DataFrame, dt_h: float) -> float:
    """Number of timesteps where the EV SoC dropped below 0%."""
    if "ev_soc" in df.columns:
        return float((df["ev_soc"] < 0.0).sum())
    return 0.0


@agent_kpi("total_import")
def total_import(df: pd.DataFrame, dt_h: float) -> float:
    """Total electrical energy imported from the grid [kWh]."""
    return float(df["grid"].clip(lower=0).sum() * dt_h)


@agent_kpi("max_import")
def max_import(df: pd.DataFrame, dt_h: float) -> float:
    """Maximum grid import power [kW]."""
    return float(df["grid"].clip(lower=0).max())


@agent_kpi("total_feed")
def total_feed(df: pd.DataFrame, dt_h: float) -> float:
    """Total energy exported to the grid [kWh]."""
    return float((-df["grid"]).clip(lower=0).sum() * dt_h)


@agent_kpi("max_feed")
def max_feed(df: pd.DataFrame, dt_h: float) -> float:
    """Maximum export power magnitude [kW]."""
    return float((-df["grid"]).clip(lower=0).max())


@agent_kpi("energy_consumption")
def energy_consumption(df: pd.DataFrame, dt_h: float) -> float:
    """Total energy consumed by all systems and electric load [kWh]."""
    base_load = df["p_el_load"].clip(lower=0).sum() if "p_el_load" in df.columns else 0.0
    hp = df["hp_p_in"].clip(lower=0).sum() if "hp_p_in" in df.columns else 0.0
    ev = df["ev_p_net"].clip(lower=0).sum() if "ev_p_net" in df.columns else 0.0
    return float((base_load + hp + ev) * dt_h)


@agent_kpi("self_consumption")
def self_consumption(df: pd.DataFrame, dt_h: float) -> float:
    """PV generation used on-site [kWh]."""
    if "p_el_pv" not in df.columns:
        return 0.0
    pv_gen = float(df.loc[df["p_el_pv"] > 0, "p_el_pv"].sum() * dt_h)
    export = float((-df.loc[df["grid"] < 0, "grid"]).sum() * dt_h)
    return max(0.0, pv_gen - export)


@agent_kpi("self_sufficiency")
def self_sufficiency(df: pd.DataFrame, dt_h: float) -> float:
    """Ratio of total demand covered by local PV generation."""
    if energy_consumption(df, dt_h) <= 0:
        return 0.0
    return self_consumption(df, dt_h) / energy_consumption(df, dt_h)


@agent_kpi("costs")
def costs(df: pd.DataFrame, dt_h: float) -> float:
    """Total cost of electricity purchased from the grid ."""
    if "c_sup" in df.columns:
        values = (df["grid"] * df["c_sup"]).where(df["grid"] >= 0, 0.0)
        return float(values.sum() * dt_h)
    return 0.0


@agent_kpi("revenue")
def revenue(df: pd.DataFrame, dt_h: float) -> float:
    """Revenue from energy exported to the grid ."""
    if {"grid", "c_feed"}.issubset(df.columns):
        # only negative grid values (exports)
        revenues = (-df["grid"]).clip(lower=0) * df["c_feed"]
        return float(revenues.sum() * dt_h)
    return 0.0


@agent_kpi("profit")
def profit(df: pd.DataFrame, dt_h: float) -> float:
    """Net profit from energy trading: revenue minus costs ."""
    return revenue(df, dt_h) - costs(df, dt_h)


@agent_kpi("net_grid_energy")
def net_grid_energy(df: pd.DataFrame, dt_h: float) -> float:
    """Net energy balance with the grid: imports minus exports [kWh]."""
    return total_import(df, dt_h) - total_feed(df, dt_h)


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

