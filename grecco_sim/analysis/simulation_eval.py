import pathlib
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
import pandas as pd

from grecco_sim.sim_models import battery
from grecco_sim.simulator import results
from grecco_sim.util import sig_types, type_defs, style


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _masked_sum(values: np.ndarray, mask: np.ndarray, dt_h: float | None = None) -> float:
    """Sum of `values` over `mask`. If `dt_h` is provided, multiply by it.
    Returns 0.0 if mask selects no items.
    """
    if values.size == 0:
        return 0.0
    if mask.size == 0:
        return 0.0
    sel = values[mask]
    if sel.size == 0:
        return 0.0
    total = float(sel.sum())
    return total * dt_h if dt_h is not None else total


def _safe_max(values: np.ndarray, default=np.nan) -> float:
    return float(values.max()) if values.size > 0 else float(default)


def _safe_min(values: np.ndarray, default=np.nan) -> float:
    return float(values.min()) if values.size > 0 else float(default)


# ---------------------------------------------------------------------------
# Aggregated profile KPIs
# ---------------------------------------------------------------------------


def _get_aggregated_ts_result(sim_results: results.SimulationResult) -> pd.Series:
    """KPIs computed from the aggregated grid power profile.

    Returns a **pd.Series** with the documented KPI fields as index.
    """
    DOCUMENTED_KPIS = {
        "dt_h",
        "max_load",  # max positive demand (kW)
        "max_feed",  # max export magnitude (kW, positive value)
        "MV_demand",  # total consumption (kWh)
        "MV_feed",  # total export to upper grid (kWh)
        "agg_signal",
        "congested_times",
        "congested_pv",
        "congested_load",
    }

    # Sum agents across columns -> aggregated net power at PCC (kW)
    res_load: pd.Series = sim_results.ts_grid.sum(axis=1)
    dt_h = sim_results.run_pars.dt.seconds / 3600

    data: Dict[str, Any] = {}
    data["dt_h"] = dt_h

    # Peak import/export magnitudes in kW
    data["max_load"] = max(0.0, float(res_load.max()))
    data["max_feed"] = max(0.0, float(-res_load.min()))

    # Energy integrals in kWh (positive import, negative export)
    pos = res_load > 0
    neg = res_load < 0
    data["MV_demand"] = float(res_load.loc[pos].sum()) * dt_h
    data["MV_feed"] = float(-res_load.loc[neg].sum()) * dt_h

    # Signals (sum all assigned grid fee columns for all systems)
    sys_ids = sim_results.sys_ids
    fee_cols = [
        f"fee_{sys_id}"
        for sys_id in sys_ids
        if f"fee_{sys_id}" in sim_results.assigned_grid_fees.columns
    ]
    if fee_cols:
        data["agg_signal"] = float(sim_results.assigned_grid_fees.loc[:, fee_cols].to_numpy().sum())
    else:
        data["agg_signal"] = 0.0

    # Congestion counts (power limit p_lim in kW)
    p_lim = float(sim_results.grid_pars.p_lim)
    data["congested_load"] = int((res_load > p_lim).sum())
    data["congested_pv"] = int((res_load < -p_lim).sum())
    data["congested_times"] = int(data["congested_load"] + data["congested_pv"])

    # Ensure exact key set
    assert set(data.keys()) == DOCUMENTED_KPIS, "Make sure that the return is correctly documented!"
    return pd.Series(data)


# ---------------------------------------------------------------------------
# Flex analysis across agents
# ---------------------------------------------------------------------------


def flex_analysis(
    agent_ts: Mapping[str, pd.DataFrame],
    sizing: Mapping[str, type_defs.SysParsPVBat],
    dt_h: float,
) -> pd.Series:
    """Aggregate per-agent quantities for quick flex overview.

    Returns a **pd.Series** so the whole evaluation can be composed into a
    single-row DataFrame easily.
    """
    costs_all = 0.0
    en_in_bat_ts: np.ndarray | None = None
    combined_cap = 0.0
    en_loss = 0.0
    agg_feed = 0.0
    agg_bat_to_grid = 0.0

    for sys_id, df in agent_ts.items():
        pv = df["p_el_pv"] if "p_el_pv" in df.columns else pd.Series(0.0, index=df.index)
        grid = df["grid"]

        supp = grid.clip(lower=0)
        feed = grid.clip(upper=0)

        costs_supp = (
            float((supp.to_numpy() * df["c_sup"].to_numpy())[supp.to_numpy() > 0].sum()) * dt_h
        )
        costs_feed = (
            float((feed.to_numpy() * df["c_feed"].to_numpy())[feed.to_numpy() < 0].sum()) * dt_h
        )
        costs_all += costs_supp + costs_feed

        # Battery export estimation (battery to grid = net export beyond PV)
        bat_to_grid = (grid + pv).to_numpy()
        bat_to_grid = bat_to_grid[bat_to_grid < 0]
        agg_bat_to_grid += float(bat_to_grid.sum()) * dt_h
        agg_feed += float(feed.sum()) * dt_h

        # Battery energy-in-timeseries accumulation
        if isinstance(sizing.get(sys_id), type_defs.SysParsPVBat) and "bat_soc" in df.columns:
            cap = float(sizing[sys_id].capacity)
            contrib = df["bat_soc"].to_numpy() * cap
            en_in_bat_ts = contrib if en_in_bat_ts is None else (en_in_bat_ts + contrib)
            combined_cap += cap
            en_loss += float((df["bat_p_ac"] - df["bat_p_net"]).sum()) * float(sizing[sys_id].dt_h)

    charged_energy = 0.0 if en_in_bat_ts is None else float(en_in_bat_ts[-1] - en_in_bat_ts[0])
    return pd.Series(
        {
            "costs_all": costs_all,
            "en_in_bat_ts": en_in_bat_ts,
            "charged_energy": charged_energy,
            "combined_capacity": combined_cap,
            "losses_bat": en_loss,
            "agg_feed": agg_feed,
            "agg_bat_to_grid": agg_bat_to_grid,
        }
    )


def signal_analysis(
    run_pars: type_defs.RunParameters, raw_output: Mapping[str, Dict[str, Any]]
) -> None:
    if not getattr(run_pars, "plot", False):
        return

    if run_pars.coordination_mechanism not in {"admm", "second_order"}:
        print(
            f"No analysis to be made for '{run_pars.coordination_mechanism}' coordination mechanism."
        )
        return

    fig_sig, ax_sig = style.styled_plot(title="Signal over time", ylabel="Signal", figsize=(16, 8))
    fig_ref, ax_ref = style.styled_plot(
        title="Reference power over time", ylabel="Power / kW", figsize=(16, 8)
    )

    first = next(iter(raw_output.values()), None)
    if not first or "signals" not in first:
        print("No 'signals' in raw_output; skip signal plots.")
        return

    for k, signal in first["signals"].items():
        if not isinstance(signal, sig_types.SecondOrderSignal):
            print(f"Signal not SecondOrderSignal but {type(signal)}")
            return
        x = np.arange(start=k, stop=k + signal.signal_len)
        sig = ax_sig.plot(x, signal.mul_lambda, label=f"k = {k}", drawstyle="steps-post")
        ax_sig.plot(k, signal.mul_lambda[0], marker="s", color=sig[0].get_color(), label=None)
        ref = ax_ref.plot(x, signal.res_power_set, label=f"k = {k}", drawstyle="steps-post")
        ax_ref.plot(k, signal.res_power_set[0], marker="s", color=ref[0].get_color(), label=None)

    ax_sig.legend()
    ax_ref.legend()
    fig_sig.tight_layout()
    fig_ref.tight_layout()


# ---------------------------------------------------------------------------
# Per-agent KPI analysis
# ---------------------------------------------------------------------------

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


def _cycles_from_soc(soc: np.ndarray) -> float:
    """Return charging-throughput in *equivalent full cycles on SoC basis*.

    This equals sum of positive SoC increments. If SoC is in [0,1], this is EFC.
    """
    if soc.size < 2:
        return 0.0
    diff = np.diff(soc)
    return float(diff[diff > 0].sum())


def agent_analysis(
    agent_ts: Mapping[str, pd.DataFrame], dt_h: float
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Compute KPIs for each agent.

    Definitions / Notes
    -------------------
    - All energy metrics integrate power with `dt_h` to kWh.
    - `total_feed` is export energy to grid (kWh, positive magnitude).
    - `max_feed` is the maximum export power magnitude (kW, positive value).
    - EV energy is integrated from `ev_p_net` (kW), not from SoC.
    - Self-consumption is PV generation used on-site: PV_gen - feed_to_grid, bounded below by 0.
    - Profit = revenue - costs, where revenue is positive cash inflow from exports.
    """
    energy_map = {"hp": "hp_energy_el", "ev": "ev_energy"}
    agents_res = pd.DataFrame(
        index=list(agent_ts.keys()), columns=sorted(AGENT_KPI_FIELDS), dtype=float
    )

    for ag, df in agent_ts.items():
        res: Dict[str, float] = {k: np.nan for k in AGENT_KPI_FIELDS}

        # Basic consumption/cost metrics
        grid = df["grid"].to_numpy()
        c_sup = df["c_sup"].to_numpy() if "c_sup" in df.columns else np.zeros_like(grid)
        c_feed = df["c_feed"].to_numpy() if "c_feed" in df.columns else np.zeros_like(grid)

        demand_mask = grid >= 0
        feed_mask = grid < 0

        res.update(
            {
                "grid_demand": _masked_sum(grid, demand_mask, dt_h),
                "max_grid_demand": _safe_max(grid[demand_mask]) if demand_mask.any() else 0.0,
                "costs": _masked_sum(grid * c_sup, demand_mask, dt_h),
            }
        )

        # Battery metrics
        if {"bat_soc", "bat_p_ac", "bat_p_net"}.issubset(df.columns):
            bat_soc = df["bat_soc"].to_numpy()
            bat_p_ac = df["bat_p_ac"].to_numpy()
            bat_p_net = df["bat_p_net"].to_numpy()

            charge_mask = bat_p_net > 0
            grid_charge_mask = (bat_p_net > 0) & (grid > 0)
            grid_discharge_mask = (bat_p_ac < 0) & (grid < 0)

            res.update(
                {
                    "battery_energy": _masked_sum(bat_p_net, charge_mask, dt_h),
                    "battery_energy_from_grid": _masked_sum(bat_p_ac, grid_charge_mask, dt_h),
                    "battery_max_from_grid": (
                        _safe_max(bat_p_ac[grid_charge_mask]) if grid_charge_mask.any() else np.nan
                    ),
                    "battery_energy_to_grid": _masked_sum(bat_p_ac, grid_discharge_mask, dt_h),
                    "battery_max_to_grid": (
                        _safe_min(bat_p_ac[grid_discharge_mask])
                        if grid_discharge_mask.any()
                        else np.nan
                    ),
                    "overcharge_bat": float((bat_soc > 1).sum()),
                    "undercharge_bat": float((bat_soc < 0).sum()),
                    "charging_cycle_equivalents": _cycles_from_soc(bat_soc),
                }
            )

        # Heat pump metrics
        if {"hp_temp", "hp_p_in"}.issubset(df.columns):
            hp_temp = df["hp_temp"].to_numpy()
            hp_p_in = df["hp_p_in"].to_numpy()
            hp_mask = hp_p_in > 0
            upper_mask = hp_temp > 23
            lower_mask = hp_temp < 18

            res.update(
                {
                    "mean_temp": float(hp_temp.mean()) if hp_temp.size else np.nan,
                    "hp_energy_el": _masked_sum(hp_p_in, hp_mask, dt_h),
                    "hp_p_max": _safe_max(hp_p_in[hp_mask]) if hp_mask.any() else np.nan,
                    "overheating": float(upper_mask.sum()),
                    "underheating": float(lower_mask.sum()),
                    "above_t": float(upper_mask.sum()),
                    "under_t": float(lower_mask.sum()),
                }
            )

        # EV metrics
        if {"ev_soc", "ev_p_ac", "ev_p_net"}.issubset(df.columns):
            ev_soc = df["ev_soc"].to_numpy()
            ev_p_ac = df["ev_p_ac"].to_numpy()
            ev_p_net = df["ev_p_net"].to_numpy()

            charge_mask_ev = ev_p_net > 0
            grid_charge_mask_ev = (ev_p_net > 0) & (grid > 0)
            grid_discharge_mask_ev = (ev_p_ac < 0) & (grid < 0)

            res.update(
                {
                    "ev_energy": _masked_sum(ev_p_net, charge_mask_ev, dt_h),
                    "ev_energy_from_grid": _masked_sum(ev_p_ac, grid_charge_mask_ev, dt_h),
                    "ev_max_from_grid": (
                        _safe_max(ev_p_ac[grid_charge_mask_ev])
                        if grid_charge_mask_ev.any()
                        else np.nan
                    ),
                    "ev_energy_to_grid": _masked_sum(ev_p_ac, grid_discharge_mask_ev, dt_h),
                    "overcharge_ev": float((ev_soc > 1).sum()),
                    "undercharge_ev": float((ev_soc < 0).sum()),
                }
            )

        # Overall energy demand sans battery netting
        p_el_load = df["p_el_load"].to_numpy() if "p_el_load" in df.columns else np.zeros_like(grid)
        systems = [
            k for k in ("hp", "ev") if (energy_map[k] in res and not np.isnan(res[energy_map[k]]))
        ]
        total_demand_kwh = sum(res[energy_map[k]] for k in systems) + float(p_el_load.sum()) * dt_h
        res["energy_consumption"] = total_demand_kwh

        # PV + self-consumption + revenue/profit
        if "p_el_pv" in df.columns:
            pv = df["p_el_pv"].to_numpy()
            # Export energy magnitude (kWh, positive):
            export_kwh = _masked_sum(
                -grid, feed_mask, dt_h
            )  # -grid turns negative export power into positive magnitude
            pv_gen_kwh = float(pv[pv > 0].sum()) * dt_h
            self_consumption_kwh = max(0.0, pv_gen_kwh - export_kwh)

            max_feed_kw = abs(_safe_min(grid[feed_mask])) if feed_mask.any() else np.nan

            # Revenue: cash inflow from export; assume feed-in price is positive in `c_feed`
            revenue = _masked_sum((-grid) * c_feed, feed_mask, dt_h)

            res.update(
                {
                    "total_feed": export_kwh,
                    "max_feed": max_feed_kw,
                    "revenue": revenue,
                    "self-consumption": self_consumption_kwh,
                    "self-sufficiency": (
                        (self_consumption_kwh / total_demand_kwh)
                        if total_demand_kwh > 0
                        else np.nan
                    ),
                }
            )

        # Profit calculation
        if not np.isnan(res.get("revenue", np.nan)) and not np.isnan(res.get("costs", np.nan)):
            res["profit"] = res.get("revenue", 0.0) - res.get("costs", 0.0)

        agents_res.loc[ag] = pd.Series(res, dtype=float)

    # Aggregate statistics
    numeric_agents = agents_res.astype(float)
    agent_stats: Dict[str, pd.Series] = {
        "mean": numeric_agents.mean(numeric_only=True),
        "median": numeric_agents.median(numeric_only=True),
        "25_quantile": numeric_agents.quantile(0.25, numeric_only=True),
        "75_quantile": numeric_agents.quantile(0.75, numeric_only=True),
        "max": numeric_agents.max(numeric_only=True),
        "min": numeric_agents.min(numeric_only=True),
    }

    return agents_res, agent_stats


# ---------------------------------------------------------------------------
# Battery cycles (public API maintained for backwards compatibility)
# ---------------------------------------------------------------------------


def calculate_ccs(soc: np.ndarray) -> float:
    """Compatibility shim: return equivalent full cycles based on SoC increments.

    Formerly returned "charging energy only". Now returns sum of positive SoC
    deltas which equals EFC if SoC is normalized [0,1].
    """
    return _cycles_from_soc(soc)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


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
    out_dir: pathlib.Path = run_pars.output_file_dir
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


# ---------------------------------------------------------------------------
# Main evaluation entrypoint
# ---------------------------------------------------------------------------


def evaluate_sim(sim_result: results.SimulationResult) -> Dict[str, Any]:
    """Evaluate a simulation result and persist KPIs.

    Returns a dict with keys: "general_res", "agent_res", "agent_stats".
    """
    # Aggregated KPIs (Series)
    agg_series = _get_aggregated_ts_result(sim_result)

    # Flex analysis (Series)
    flex_series = flex_analysis(sim_result.agents_ts, sim_result.sys_pars,
                                float(agg_series["dt_h"]))  # type: ignore[arg-type]

    # Compose one evaluation row
    eval_row = pd.concat(
        [
            agg_series,
            flex_series.drop(labels=["en_in_bat_ts"], errors="ignore"),
            pd.Series(
                {
                    "tag": sim_result.run_pars.sim_tag,
                    "calc_time": sim_result.execution_time,
                }
            ),
        ]
    )

    eval_df = eval_row.to_frame().T
    eval_df.index = [sim_result.run_pars.sim_tag]
    eval_df.index.name = "tag"

    # Per-agent analysis
    agent_kpis, agent_stats = agent_analysis(sim_result.agents_ts, float(agg_series["dt_h"]))  # type: ignore[arg-type]

    # Logging to console (compact)
    print(f"Analysis result for sim {sim_result.run_pars.sim_tag}: {eval_row}")
    _write_to_files(sim_result.run_pars, eval_df, "kpis")

    # Persist agent statistics with the tag row
    stats_with_tag: Dict[str, pd.Series] = {}
    for key, ser in agent_stats.items():
        ser = ser.copy()
        ser.loc["tag"] = sim_result.run_pars.sim_tag
        stats_with_tag[key] = ser
    print(
        f"Mean agent analysis result for sim {sim_result.run_pars.sim_tag}: {agent_stats['mean']}"
    )
    _write_to_files(sim_result.run_pars, stats_with_tag, "agent_stats")

    return {"general_res": eval_row, "agent_res": agent_kpis, "agent_stats": agent_stats}
