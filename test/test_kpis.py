from grecco_sim.simulator.metrics import AGENT_KPIS, NETWORK_KPIS

AGENT_KPI_FIELDS = {
    "grid_demand",
    "total_import",
    "max_import",
    "energy_consumption",
    "max_grid_demand",
    "total_feed",
    "max_feed",
    "self_consumption",
    "self_sufficiency",
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
    "net_grid_energy",
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

def test_completeness_of_KPIs():
    for KPI in AGENT_KPI_FIELDS:
        assert KPI in AGENT_KPIS.keys()
    for KPI in AGENT_KPIS.keys():
        assert KPI in AGENT_KPI_FIELDS