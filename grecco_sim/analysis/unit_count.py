import pandas as pd
import os
from grecco_sim.util import config


def count_units_in_scenario(scenario_folder):
    """
    Count heat pumps, electric vehicles, batteries and PV systems in a scenario folder.

    Args:
        scenario_folder (str): Path to the scenario folder containing CSV files

    Returns:
        dict: Dictionary with counts for:
              - 'heat_pumps': Number of heat pumps (from loads.csv)
              - 'electric_vehicles': Number of electric vehicles (from storage_units.csv)
              - 'batteries': Number of batteries (from storage_units.csv)
              - 'pv_systems': Number of PV systems (from generators.csv)
    """
    # Initialize counts
    counts = {"heat_pumps": 0, "electric_vehicles": 0, "batteries": 0, "pv_systems": 0}

    # Count heat pumps from loads.csv
    loads_path = os.path.join(scenario_folder, "loads.csv")
    if os.path.exists(loads_path):
        loads = pd.read_csv(loads_path)
        if "carrier" in loads.columns:
            counts["heat_pumps"] = (loads["carrier"] == "heat_pump").sum()
        else:
            print(f"Warning: 'carrier' column not found in {loads_path}")

    # Count electric vehicles from storage_units.csv
    storage_path = os.path.join(scenario_folder, "storage_units.csv")
    if os.path.exists(storage_path):
        storage_units = pd.read_csv(storage_path)
        if "org_ev_name" in storage_units.columns:
            # Non-empty org_ev_name indicates EV presence
            ev_mask = storage_units["org_ev_name"].notna() & (storage_units["org_ev_name"] != "")
            counts["electric_vehicles"] = ev_mask.sum()
            counts["batteries"] = len(storage_units) - counts["electric_vehicles"]
        else:
            print(f"Warning: 'org_ev_name' column not found in {storage_path}")

    # Count PV systems from generators.csv
    generators_path = os.path.join(scenario_folder, "generators.csv")
    if os.path.exists(generators_path):
        generators = pd.read_csv(generators_path)
        counts["pv_systems"] = len(generators)

    return counts


# Example usage:
def analyze_scenarios(base_folder, szenario):
    """
    Analyze all scenario folders in the base directory.

    Args:
        base_folder (str): Path to the base directory containing scenario folders
    """
    folder_path = base_folder / szenario
    scenarios = []

    if os.path.isdir(folder_path):
        counts = count_units_in_scenario(folder_path)
        scenario_info = {
            "scenario": szenario,
            "heat_pumps": counts["heat_pumps"],
            "electric_vehicles": counts["electric_vehicles"],
            "batteries": counts["batteries"],
            "pv_systems": counts["pv_systems"],
        }
        scenarios.append(scenario_info)

    # Create a DataFrame for easier analysis
    df = pd.DataFrame(scenarios).set_index("scenario")
    return df


scenarios = []
years = [2024, 2028, 2033, 2037, 2045, "full_el"]
for year in years:
    data_root = config.data_root() / f"{year}"

    ev_cases = ["evconservative", "evextreme"]
    if year != 2024:
        for ev_case in ev_cases:
            szenario = f"{year}_{ev_case}"
            scenarios.append(analyze_scenarios(data_root, szenario))
    elif year == 2024:
        szenario = f"{year}"
        scenarios.append(analyze_scenarios(data_root, szenario))
final_df = pd.concat(scenarios)
print(final_df)
