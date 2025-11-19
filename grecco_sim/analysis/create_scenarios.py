import pandas as pd
import os
import numpy as np
import shutil
from pathlib import Path
from grecco_sim.util import config
import logging

logging.basicConfig(level=logging.INFO)


def generate_higher_scenarios(
    base_folder, output_folder, database_folder, scenario_configs, random_seed=None
):
    """
    Generate higher penetration scenarios
    Args:
        base_folder (str): Path to the base scenario folder (2024)
        output_folder (str): Path to save the generated scenarios
        database_folder (str): Path to the database folder (full_el_evconservative)
        scenario_configs (dict): Dictionary defining unit counts for each scenario
        random_seed (int, optional): Seed for random number generator
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    allowed_cfgs = {
    ("load", "pv"),
    ("load",),
    ("bat", "load", "pv"),
    ("bat", "load"),
    ("ev", "load", "pv"),
    ("ev", "ev", "load", "pv"),
    ("ev", "ev", "load"),
    ("ev", "load"),
    ("hp", "load"),
    ("hp", "load", "pv"),
    ("bat", "ev", "load", "pv"),
    ("bat", "ev", "ev", "load", "pv"),
    ("bat", "hp", "load", "pv"),
    ("ev", "hp", "load", "pv"),
    ("ev", "ev", "hp", "load", "pv"),
    ("ev", "ev", "load", "pv"),
    ("bat", "hp", "load"),
    ("ev", "hp", "load"),
    ("ev", "ev", "load"),
    ("bat", "ev", "hp", "load", "pv"),
    }

    # Load database to select units from
    db_loads = pd.read_csv(os.path.join(database_folder, "loads.csv"))
    db_storage = pd.read_csv(os.path.join(database_folder, "storage_units.csv"))
    db_generators = pd.read_csv(os.path.join(database_folder, "generators.csv"))
    db_heat_pumps = db_loads.drop(db_loads[db_loads["carrier"] != "heat_pump"].index)
    db_evs = db_storage[db_storage["org_ev_name"].notna()]
    db_bats = db_storage[db_storage["org_ev_name"].isna()]
    slack = db_generators[db_generators["carrier"] != "solar"]

    # db_evs = db_evs_reduce(db_evs)

    # initialize base data
    base_data = {}
    base_data["loads"] = db_loads[db_loads["carrier"] != "heat_pump"].copy()

    # for penetration reduction
    evs_on_hold = pd.DataFrame()

    # Process each scenario
    for scenario_name, unit_counts in scenario_configs.items():

        if scenario_name == "2024":
            # generate base scenario
            logging.info("Creating base scenario...")

            new_hps = db_heat_pumps.sample(n=unit_counts["heat_pumps"])
            base_data["loads"] = pd.concat([base_data["loads"], new_hps])

            new_pv = db_generators.sample(n=unit_counts["pv_systems"] - 1)
            new_pv = pd.concat([new_pv, slack])  # ensure MV generator is included
            base_data["generators"] = new_pv

            available_bats = db_bats[
                db_bats["bus"].isin(new_pv["bus"])
            ]  # only households with pv can have a bat
            new_bats = available_bats.sample(n=unit_counts["batteries"])
            base_data["storage_units"] = new_bats

            new_ev = db_evs[db_evs["bus"].isin(new_pv["bus"])].sample(1)
            new_evs = pd.concat([new_ev, db_evs.sample(n=unit_counts["electric_vehicles"] - 1)])
            base_data["storage_units"] = pd.concat(
                [base_data["storage_units"], new_evs], ignore_index=True
            )

            # Identify unit types
            hp_households = set()
            ev_households = set()
            evs = set()
            bat_households = set()
            pv_households = set()

            # Get households with heat pumps
            for _, load in base_data["loads"].iterrows():
                if load.get("carrier") == "heat_pump":
                    household_id = load["bus"]
                    hp_households.add(household_id)

            # Get households with storage units or EVs
            for _, storage in base_data["storage_units"].iterrows():
                household_id = storage["bus"]

                if type(storage.get("org_ev_name", "")) == str:
                    ev_households.add(household_id)  # EVs
                    evs.add(storage["name"])
                else:
                    bat_households.add(household_id)  # Batteries

            # Identify households with PV
            for _, gen in base_data["generators"].iterrows():
                household_id = gen["bus"]
                pv_households.add(household_id)

            # Verify we have PV households if batteries are to be added
            if any(
                "batteries" in config and config["batteries"] > 0
                for config in scenario_configs.values()
            ):
                if not pv_households:
                    raise ValueError(
                        "No PV households found in base scenario. Cannot add batteries."
                    )

            # configurations of hhs with units
            flex_hhs = (
                set(hp_households) | set(ev_households) | set(bat_households) | set(pv_households)
            )
            hh_config = {}
            for hh in flex_hhs:
                hh_config[hh] = {
                    "hp": 1 if hh in hp_households else 0,
                    "ev": 1 if hh in ev_households else 0,
                    "bat": 1 if hh in bat_households else 0,
                    "pv": 1 if hh in pv_households else 0,
                }

        logging.info(f"Generating scenario: {scenario_name}")

        # Create scenario directory
        scenario_path = os.path.join(output_folder, scenario_name)
        os.makedirs(scenario_path, exist_ok=True)

        # Calculate additional units needed
        current_hps = len(hp_households)
        current_evs = len(evs)
        current_bats = len(bat_households)
        current_pv = len(pv_households)

        additional_hps = unit_counts.get("heat_pumps", 0) - current_hps
        additional_evs = unit_counts.get("electric_vehicles", 0) - current_evs
        additional_bats = unit_counts.get("batteries", 0) - current_bats
        additional_pv = unit_counts.get("pv_systems", 0) - current_pv

        # Add heat pumps
        if additional_hps > 0:

            # select hps from database. ensure only unused ones are selected.
            available_hps = db_heat_pumps[~db_heat_pumps["bus"].isin(hp_households)]
            if len(available_hps) < additional_hps:
                raise ValueError("Not enough unused heat pumps available.")
            new_hps = available_hps.sample(n=additional_hps)
            base_data["loads"] = base_data["loads"]._append(new_hps, ignore_index=True)

            # update households
            hp_households.update(new_hps["bus"].values)

        # Add EVs
        # --- Add EVs --------------------------------------------------------------
        if additional_evs > 0:
            reused = pd.DataFrame()
            nonprio_evs = pd.DataFrame()

            if not evs_on_hold.empty:
                reuse_evs = min(len(evs_on_hold), additional_evs)
                reused = evs_on_hold.sample(n=reuse_evs)
                evs_on_hold = evs_on_hold.drop(reused.index).reset_index(drop=True)
                evs.update(reused["name"].values)
                additional_evs -= reuse_evs

            # Step 1: EVs for HH without any EV
            available_evs = db_evs[~db_evs["name"].isin(evs)]
            hh_without_ev = available_evs[~available_evs["bus"].isin(ev_households)]

            to_add_first = min(len(hh_without_ev), additional_evs)
            new_evs_first = hh_without_ev.sample(n=to_add_first)
            additional_evs -= to_add_first

            # Step 2: if still more EVs desired, allow households to receive a 2nd EV
            new_evs_second = pd.DataFrame()
            if additional_evs > 0:
                hh_with_ev = available_evs[available_evs["bus"].isin(ev_households)]
                if len(hh_with_ev) < additional_evs:
                    raise ValueError("Not enough EVs available for second assignments.")
                new_evs_second = hh_with_ev.sample(n=additional_evs)

            # Final set of newly added EVs
            new_evs = pd.concat([reused, new_evs_first, new_evs_second], ignore_index=True)
            base_data["storage_units"] = pd.concat(
                [base_data["storage_units"], new_evs], ignore_index=True
            )

            # update sets
            ev_households.update(new_evs["bus"].values)
            evs.update(new_evs["name"].values)
                    )



        # penetration level reduction for EVs
        elif additional_evs < 0:
            # take from hh with two evs first
            hh_with_multiple_evs = {hh: 0 for hh in ev_households if list(evs).count(hh) > 1}
            evs_on_hold = available_evs[
                available_evs["bus"].isin(hh_with_multiple_evs.keys())
            ].sample(n=abs(additional_evs))
            base_data["storage_units"] = base_data["storage_units"][
                ~base_data["storage_units"]["name"].isin(evs_on_hold["name"])
            ]
            evs.difference_update(evs_on_hold["name"].values)

        # Add PV
        if additional_pv > 0:
            available_pv = db_generators[~db_generators["bus"].isin(pv_households)]
            if len(available_pv) < additional_pv:
                raise ValueError("Not enough unused PVs available.")
            # get PVs from new battery households
            new_pv = available_pv.sample(n=additional_pv)
            base_data["generators"] = base_data["generators"]._append(new_pv, ignore_index=True)

            # update households
            pv_households.update(new_pv["bus"].values)

        # Add batteries
        if additional_bats > 0:
            # select bats from database. ensure only unused ones are selected.
            available_bats = db_bats[~db_bats["bus"].isin(bat_households)]
            # only households with pv
            buses_with_pv = base_data["generators"]["bus"]  # households with pv
            available_bats = available_bats[available_bats["bus"].isin(buses_with_pv) == True]
            if available_bats.empty:
                raise ValueError("Not enough unused batteries available.")
            new_bats = available_bats.sample(n=additional_bats)
            base_data["storage_units"] = base_data["storage_units"]._append(
                new_bats, ignore_index=True
            )

            # update households
            bat_households.update(new_bats["bus"].values)

        # Save updated files
        logging.info("Saving base data...")
        base_data["loads"].to_csv(os.path.join(scenario_path, "loads.csv"), index=False)
        base_data["storage_units"].to_csv(
            os.path.join(scenario_path, "storage_units.csv"), index=False
        )
        base_data["generators"].to_csv(os.path.join(scenario_path, "generators.csv"), index=False)
        logging.info("Done.")
        logging.info("Completing scenario data...")

        # complete data with missing files
        # complete_data(base_data, database_folder, scenario_path)
        logging.info("Scenario completed.\n")

    return

def get_current_config(bus, hp_households, ev_households, bat_households, pv_households, base_data):
    """Return current configuration tuple for a given household bus."""
    cfg = []
    if bus in bat_households: cfg.append("bat")
    # count EVs explicitly
    ev_count = sum(base_data["storage_units"]["bus"] == bus)
    cfg += ["ev"] * ev_count
    if bus in hp_households: cfg.append("hp")
    cfg.append("load")
    if bus in pv_households: cfg.append("pv")
    return tuple(cfg)

def would_be_valid(bus, new_unit, allowed_cfgs, hp_households, ev_households, bat_households, pv_households, base_data):
    """Return whether adding `new_unit` keeps the configuration valid."""
    cfg = list(get_current_config(bus, hp_households, ev_households, bat_households, pv_households, base_data))
    cfg.append(new_unit)
    return tuple(sorted(cfg, key=lambda x: ["bat","ev","hp","load","pv"].index(x) if x in ["bat","ev","hp","load","pv"] else x)) in allowed_cfgs

def complete_data(base_data, database_folder, scenario_path):
    """
    Complete scenario data by copying missing files from base scenario
    and select units by base data
    Args:
        base_data (dict): Dictionary with new unit data from which to select units from database_folder
        database_folder (str): Path to the database folder from which to copy missing files
        scenario_path (str): Path to the scenario folder
    """
    copy_files = [
        "connection_points",
        "investment_periods",
        "line_types",
        "lines",
        "transformer_types",
        "transformers",
        "network",
        "shapes",
        "snapshots",
        "switches",
        "buses",
    ]

    for file in copy_files:
        shutil.copy(
            os.path.join(database_folder, f"{file}.csv"),
            scenario_path,
        )

    generator_files = ["generators-p_set", "generators-p_set_max", "generators-p_set_min"]
    heat_files = ["heat_demand", "loads-p_set"]  # , "loads-p_set_max", "loads-p_set_min"]
    storage_files = [
        "storage_units-p_set",
        "storage_units-p_set_max",
        "storage_units-p_set_min",
        "storage_units-state_of_charge",
    ]
    ev_file = "storage_units-plugged_in"

    storages = {}
    evs = base_data["storage_units"][base_data["storage_units"]["org_ev_name"].notna()]["name"]
    df = pd.read_csv(os.path.join(database_folder, f"{ev_file}.csv"), index_col=0)
    df[evs].to_csv(os.path.join(scenario_path, f"{ev_file}.csv"), index=True)
    for file in storage_files:
        df = pd.read_csv(os.path.join(database_folder, f"{file}.csv"), index_col=0)
        storages[file] = df[evs]

    # hps = base_data["loads"][base_data["loads"]["carrier"] == "heat_pump"]["name"]
    loads = base_data["loads"]["name"]
    # loads_bus = base_data["loads"]["bus"]
    hp_bus = base_data["loads"][base_data["loads"]["carrier"] == "heat_pump"]["bus"]
    for file in heat_files:
        df = pd.read_csv(os.path.join(database_folder, f"{file}.csv"), index_col=0)
        try:
            df[loads].to_csv(os.path.join(scenario_path, f"{file}.csv"), index=True)
        except KeyError:
            df[hp_bus].to_csv(os.path.join(scenario_path, f"{file}.csv"), index=True)

    bats = base_data["storage_units"][base_data["storage_units"]["org_ev_name"].isna()]["name"]
    for file in storage_files:
        df = pd.read_csv(os.path.join(database_folder, f"{file}.csv"), index_col=0)
        storages[file][bats] = df[bats]
        storages[file].to_csv(os.path.join(scenario_path, f"{file}.csv"), index=True)

    pvs = base_data["generators"][base_data["generators"]["carrier"] == "solar"]["name"]
    for file in generator_files:
        df = pd.read_csv(os.path.join(database_folder, f"{file}.csv"), index_col=0)
        df[pvs].to_csv(os.path.join(scenario_path, f"{file}.csv"), index=True)


def db_evs_reduce(db_evs):
    """Reduce the EV database to unique households only, keeping one EV per household.

    Args:
        db_evs (pd.DataFrame): DataFrame containing EV storage units with 'bus' and 'name' columns.

    Returns:
        pd.DataFrame: Reduced DataFrame with unique households and one EV per household.
    """
    # Keep only one EV per household (bus)
    return db_evs.loc[db_evs.groupby("bus")["name"].idxmax()]


base_folder = config.data_root() / "2024" / "2024"
output_folder = "higher_penetration_scenario_5"
scenario_configs = {
    "2024": {
        "heat_pumps": 21,
        "electric_vehicles": 4,
        "batteries": 22,
        "pv_systems": 37,
    },
    "2028_evconservative": {
        "heat_pumps": 26,
        "electric_vehicles": 64,
        "batteries": 34,
        "pv_systems": 47,
    },
        "2028_evextreme": {
        "heat_pumps": 26,
        "electric_vehicles": 118,
        "batteries": 34,
        "pv_systems": 47,
    },
    "2033_evconservative": {
        "heat_pumps": 31,
        "electric_vehicles": 114,
        "batteries": 47,
        "pv_systems": 69,
    },
    "2033_evextreme": {
        "heat_pumps": 31,
        "electric_vehicles": 204,
        "batteries": 47,
        "pv_systems": 68,
    },
    "2037_evconservative": {
        "heat_pumps": 41,
        "electric_vehicles": 153,
        "batteries": 59,
        "pv_systems": 86,
    },
        "2037_evextreme": {
        "heat_pumps": 41,
        "electric_vehicles": 277,
        "batteries": 59,
        "pv_systems": 86,
    },
    "2045_evconservative": {
        "heat_pumps": 51,
        "electric_vehicles": 186,
        "batteries": 67,
        "pv_systems": 102,
    },
        "2045_evextreme": {
        "heat_pumps": 51,
        "electric_vehicles": 336,
        "batteries": 67,
        "pv_systems": 102,
    },
    "full_el_evconservative": {
        "heat_pumps": 131,
        "electric_vehicles": 346,
        "batteries": 131,
        "pv_systems": 132,
    },
}


# Generate scenarios
generate_higher_scenarios(
    base_folder=base_folder,
    output_folder=output_folder,
    database_folder=os.path.join(os.getcwd(), "data", "full_el_evextreme"),
    scenario_configs=scenario_configs,
    random_seed=42,
)
