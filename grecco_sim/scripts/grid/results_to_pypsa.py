import pathlib
import pandas as pd
import pypsa
import os
import re
import shutil

kW_TO_MW = 1e-3


class ResultsToPypsa:

    # Initialization
    def __init__(
        self, original_grid_object_path, results_folder, grid_results_folder, multiplier=1
    ):
        """
        multiplier is used to scale the results to e.g. force a grid violation
        """
        self.original_grid_object_path = original_grid_object_path
        self.results_folder = results_folder
        self.grid_results_folder = grid_results_folder
        self.start_grid = pypsa.Network(original_grid_object_path)

        self.mutliplier = multiplier

    def generate_and_export_grid_objects(self, get_indigo_attributes=True):
        """
        get_indigo_attributes copies the csv files of switches and connection points
        into the
        """
        results = [f for f in os.listdir(self.results_folder) if "flex_power" in f]
        for result in results:
            grid_object = self.generate_result_grid_object(result)
            match = re.search(r"flex_power_(.*?)\.csv", result)
            extracted_str = match.group(1)
            export_folder = self.grid_results_folder / extracted_str  # Remove the .csv extension
            os.makedirs(export_folder, exist_ok=True)
            grid_object.export_to_csv_folder(export_folder)
            if get_indigo_attributes:
                files_to_copy = ["connection_points.csv", "switches.csv"]
                for f in files_to_copy:
                    source = os.path.join(self.original_grid_object_path, f)
                    shutil.copy(source, export_folder)
                print("Indigo attributes have been copied to target folder")

    def generate_result_grid_object(self, result):
        grid_copy = self.start_grid.copy()
        result_df = pd.read_csv(self.results_folder / result, index_col=0, sep=";") * kW_TO_MW
        result_df.index = pd.to_datetime(result_df.index).strftime("%Y-%m-%d %H:%M:%S")
        flexibility_columns = [col for col in result_df.columns if "_p_in" in col or "_p_ac" in col]
        filtered_result_df = result_df[flexibility_columns]

        # Adding heat pump results
        hp_columns = [col for col in filtered_result_df.columns if "hp" in col]
        hp_loads = self.get_hp_load_names(hp_columns, grid_copy)
        heat_pump_df = filtered_result_df.rename(columns=hp_loads)[hp_loads.values()]
        for col in heat_pump_df.columns:  # Issues with datetime
            new_profile = heat_pump_df[col]._values
            grid_copy.loads_t["p_set"].loc[heat_pump_df.index, col] = new_profile
            new_profile = heat_pump_df[col]._values * self.mutliplier
            grid_copy.loads_t["p_set"].loc[heat_pump_df.index, col] = new_profile

        # Adding storage units results
        ev_columns = [col for col in filtered_result_df.columns if "ev" in col]
        bat_columns = [col for col in filtered_result_df.columns if "bat" in col]
        storage_columns = ev_columns + bat_columns
        storage_names_dict = self.get_storage_names(storage_columns, grid_copy)
        storage_df = filtered_result_df.rename(columns=storage_names_dict)[
            storage_names_dict.values()
        ]
        for col in storage_df.columns:
            new_profile = storage_df[col]._values
            grid_copy.storage_units_t["p_set"].loc[storage_df.index, col] = new_profile
            new_profile = storage_df[col]._values * self.mutliplier
            grid_copy.storage_units_t["p_set"].loc[storage_df.index, col] = new_profile

        return grid_copy

    def get_hp_load_names(self, hp_columns, grid_copy):
        loads = grid_copy.loads
        heat_pumps = loads[loads.carrier == "heat_pump"]
        hp_loads = {}
        for col in hp_columns:
            bus = self.get_bus_from_name(col)
            heat_pump = heat_pumps[heat_pumps.bus == bus].index
            hp_loads[col] = heat_pump[0]
        return hp_loads

    def get_storage_names(self, storage_columns, grid_copy):
        storage_units = grid_copy.storage_units
        ev_rows = [row for row in storage_units.index if "emob" in row]
        bat_rows = [row for row in storage_units.index if "battery" in row]
        evs = storage_units.loc[ev_rows]
        batteries = storage_units.loc[bat_rows]
        storage_names = {}
        for col in storage_columns:
            bus = self.get_bus_from_name(col)
            if col in evs:
                storage_unit = evs[evs.bus == bus].index
                storage_names[col] = storage_unit[0]
            if col in batteries:
                storage_unit = batteries[batteries.bus == bus].index
                storage_names[col] = storage_unit[0]
        return storage_names

    def get_bus_from_name(self, name):
        match = re.search(r"bus_(.*?_\d+)_load", name)

        if match:
            extracted_str = match.group(1)
            return extracted_str
        else:
            match = re.search(r"bus_(\d+)_", name)  # In case the bus name is just a number
            if match:
                extracted_str = match.group(1)
                return extracted_str
            else:
                raise ValueError(f"No match found in string: {name}")


if __name__ == "__main__":
    main_folder = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
    original_grid_object_path = main_folder / "data" / "Opfingen_Scenarios" / "2024"
    results_folder = main_folder / "results" / "default" / "250704_0913"
    grid_results_folder = main_folder / "results" / "default" / "grid_object_forced_violation"
    # "results" / "parallelized" / "test_availables" / "second_order_it_5"
    results_to_pypsa = ResultsToPypsa(
        original_grid_object_path, results_folder, grid_results_folder
    )
    results_to_pypsa = ResultsToPypsa(
        original_grid_object_path, results_folder, grid_results_folder, multiplier=5
    )
    results_to_pypsa.generate_and_export_grid_objects(get_indigo_attributes=True)
