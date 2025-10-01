import pathlib
import pandas as pd
import pypsa 
import simbench
import math 

class SimbenchToPypsa():
    # Initialization
    def __init__(self, sb_code = "1-LV-semiurb4--2-no_sw"):
        self.sb_code = sb_code
        self.net_simbench = simbench.get_simbench_net(self.sb_code)
        self.profiles = simbench.get_absolute_values(self.net_simbench, profiles_instead_of_study_cases = True)
        self.net_pypsa = pypsa.Network() 

    # Adding network components from SimBench to PyPSA 
    def simbench_to_pypsa(self):

        # Adding the snapshots (one year - 15 min resolution)
        # self.net_simbench.profiles["load"]["time"] (to get all the snapshots but duplicate datetime instances can occur)
        start_datetime = self.net_simbench.profiles["load"]["time"].iloc[0]
        end_datetime = self.net_simbench.profiles["load"]["time"].iloc[-1]
        snapshots = pd.date_range(start = start_datetime, end = end_datetime, freq = "15T")
        snapshots = pd.Series(snapshots)
        snapshots = snapshots.dt.tz_localize("UTC")
        self.net_pypsa.set_snapshots(snapshots)

        # Adding buses 
        for index, bus in self.net_simbench.bus.iterrows():
            self.net_pypsa.add("Bus", name = str(self.net_simbench.bus_geodata.index[index]), v_nom = bus["vn_kv"],
                        x = self.net_simbench.bus_geodata.loc[index, "x"], y = self.net_simbench.bus_geodata.loc[index, "y"],
                        sub_network = bus["subnet"], v_mag_pu_max = bus["max_vm_pu"], 
                        v_mag_pu_min = bus["min_vm_pu"])

        # Adding lines 
        for index, line in self.net_simbench.line.iterrows():
            self.net_pypsa.add("Line", name = str(index), bus0 = str(line["from_bus"]), bus1 = str(line["to_bus"]), 
                        length = line["length_km"], num_parallel = line["parallel"], s_nom = math.sqrt(3) * line["voltLvl"] * line["max_i_ka"],
                        x = line["x_ohm_per_km"] * line["length_km"], 
                        r = line["r_ohm_per_km"] * line["length_km"],
                        g = line["g_us_per_km"] * 10**(-6) * line["length_km"],
                        b = 2 * math.pi * 50 * line["c_nf_per_km"] * 10**(-9) * line["length_km"],
                        sub_network = line["subnet"])

        # Adding static generators (renewable powerplants)
        for index, sgen in self.net_simbench.sgen.iterrows():
            self.net_pypsa.add("Generator", name = str(index), bus = str(sgen["bus"]),
                        p_set = sgen["p_mw"], q_set = sgen["q_mvar"], carrier = "solar" if sgen["type"] == "PV" else sgen["type"])

        # Adding normal generators (conventional powerplants)
        for index, gen in self.net_simbench.gen.iterrows():
            self.net_pypsa.add("Generator", name = str(index), bus = str(gen["bus"]),
                        p_set = gen["p_mw"], carrier = gen["type"])
            
        # Adding loads 
        for index, load in self.net_simbench.load.iterrows():
            self.net_pypsa.add("Load", name = str(index), bus = str(load["bus"]),
                        p_set = load["p_mw"], q_set = load["q_mvar"])

        # Adding transformers 
        for index, trafo in self.net_simbench.trafo.iterrows():
            self.net_pypsa.add("Transformer", name = str(index), bus0 = str(trafo["hv_bus"]),
                        bus1 = str(trafo["lv_bus"]), phase_shift = trafo["shift_degree"],
                        s_nom = trafo["sn_mva"], 
                        tap_side = 0 if trafo["tap_side"] == "hv" else 1 if trafo["tap_side"] == "lv" else 0,
                        tap_position = trafo["tap_pos"], num_parallel = trafo["parallel"],
                        sub_network = trafo["subnet"], tap_ratio = trafo["tap_step_percent"] / 100,
                        r = trafo["vkr_percent"] / 100, 
                        x = math.sqrt((trafo["vk_percent"] / 100)**2 - (trafo["vkr_percent"] / 100)**2))

        # Adding storage units 
        for index, storage in self.net_simbench.storage.iterrows():
            self.net_pypsa.add("StorageUnit", name = str(index), bus = storage["bus"],
                        type = storage["type"], p_set = storage["p_mw"], q_set = storage["q_mvar"],
                        p_nom = storage["max_p_mw"])
            
        # Adding time-varying data
        self.net_pypsa.loads_t["p_set"] = self.profiles[("load", "p_mw")]
        self.net_pypsa.loads_t["q_set"] = self.profiles[("load", "q_mvar")]

        if self.profiles[("gen", "p_mw")].empty and not self.profiles[("sgen", "p_mw")].empty:
            self.net_pypsa.generators_t["p_set"] = self.profiles[("sgen", "p_mw")]
        elif self.profiles[("sgen", "p_mw")].empty and not self.profiles[("gen", "p_mw")].empty:
            self.net_pypsa.generators_t["p_set"] = self.profiles[("gen", "p_mw")]
        else:
            self.net_pypsa.generators_t["p_set"] = self.profiles[("sgen", "p_mw")] + self.profiles[("gen", "p_mw")]

        self.net_pypsa.storage_units_t["p_set"] = self.profiles[("storage", "p_mw")]

        return self.net_pypsa
    
    # Generate PyPSA files 
    def generate_csv_files(self):
        path = pathlib.Path(__file__).parent.absolute()
        path = path.parent.parent
        path = path / "data" / "pypsa_files"
        self.net_pypsa.export_to_csv_folder(path)
    
if __name__ == "__main__":
    # Instantiate the class -  check https://simbench.readthedocs.io/en/stable/networks.html for getting more simbench examples
    class_instance = SimbenchToPypsa()
    net_pypsa = class_instance.simbench_to_pypsa()
    class_instance.generate_csv_files()
