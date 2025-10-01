import pandas as pd
import pathlib
import matplotlib.pyplot as plt

grid_data_path  =  pathlib.Path(__file__).parent.parent.parent 
load_path  = grid_data_path / "Opfingen_Profiles_2023" / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all" / "loads-p.csv"
snaps_path = grid_data_path / "Opfingen_Profiles_2023" / "Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023_with_h0_batterypf_all" / "snapshots.csv"

load = pd.read_csv(load_path, sep=",")
snapshots = pd.read_csv(snaps_path, sep=",", parse_dates=["snapshot"])
snapshots["snapshot"].index = pd.to_datetime(snapshots["snapshot"], format = '%Y-%m-%d %H:%M:%S%z')


plt.plot(snapshots["snapshot"].iloc[970:1066], load.iloc[970:1066,8])
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Load Profile")
plt.show()