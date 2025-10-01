"""
Simple descriptive study of a low-voltage SimBench power grid 
"""

# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.plotting as plot
import simbench as sb

# LV SimBench power grid
sb_code = "1-LV-semiurb4--2-no_sw"
net_simbench = sb.get_simbench_net(sb_code)
net_simbench

# Simple plot pf the SimBench power grid
plot.simple_plot(net_simbench)

# Extract the profiles
profiles = sb.get_absolute_values(net_simbench, profiles_instead_of_study_cases = True)
profiles.keys()

# Types of static generators
net_simbench.sgen["type"].unique()

# Static generation (active power)
total_static_generation = pd.DataFrame(profiles[('sgen', 'p_mw')].sum(axis = 1))
total_static_generation.rename(columns = {0: "total_static_generation"}, inplace = True)
total_static_generation.describe()

# Load (active power)
total_load_active_power = pd.DataFrame(profiles[('load', 'p_mw')].sum(axis=1))
total_load_active_power.rename(columns = {0: "total_load_active_power"}, inplace = True)
total_load_active_power.describe()

# Storage (active power)
total_storage = pd.DataFrame(profiles[('storage', 'p_mw')].sum(axis=1))
total_storage.rename(columns = {0: "total_storage"}, inplace=True)
total_storage.describe()

# Plot active power of storage
plt.plot(total_storage, color = "salmon")
plt.title("Storage")
plt.xlabel("Timestep")
plt.ylabel("Active Power (MW)")
plt.show()

# Load and Generation Comparison
plt.plot(range(0, 96), total_load_active_power["total_load_active_power"][0:96], label = "Load", color = "blue")
plt.plot(range(0, 96), total_static_generation["total_static_generation"][0:96], label = "Generation", color = "red")
plt.title("Load vs. Generation - One Day")
plt.xlabel("Timestep")
plt.ylabel("Active Power (MW)")
legend = plt.legend(loc = "upper left", edgecolor = "black")
legend.get_frame().set_linewidth(0.8)
plt.show()

plt.plot(range(0, 672), total_load_active_power["total_load_active_power"][0:672], label = "Load", color = "blue")
plt.plot(range(0, 672), total_static_generation["total_static_generation"][0:672], label = "Generation", color = "red")
plt.title("Load vs. Generation - One Week")
plt.xlabel("Timestep")
plt.ylabel("Active Power (MW)")
legend = plt.legend(loc = "upper left", edgecolor = "black")
legend.get_frame().set_linewidth(0.8)
plt.show()

plt.plot(total_load_active_power.index, total_load_active_power["total_load_active_power"], label = "Load", color = "blue")
plt.plot(total_static_generation.index, total_static_generation["total_static_generation"], label = "Generation", color = "red")
plt.title("Load vs. Generation - One Year")
plt.xlabel("Timestep")
plt.ylabel("Active Power (MW)")
legend = plt.legend(loc = "upper left", edgecolor = "black")
legend.get_frame().set_linewidth(0.8)
plt.show()

# Power flow analysis - One week simulation 
time_steps = 672

# Initialization
bus_voltage = pd.DataFrame()
line_loading = pd.DataFrame()
transformer_loading = pd.DataFrame()
ext_grid_active_power = pd.DataFrame()
ext_grid_reactive_power = pd.DataFrame()
load_active_power = pd.DataFrame()
load_reactive_power = pd.DataFrame()
static_gen_active_power = pd.DataFrame()
static_gen_reactive_power = pd.DataFrame()
storage_active_power = pd.DataFrame()
storage_reactive_power = pd.DataFrame()

# For loop
for timepoint in range(time_steps):

  # Insert the relavant values into the grid
  net_simbench.gen["p_mw"] = profiles[('gen', 'p_mw')].T[timepoint]
  net_simbench.load["p_mw"] = profiles[('load', 'p_mw')].T[timepoint]
  net_simbench.load["q_mvar"] = profiles[('load', 'q_mvar')].T[timepoint]
  net_simbench.sgen["p_mw"] = profiles[('sgen', 'p_mw')].T[timepoint]
  net_simbench.storage["p_mw"] = profiles[('storage', 'p_mw')].T[timepoint]

  # Run the powerflow
  pp.runpp(net_simbench, numba = False)

  # Save the results at each timestep
  bus_voltage[timepoint] = net_simbench.res_bus["vm_pu"]
  line_loading[timepoint] = net_simbench.res_line["loading_percent"]
  transformer_loading[timepoint] = net_simbench.res_trafo["loading_percent"]
  ext_grid_active_power[timepoint] = net_simbench.res_ext_grid["p_mw"]
  ext_grid_reactive_power[timepoint] = net_simbench.res_ext_grid["q_mvar"]
  load_active_power[timepoint] = net_simbench.res_load["p_mw"]
  load_reactive_power[timepoint] = net_simbench.res_load["q_mvar"]
  static_gen_active_power[timepoint] = net_simbench.res_sgen["p_mw"]
  static_gen_reactive_power[timepoint] = net_simbench.res_sgen["q_mvar"]
  storage_active_power[timepoint] = net_simbench.res_storage["p_mw"]
  storage_reactive_power[timepoint] = net_simbench.res_storage["q_mvar"]

# Visualize the power flow results at t = 671 (last timestep in the week)
plot.pf_res_plotly(net_simbench)

# Calculate the bus voltage average 
total_bus_voltage_transposed = bus_voltage.transpose().mean(axis=1)

# Plot bus voltage average 
plt.plot(total_bus_voltage_transposed)
plt.title("Average Bus Voltages")
plt.xlabel("Timestep")
plt.ylabel("Bus Voltage (p.u.)")
plt.show()

# Select the lines with loading percentage above 100% at each timestep
overloaded_lines = []
number_overloaded_lines = []

for i in range(time_steps):
  line_index = line_loading[line_loading[i] > 100].index.tolist()
  number_overloaded_lines.append(len(line_index)) 

  for j in range(len(line_index)):
    overloaded_lines.append(net_simbench.line.loc[line_index[j]])

overloaded_lines_df = pd.DataFrame(overloaded_lines)
number_overloaded_lines_df = pd.DataFrame(number_overloaded_lines)