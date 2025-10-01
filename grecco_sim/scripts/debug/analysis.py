import pandas as pd

from grecco_sim.simulator import results 
from grecco_sim.analysis import simulation_eval
from grecco_sim.analysis import plotter



def load_results():

    sim_res = results.SimulationResult()

    sim_res.from_files("/home/agross/src/python/grecco/grecco_sim/results/parallelized/res_example/plain_grid_fee")

    simulation_eval.evaluate_sim(sim_res)
    plotter.make_plots(sim_res)


# def look_at_alvaros_file():
#     filename = "/home/agross/src/python/grecco_sim/data/Opfingen_Profiles_2023/Opfingen_scenario_pv_20_ev_efh_0_ev_mfh_0_evghd_0_ev_fleet_0_hp_15_2023pf_all_ts_reduced/snapshots.csv"
#     df = pd.read_csv(filename, index_col="snapshot", parse_dates=True)
#     df = df.iloc[:, 1:]

#     df.plot()
#     print(df)

# def find_exemplary_days():
#     sim_res = results.SimulationResult()
#     sim_res.from_files("/home/agross/src/python/grecco_sim/results/default/240912_1216")

#     trafo_ts = sim_res.agents_ts.sum(axis=1)

def main():

    load_results()
    # look_at_alvaros_file()
    # find_exemplary_days()



    


if __name__ == "__main__":
    main()

