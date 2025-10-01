import os
import pathlib
import cProfile
import pstats
import datetime
import traceback
from typing import Iterable, List, Dict

from concurrent.futures import ProcessPoolExecutor
import warnings

import pandas as pd

from grecco_sim.simulator import simulation_setup
from grecco_sim.util import data_io, type_defs


def _profiled_run(
    run_parameters: type_defs.RunParameters,
    opt_pars: type_defs.OptParameters,
    grid_pars: type_defs.GridDescription,
) -> Dict:
    """
    Perform single sim, but profile run.

    Inspect with
    ``snakeviz <stamp.prof>``
    """

    with cProfile.Profile() as pr:
        res = _run_single_sim(run_parameters, opt_pars, grid_pars)

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

        stats.dump_stats(filename=run_parameters.output_file_dir / f"{stamp}.prof")

    return res


def _run_single_sim(
    run_parameters: type_defs.RunParameters,
    opt_pars: type_defs.OptParameters,
    grid_pars: type_defs.GridDescription,
) -> Dict:
    """Perform a single simulation with given parameters."""
    try:
        simulator = simulation_setup.SimulationSetup(run_parameters)
        kpis = simulator.run_sim(opt_pars, grid_pars)
        return kpis
    except Exception:
        if not os.path.exists(run_parameters.output_file_dir):
            os.makedirs(run_parameters.output_file_dir, exist_ok=True)
        data_io.dump_parameterization(
            run_parameters.output_file_dir / "run_pars.json", [run_parameters, opt_pars, grid_pars]
        )
        with open(run_parameters.output_file_dir / "error.csv", "w", encoding="utf-8") as err_file:
            err_file.write(traceback.format_exc())
        return {}


def _summarize_single_sims(
    base_dir: pathlib.Path,
    sim_results: Iterable,
):
    # Make sure these col names have the same name as defined in simulation_eval!
    COL_NAMES = ["max_load", "max_feed", "costs_all", "agg_grid_fees", "calc_time"]

    res_df = pd.DataFrame(columns=COL_NAMES)

    for res in sim_results:
        if res:
            res_df.loc[res["tag"], :] = [res[col_name] for col_name in COL_NAMES]

    if os.path.exists(base_dir / "summary.csv"):
        res_df = pd.concat([pd.read_csv(base_dir / "summary.csv", index_col=0), res_df])

    res_df.to_csv(base_dir / "summary.csv")


def parallel_sim(
    base_dir: pathlib.Path,
    run_parameter_sets: List[type_defs.RunParameters],
    opt_par_sets: List[type_defs.OptParameters],
    grid_par_sets: List[type_defs.GridDescription],
    n_workers: int = 12,
):
    """Interface for parallel execution of grecco grid simulations."""

    assert len(run_parameter_sets) == len(opt_par_sets) == len(grid_par_sets)

    match len(run_parameter_sets):
        case 0:
            warnings.warn("Length of given simulations is 0. Nothing done.")

        case 1:
            print("Switching to single household mode.")
            if run_parameter_sets[0].profile_run:
                _profiled_run(
                    run_parameter_sets[0],
                    opt_par_sets[0],
                    grid_par_sets[0],
                )
            else:
                _run_single_sim(
                    run_parameter_sets[0],
                    opt_par_sets[0],
                    grid_par_sets[0],
                )

        case _:
            if n_workers == 1:
                warnings.warn("This is probably not as intended as it only runs 1 sim.")
                _run_single_sim(
                    run_parameter_sets[0],
                    opt_par_sets[0],
                    grid_par_sets[0],
                )

            else:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = executor.map(
                        _run_single_sim,
                        run_parameter_sets,
                        opt_par_sets,
                        grid_par_sets,
                    )

                _summarize_single_sims(base_dir, results)
