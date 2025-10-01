"""
Module with some analysis over different simulation runs.
"""

from grecco_sim.simulator import results
from grecco_sim.util import type_defs
from grecco_sim.util import style


def _read_results(
    run_par_sets: list[type_defs.RunParameters],
) -> dict[str, results.SimulationResult]:
    all_results = {}

    for run_params in run_par_sets:
        sim_res = results.SimulationResult()
        sim_res.from_files(run_params.output_file_dir)
        all_results[run_params.sim_tag] = sim_res

    return all_results


def _label(sim_tag: str) -> str:
    if "self_suff" in sim_tag:
        return "Standard Control"
    elif "plain_grid_fee" in sim_tag:
        return "First Order plain grid fee"
    elif "none" in sim_tag:
        return "No batteries"
    elif "gradient_descent" in sim_tag:
        return f"Max iterations {sim_tag.split('_')[-1]}"

    return sim_tag


def compare(run_par_sets: list[type_defs.RunParameters]):
    """Plot the grid time series of a set of sim results for an arbitrary agent to comapre."""
    all_results = _read_results(run_par_sets)

    # Plot Trafo power
    fig, ax = style.styled_plot(figsize=(11, 7), ylabel="Power at transformer / kW", date_axis=True)

    for sim_tag, sim_res in all_results.items():
        ax.plot(
            sim_res.trafo_sum,
            label=sim_tag + " " + sim_res.opt_pars.solver_name,
            drawstyle="steps-post",
        )

        if sim_res.run_params.coordination_mechanism == "plain_grid_fee":
            signals = sim_res.assigned_grid_fees.drop(columns="iterations").sum(axis=1)

            ax.plot(
                signals,
                label=f"Assigned signals {sim_res.opt_pars.solver_name}",
                drawstyle="steps-post",
            )

            ax.plot(
                signals.index,
                [sim_res.grid_pars.p_lim] * len(signals),
                label="Grid Limit Load",
            )
            ax.plot(
                signals.index,
                [-sim_res.grid_pars.p_lim] * len(signals),
                label="Grid Limit Feed-In",
            )

    ax.legend()
    fig.tight_layout()
    # plt.show()
