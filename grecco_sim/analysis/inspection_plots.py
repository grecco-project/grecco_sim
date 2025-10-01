
"""
THis module is thought of as some plots that can be used when using the debugger.
Just to look at something.
"""
import numpy as np
from grecco_sim.util import style


def plot_grid_powers(
        flex_sum: np.ndarray, inflex_sum: np.ndarray, y_static: np.ndarray,
        constr_val,                      tol=1e-3):
    fig, ax = style.styled_plot(figsize="landscape", title=f"Constraint: {constr_val: 0.04f}", ylabel="Grid Power / kW")
    ax.plot(inflex_sum, label="inflex")

    ax.plot(flex_sum, label="flex")
    ax.plot(y_static + flex_sum, label="constraint")
    ax.plot(y_static-inflex_sum, label="ag_slack")

    for i in range(len(y_static)):
        if abs((y_static + flex_sum)[i]) > tol:
            ax.axvspan(i, i+1, alpha=0.3)

    fig.legend()

    return ax


def plot_sec_order_ref_power(ref_power_grid, inflex_sum, flex_sum, mul_lam):


    ystatic = ref_power_grid["ag_slack"]

    ag_sum = np.array([yg for yg in ref_power_grid.values()]).sum(axis=0)
    constraint_value = np.abs(ag_sum).sum() / len(ag_sum)

    fig, ax = style.styled_plot(figsize="landscape", title=f"Constraint = {constraint_value}")
    ax.plot(ystatic, label="static")
    ax.plot(inflex_sum, label="inflex")
    ax.plot(flex_sum, label="flex")
    ax.plot(ag_sum - ystatic, label="sum - static (From OCP)")
    ax.plot(mul_lam * 100., label="lambda")
    fig.legend()


    

