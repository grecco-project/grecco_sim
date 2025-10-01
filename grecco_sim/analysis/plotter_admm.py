import numpy as np
import matplotlib.colors as mcolors

from grecco_sim.util import style


class Plotter(object):
    def __init__(self, horizon, sys_ids, p_lim):
        self.horizon = horizon
        self.p_lim = p_lim

        # for logging the convergence performance
        self.constraint_vals = []
        self.lam = {sys_id: [] for sys_id in sys_ids}

    def plot_signal_update(self, prev_signal, signal, k):
        fig, ax = style.styled_plot(y_label="Signal", x_label="Horizon index", figsize="landscape")

        for sys_id in signal:

            ax.plot(np.arange(self.horizon) + k, prev_signal[sys_id]["lambda"],
                    label=f"Previous Iteration {sys_id}", drawstyle="steps-post")
            ax.plot(np.arange(self.horizon) + k, signal[sys_id]["lambda"],
                    label=f"New Iteration {sys_id}", drawstyle="steps-post")

        ax.legend()

        fig2, ax2 = style.styled_plot(y_label="Central Solution", x_label="Horizon index", figsize="landscape")

        _prev_sys = np.zeros(self.horizon)
        index = np.arange(self.horizon) + k
        color_list = [xkcd_color[1] for xkcd_color in list(mcolors.XKCD_COLORS.items())[:len(signal)]]

        for i, sys_id in enumerate(signal):

            grid_power_sys = signal[sys_id]["res_power_set"]

            ax2.fill_between(index, _prev_sys, grid_power_sys + _prev_sys, color=color_list[i], alpha=0.2, step="post")
            ax2.plot(index, grid_power_sys + _prev_sys, label=sys_id, drawstyle="steps-post", color=color_list[i])
            _prev_sys += grid_power_sys

        ax2.plot(index, _prev_sys, label="Sum", linestyle="dashed", color="black", drawstyle="steps-post")
        ax2.plot(index, [self.p_lim] * len(_prev_sys), label="Limit Power", linestyle="dotted", color="black",
                 drawstyle="steps-post")

        ax2.legend(title=f"New signal slack+res_power {k}")
        fig.tight_layout

    def plot_futures(self, futures, k):

        color_list = [xkcd_color[1] for xkcd_color in list(mcolors.XKCD_COLORS.items())[:len(futures)]]

        fig, ax = style.styled_plot(xlabel="Horizon index", ylabel="Power", figsize="landscape",
                                    )
        _prev_sys = np.zeros(self.horizon)
        index = np.arange(self.horizon) + k

        for i, sys_id in enumerate(futures):

            grid_power_sys = futures[sys_id]["flex_schedule"]
            # grid_power_sys = futures[sys_id]["fc"] + futures[sys_id]["flex_schedule"]

            ax.fill_between(index, _prev_sys, grid_power_sys + _prev_sys, color=color_list[i], alpha=0.2, step="post")
            ax.plot(index, grid_power_sys + _prev_sys, label=sys_id, drawstyle="steps-post", color=color_list[i])
            _prev_sys += grid_power_sys

        ax.plot(index, _prev_sys, label="Sum", linestyle="dashed", color="black", drawstyle="steps-post")

        ax.legend(title=f"Futures before update at {k}")
        fig.tight_layout()

    def log_iteration(self, constraint_val, lam):
        self.constraint_vals += [constraint_val]
        for sys_id in lam:
            self.lam[sys_id] += [lam[sys_id].sum()]

    def clear_log(self):
        self.constraint_vals = []
        self.lam = {sys_id: [] for sys_id in self.lam}

    def plot_convergence(self, clear=True, k=-1):
        fig, ax = style.styled_plot(figsize="landscape", title=f"Constraint at k={k}")
        index = range(len(self.constraint_vals))
        ax.plot(index, self.constraint_vals, marker="s")

        fig2, ax2 = style.styled_plot(figsize="landscape", title=f"Multiplier at k={k}")
        for sys_id in self.lam:
            ax2.plot(index, self.lam[sys_id], marker="s", label=sys_id)

        ax2.legend()

        if clear:
            self.clear_log()


