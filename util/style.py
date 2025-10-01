
import pytz
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import matplotlib as mpl
from mpl_toolkits import mplot3d

import numpy as np

"""
This module assembles some things which where helpful for Arne in generating plots etc for the PHD work

They might be helpful for some other people as well

Labels for common column names in Dataframes etc
------------------------------------------------

"""


class LabelObject(object):
    """
    This class provides an object that has a [] lookup function just as a dictionnary.
    Except, it returns the key if no value matching it is present.

    Use it to generate plots with meaningful axis labels or data labels

    Exemplary usage:

    .. code-block:: python

        my_labels = {"power_ac": "AC Power of Battery"}
        my_label_object = LabelObject(my_labels)

        # Could also work with a dictionary
        ax.plot(my_df.index, my_df["power_ac"], label=my_label_obj["power_ac"])

        # Would raise a ValueError with a dictionary
        ax.plot(my_df.index, my_df["losses_bat"], label=my_label_obj["losses_bat"])

    Some objects with labels for common data series are defined below.

    """
    def __init__(self, label_dict):
        self.label_dict = label_dict

    def __getitem__(self, item):
        if item in self.label_dict:
            return self.label_dict[item]
        else:
            return item


"""
Accessing the plt.subplots() function with an easier interface
--------------------------------------------------------------

There are some attributes which are often modified for an axes object.
This Section intends to make these available with a single function call.

.. code-block:: python
    
    fig, ax = style.styled_plot(ylabel=ag_style.kpi_labels[kpi], xlabel=ag_style.kpi_labels[param], figsize="landscape")

"""
plot_defaults = dict(
    major_formatter="%H:%M",
    timezone=pytz.utc,
    figsize=(10, 10),
    date_axis=False,
    title="",
    xlabel="",
    ylabel="",
    ylim=None,
    xlim=None,
)
figsize_defaults = dict(landscape=(8, 4), policies=(8, 5), portrait=(4, 4), slim=(3, 4))


def styled_plot(**kwargs):

    # warnings.warn("should also offer an a posteriori function redoing the labels or something")
    # print(kwargs)

    specs = {}
    specs.update(plot_defaults)
    specs.update(kwargs)

    if specs["figsize"] is not None:
        if specs["figsize"] in figsize_defaults:
            fig, ax = plt.subplots(figsize=figsize_defaults[specs["figsize"]])
        else:
            fig, ax = plt.subplots(figsize=specs["figsize"])
    else:
        fig, ax = plt.subplots()

    if specs["date_axis"]:
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(specs["major_formatter"], tz=specs["timezone"])
        )
        ax.set_xlabel("Time")
    else:
        ax.set_xlabel(specs["xlabel"])

    ax.set_ylabel(specs["ylabel"])

    if specs["xlim"] is not None:
        ax.set_xlim(specs["xlim"])

    if specs["ylim"] is not None:
        ax.set_ylim(specs["ylim"])

    ax.set_title(specs["title"])

    ax.set_axisbelow(True)

    return fig, ax


def plot_2d(x_axis, y_axis, data, surface_3d=True, **kwargs):

    defaults = dict(y_label="", x_label="", z_label="", title="", cmap="YlGn")
    defaults.update(kwargs)

    fig = plt.figure()
    if surface_3d:
        ax = mplot3d.Axes3D(fig)
    else:
        ax = fig.add_subplot(1, 1, 1)

    X, Y = np.meshgrid(x_axis, y_axis)

    if surface_3d:
        ax.plot_surface(
            X,
            Y,
            data,
            rstride=1,
            cstride=1,
            cmap=plt.get_cmap(defaults["cmap"]),
            linewidth=0,
            antialiased=False,
        )
        ax.set_zlabel(defaults["z_label"], labelpad=20.0)
    else:
        p = ax.pcolor(X, Y, data, cmap=plt.get_cmap(defaults["cmap"]))
        fig.colorbar(p)

    ax.set_xlabel(defaults["x_label"], labelpad=0.0)
    ax.set_ylabel(defaults["y_label"], labelpad=0.0)

    ax.set_title(defaults["title"], y=1.05)


def latexify():
    params = {
        'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
        'font.family': 'serif'
    }
    mpl.rcParams.update(params)
