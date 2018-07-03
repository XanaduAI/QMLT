#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-arguments
"""
Plotting functions
========================================================

**Module name:** :mod:`qmlt.numerical.plot`

.. currentmodule:: qmlt.numerical.plot

.. codeauthor:: Josh Izaac <josh@xanadu.ai>

This module contains the functions required to plot the parameters
of the numeric learner.

These are auxillary functions, it is recommended you instead use the
plotting method available in the numeric learner, which will provide
live plots of the training progress and monitored parameters. This can be
turned on by passing the ``plot`` key to the hyperparameters dictionary.

For example,

>>> hyperparams = {'circuit': circuit,
               'log_every': 10,
               'plot': True
               }

Here, the integer value of ``log_every`` specifies at how many global steps
the live plots should be updated. When the training is complete, the terminal
will show the message

.. code-block:: console

    Training complete. Close the live plot window to exit.

To use auxillary plotting functions on a logfile:

>>> from qmlt.numerical import plot
>>> plot.plot_parameter(numerical, y='loss')

You can also chain together plots by passing through the returned
axes, to display multiple parameters on one plot:

>>> ax = plot.plot_parameter(numerical, y='loss')
>>> ax = plot.plot_parameter(numerical, y='cost', ax=ax)
>>> ax = plot.plot_parameter(numerical, y='regul', ax=ax,
...                          legend=True, save_filename="test.png")

Finally, you can also automatically plot all parameters against the global
step, on one figure as multiple subplots:

>>> plot.plot_all("numerical/logsNUM/log.csv")

Plotting functions
------------------

.. autosummary::
    plot_parameter
    plot_all

Auxillary functions
-------------------

.. autosummary::
    _squareish
    _plot

Code details
------------
"""
import os
from itertools import zip_longest
import numpy as np

try:
    import matplotlib as mpl
    if os.environ.get('DISPLAY', '') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
    from matplotlib import pyplot as plt
except:
    raise ImportError("To use the plotting functions, matplotlib must be installed")


def _squareish(n):
    """Factors an integer to two integers that closesly approximates a square

    Args:
        n (int): integer to factor.

    Returns:
        tuple(int, int): the squareish integers.
    """
    if n == 1:
        return (1, 1)
    if n == 2:
        return (2, 1)

    nsqrt = np.ceil(np.sqrt(n))
    solution = False
    x = int(nsqrt)
    while not solution:
        y = int(n/x)
        if y * x == float(n):
            solution = True
        else:
            x -= 1

    if n > 1:
        if x == 1 or y == 1:
            x = 3
            y = int(np.ceil(n/3))

    return x, y


def _plot(x, y, ax=None, xlabel=None, ylabel=None, **plot_kw):
    r"""Produces a line plot visualizing settings, training progress, or monitored parameters.

    Args:
        x (array): the data to plot on the x-axis.
        y (array): the data to plot on the y-axis.
        xlabel (str): the x-axis label.
        ylabel (str): the y-axis label.
        **plot_kw: additional keyword arguments to be passed to ``plt.plot``.

    Returns:
        axes: returns a tuple containing the figure and axes.
    """

    if ax is None:
        ax = plt.gca()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.plot(x, y, **plot_kw)

    return ax


def plot_parameter(logfile, x='global_step', y='loss', save_filename=None,
                   ax=None, legend=False, style='ggplot', legend_kw=None,
                   fig_kw=None, savefig_kw=None, **plot_kw): # pragma: no cover
    r"""Produces a line plot visualizing settings, training progress, or monitored parameters.

    Args:
        logfile (str): the location of the logfile containing the training progress
            and parameter values for each global step.
        x (str): the parameter to plot on the x-axis. By default the global step.
        y (str): the parameter to plot on the y-axis. By default the loss.
        save_filename (str): string containing the output image filename, including
            all relevant paths and file extensions. All image filetypes supported
            by matplotlib are supported here. By default, the plot is *not* saved.
        ax (matplotlib.axes.Axes): a matplotlib axes object. If none is provided,
            this is created automatically.
        legend (bool): If True, a legend is added containing the y parameter names.
        style (str): a supported matplotlib style sheet. To see the available
            styles on your system, please refer to the output of
            ``matplotlib.pyplot.style.available``.
        legend_kw (dict): dictionary of additional matplotlib keyword arguments
            to apss to ``matplotlib.pyplot.legend``.
        fig_kw (dict): dictionary of additional matplotlib keyword arguments
            to apss to ``matplotlib.figure.Figure``.
        savefig_kw (dict): dictionary of additional matplotlib keyword arguments
            to apss to ``matplotlib.figure.Figure.savefig``.
        **plot_kw: additional keyword arguments to be passed to ``matplotlib.pyplot.plot``.

    Returns:
        matplotlib.axes.Axes: returns the plotting axes.
    """
    # pragma: no cover
    if fig_kw is None:
        fig_kw = {'figsize': (12, 8)}

    if savefig_kw is None:
        savefig_kw = {}

    if legend_kw is None:
        legend_kw = {}

    data = np.genfromtxt(logfile, dtype=float, delimiter=',', names=True)
    params = data.dtype.names

    if x not in params:
        raise ValueError("The x-axis parameter {} does not exist.".format(x))

    if y not in params:
        raise ValueError("The y-axis parameter {} does not exist.".format(y))

    plt.style.use(style)

    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True, **fig_kw)

    if legend:
        ax = _plot(data[x], data[y], label=y, ylabel="", ax=ax, **plot_kw)
        plt.legend()
    else:
        ax = _plot(data[x], data[y], label=y, ylabel=y, xlabel=x, ax=ax, **plot_kw)

    if save_filename is not None:
        plt.savefig(save_filename, **savefig_kw)

    return ax


def plot_all(logfile, x='global_step', y=None, save_filename=None,
             figax=None, style='ggplot', fig_kw=None, savefig_kw=None, **plot_kw): # pragma: no cover
    r"""Produces a figure containing line plots visualizing settings,
    training progress, or monitored parameters.

    Args:
        filename (str): string containing the output image filename, including
            all relevant paths and file extensions. All image filetypes supported
            by matplotlib are supported here.
        logfile (str): the location of the logfile containing the training progress
            and parameter values for each global step.
        x (str): the parameter to plot on the x-axes. By default the global step.
        y Sequence[str]: the parameters to plot on the figure. By default, all will be plotted.
        save_filename (str): string containing the output image filename, including
            all relevant paths and file extensions. All image filetypes supported
            by matplotlib are supported here. By default, the plot is *not* saved.
        figax (tuple): a tuple containing the figure and the plotting axes. Created
            by default if not provided.
        style (str): a supported matplotlib style sheet. To see the available
            styles on your system, please refer to the output of
            ``matplotlib.pyplot.style.available``.
        fig_kw (dict): dictionary of additional matplotlib keyword arguments
            to apss to ``matplotlib.figure.Figure``.
        savefig_kw (dict): dictionary of additional matplotlib keyword arguments
            to apss to ``matplotlib.figure.Figure.savefig``.
        **plot_kw: additional keyword arguments to be passed to ``matplotlib.pyplot.plot``.

    Returns:
        tuple: returns a tuple containing the figure and the plotting axes.
    """
    if fig_kw is None:
        fig_kw = {'figsize': (12, 8)}

    if savefig_kw is None:
        savefig_kw = {}

    data = np.genfromtxt(logfile, dtype=float, delimiter=',', names=True)
    params = data.dtype.names

    if x not in params:
        raise ValueError("The x-axis parameter {} does not exist.".format(x))

    xdata = data[x]

    if y is None:
        ydata = [data[p] for p in params if p != x]
        ylabels = [p for p in params if p != x]
    else:
        try:
            ydata = [data[p] for p in y]
        except ValueError:
            raise ValueError("parameter name does not exist in logfile.")
        ylabels = y

    rows, cols = _squareish(len(ydata))

    plt.style.use(style)
    if figax is None:
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=False, **fig_kw)
    else:
        fig, ax = figax

    for idx, (yd, yl, a) in enumerate(zip_longest(ydata, ylabels, ax.ravel())):
        # get 2D grid location
        # loc = np.array(np.unravel_index([idx], (rows, cols))).flatten()

        # only label x-axis if on the bottom row
        if yd is not None:
            # if loc[0] == rows-1:
            a = _plot(xdata, yd, xlabel=x, ylabel=yl, ax=a, **plot_kw)
            # else:
                # a = _plot(xdata, yd, ylabel=yl, ax=a, **plot_kw)
        else:
            a.axis('off')


    plt.tight_layout()

    if save_filename is not None:
        fig.savefig(save_filename, **savefig_kw)

    return fig, ax
