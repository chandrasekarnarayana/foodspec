from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_ratio_by_group(ratio_values, group_labels, ax=None, kind: str = "box"):
    """
    Plot a ratio distribution across groups.

    Parameters
    ----------
    ratio_values : array-like
        Numeric ratio values (e.g., band A / band B).
    group_labels : array-like
        Group labels of same length as ratio_values.
    ax : matplotlib Axes, optional
        Axes to plot on; if None, a new figure/axes is created.
    kind : {'box', 'violin', 'strip'}, optional
        Plot style; defaults to 'box'.

    Returns
    -------
    matplotlib.axes.Axes
    """

    if ax is None:
        _, ax = plt.subplots()
    ratios = np.asarray(ratio_values)
    groups = np.asarray(group_labels)
    unique = np.unique(groups)
    data = [ratios[groups == g] for g in unique]

    if kind == "box":
        ax.boxplot(data, labels=unique, patch_artist=True)
    elif kind == "violin":
        parts = ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(unique) + 1))
        ax.set_xticklabels(unique)
    elif kind == "strip":
        for i, g in enumerate(unique):
            ax.scatter(np.full_like(data[i], i + 1), data[i], alpha=0.7)
        ax.set_xticks(range(1, len(unique) + 1))
        ax.set_xticklabels(unique)
    else:
        raise ValueError("kind must be 'box', 'violin', or 'strip'")

    ax.set_ylabel("Ratio value")
    ax.set_xlabel("Group")
    return ax


def plot_ratio_scatter(ratio_x, ratio_y, group_labels=None, ax=None):
    """
    Scatter plot of two ratios (or ratio vs ratio) with optional group coloring.

    Parameters
    ----------
    ratio_x, ratio_y : array-like
        Ratios or numeric features to plot on x/y axes.
    group_labels : array-like, optional
        Labels for coloring; if None, a single color is used.
    ax : matplotlib Axes, optional
        Axes to plot on; if None, created.

    Returns
    -------
    matplotlib.axes.Axes
    """

    if ax is None:
        _, ax = plt.subplots()
    ratio_x = np.asarray(ratio_x)
    ratio_y = np.asarray(ratio_y)
    if group_labels is not None:
        groups = np.asarray(group_labels)
        unique = np.unique(groups)
        for g in unique:
            mask = groups == g
            ax.scatter(ratio_x[mask], ratio_y[mask], label=str(g), alpha=0.8)
        ax.legend(title="Group")
    else:
        ax.scatter(ratio_x, ratio_y, alpha=0.8)
    ax.set_xlabel("Ratio X")
    ax.set_ylabel("Ratio Y")
    return ax


def plot_ratio_vs_continuous(ratio_values, y, ax=None):
    """
    Plot ratio vs a continuous outcome (e.g., heating time) with a trend line.

    Parameters
    ----------
    ratio_values : array-like
        Ratio values.
    y : array-like
        Continuous outcome (time, temperature, quality index).
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """

    if ax is None:
        _, ax = plt.subplots()
    ratio = np.asarray(ratio_values)
    cont = np.asarray(y)
    ax.scatter(cont, ratio, alpha=0.8)
    # simple linear fit
    if len(cont) > 1:
        coef = np.polyfit(cont, ratio, 1)
        xs = np.linspace(cont.min(), cont.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color="red", linestyle="--", label="trend")
        ax.legend()
    ax.set_xlabel("Continuous variable")
    ax.set_ylabel("Ratio")
    return ax
