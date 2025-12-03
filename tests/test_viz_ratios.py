import numpy as np

from foodspec.viz import (
    plot_ratio_by_group,
    plot_ratio_scatter,
    plot_ratio_vs_continuous,
)


def test_plot_ratio_by_group_runs():
    ratios = np.array([1, 2, 3, 4])
    groups = np.array(["A", "A", "B", "B"])
    ax = plot_ratio_by_group(ratios, groups)
    assert hasattr(ax, "plot")


def test_plot_ratio_scatter_runs():
    rx = np.array([1, 2, 3])
    ry = np.array([3, 2, 1])
    ax = plot_ratio_scatter(rx, ry, group_labels=["x", "y", "y"])
    assert hasattr(ax, "scatter")


def test_plot_ratio_vs_continuous_runs():
    ratios = np.array([1, 2, 3, 4])
    cont = np.array([0, 1, 2, 3])
    ax = plot_ratio_vs_continuous(ratios, cont)
    assert hasattr(ax, "scatter")
