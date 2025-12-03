import numpy as np

from foodspec.viz.ratios import plot_ratio_by_group, plot_ratio_scatter, plot_ratio_vs_continuous


def test_ratio_by_group_variants():
    ratios = np.array([1.0, 1.1, 0.9, 1.5, 1.6, 1.4])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    for kind in ("box", "violin", "strip"):
        ax = plot_ratio_by_group(ratios, groups, kind=kind)
        assert ax is not None


def test_ratio_scatter_and_continuous():
    rx = np.linspace(0.5, 1.5, 10)
    ry = np.linspace(1.0, 2.0, 10)
    groups = np.array(["A"] * 5 + ["B"] * 5)
    ax = plot_ratio_scatter(rx, ry, group_labels=groups)
    assert ax is not None
    cont = np.linspace(0, 9, 10)
    ax = plot_ratio_vs_continuous(rx, cont)
    assert ax is not None
