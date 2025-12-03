import matplotlib

matplotlib.use("Agg")  # ensure headless for tests

import numpy as np

from foodspec.viz.plots import (
    plot_spectra_overlay,
    plot_mean_with_ci,
    plot_pca_scores,
    plot_pca_loadings,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_regression_calibration,
)


def test_plot_helpers_return_axes():
    wn = np.linspace(1000, 1100, 5)
    spectra = np.array([[1, 2, 3, 4, 5], [1.1, 2.1, 3.1, 4.1, 5.1]])
    ax = plot_spectra_overlay(spectra, wn)
    assert ax is not None
    ax = plot_mean_with_ci(spectra, wn, group_labels=["a", "b"])
    assert ax is not None
    scores = np.random.default_rng(0).normal(size=(5, 3))
    ax = plot_pca_scores(scores, labels=["x", "x", "y", "y", "y"])
    assert ax is not None
    loadings = np.random.default_rng(1).normal(size=(5, 3))
    ax = plot_pca_loadings(loadings, wn)
    assert ax is not None
    cm = np.array([[2, 1], [0, 3]])
    ax = plot_confusion_matrix(cm, ["a", "b"])
    assert ax is not None
    corr = np.array([[1, 0.5], [0.5, 1]])
    ax = plot_correlation_heatmap(corr, labels=["f1", "f2"])
    assert ax is not None
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1.1, 1.9])
    ax = plot_regression_calibration(y_true, y_pred)
    assert ax is not None
