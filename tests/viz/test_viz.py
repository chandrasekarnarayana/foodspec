import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foodspec.apps.oils import OilAuthResult
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.viz.classification import plot_confusion_matrix, plot_roc_curves
from foodspec.viz.pca import plot_pca_loadings, plot_pca_scores
from foodspec.viz.report import render_html_report_oil_auth
from foodspec.viz.spectra import plot_mean_spectrum, plot_spectra


def _simple_dataset():
    wavenumbers = np.linspace(800, 1200, 50)
    x = np.vstack([np.sin(wavenumbers / 100) + i for i in range(3)])
    metadata = pd.DataFrame({"sample_id": ["a", "b", "c"], "group": ["g1", "g1", "g2"]})
    return FoodSpectrumSet(x=x, wavenumbers=wavenumbers, metadata=metadata, modality="raman")


def test_plot_spectra_and_mean():
    ds = _simple_dataset()
    ax1 = plot_spectra(ds, color_by="group")
    assert isinstance(ax1, matplotlib.axes.Axes)
    plt.close(ax1.figure)
    ax2 = plot_mean_spectrum(ds, group_by="group")
    assert isinstance(ax2, matplotlib.axes.Axes)
    plt.close(ax2.figure)


def test_plot_pca():
    scores = np.array([[1, 2], [2, 3], [3, 4]])
    labels = ["a", "b", "a"]
    ax1 = plot_pca_scores(scores, labels=labels)
    assert isinstance(ax1, matplotlib.axes.Axes)
    plt.close(ax1.figure)

    loadings = np.random.randn(5, 2)
    wavenumbers = np.linspace(1000, 1500, 5)
    ax2 = plot_pca_loadings(loadings, wavenumbers)
    assert isinstance(ax2, matplotlib.axes.Axes)
    plt.close(ax2.figure)


def test_classification_plots():
    cm = np.array([[5, 1], [0, 4]])
    class_names = ["c1", "c2"]
    ax1 = plot_confusion_matrix(cm, class_names)
    assert isinstance(ax1, matplotlib.axes.Axes)
    plt.close(ax1.figure)

    curves = {"model": (np.array([0, 0.5, 1.0]), np.array([0, 0.8, 1.0]))}
    ax2 = plot_roc_curves(curves)
    assert isinstance(ax2, matplotlib.axes.Axes)
    plt.close(ax2.figure)


def test_render_html_report(tmp_path):
    cm = np.array([[3, 1], [2, 4]])
    cv_metrics = pd.DataFrame({"fold": [1, 2], "accuracy": [0.8, 0.9]})
    result = OilAuthResult(
        pipeline=None,
        cv_metrics=cv_metrics,
        confusion_matrix=cm,
        class_labels=["olive", "sunflower"],
        feature_importances=None,
    )
    output = render_html_report_oil_auth(result, tmp_path / "report.html")
    assert output.exists()
    assert output.read_text()
