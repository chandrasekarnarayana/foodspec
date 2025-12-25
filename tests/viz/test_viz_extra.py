import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.viz.pca import plot_pca_loadings, plot_pca_scores
from foodspec.viz.spectra import plot_mean_spectrum, plot_spectra


def _fs():
    x = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    wn = np.array([100.0, 200.0, 300.0])
    meta = pd.DataFrame({"group": ["a", "b"]})
    return FoodSpectrumSet(x=x, wavenumbers=wn, metadata=meta, modality="raman")


def test_plot_spectra_and_mean():
    fs = _fs()
    fig, ax = plt.subplots()
    plot_spectra(fs, color_by="group", ax=ax)
    plt.close(fig)
    fig, ax = plt.subplots()
    plot_mean_spectrum(fs, group_by="group", ax=ax)
    plt.close(fig)


def test_plot_pca_helpers():
    scores = np.array([[0, 1], [1, 0]])
    labels = ["a", "b"]
    fig, ax = plt.subplots()
    plot_pca_scores(scores, labels=labels, ax=ax)
    plt.close(fig)
    loadings = np.array([[0.1, 0.3], [0.2, 0.2], [0.3, 0.1]])
    wn = np.array([100.0, 200.0, 300.0])
    fig, ax = plt.subplots()
    plot_pca_loadings(loadings, wavenumbers=wn, ax=ax)
    plt.close(fig)
