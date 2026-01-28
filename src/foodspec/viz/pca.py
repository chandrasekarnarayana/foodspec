"""PCA visualization helpers."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_pca_scores", "plot_pca_loadings"]


def plot_pca_scores(
    scores: np.ndarray,
    labels: Optional[Sequence[Any]] = None,
    ax=None,
):
    """Scatter plot of PCA scores (PC1 vs PC2)."""

    scores = np.asarray(scores)
    if scores.shape[1] < 2:
        raise ValueError("scores must have at least 2 components.")
    ax = ax or plt.gca()
    if labels is None:
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.8)
    else:
        labels = np.asarray(labels)
        for i, lbl in enumerate(np.unique(labels)):
            mask = labels == lbl
            ax.scatter(scores[mask, 0], scores[mask, 1], label=str(lbl), alpha=0.8)
        ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return ax


def plot_pca_loadings(loadings: np.ndarray, wavenumbers: np.ndarray, ax=None):
    """Plot PCA loadings vs wavenumbers."""

    loadings = np.asarray(loadings)
    wavenumbers = np.asarray(wavenumbers)
    if loadings.shape[0] != wavenumbers.shape[0]:
        raise ValueError("loadings rows must match wavenumbers length.")
    ax = ax or plt.gca()
    for i in range(loadings.shape[1]):
        ax.plot(wavenumbers, loadings[:, i], label=f"PC{i + 1}")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Loading")
    ax.invert_xaxis()
    ax.legend()
    return ax
