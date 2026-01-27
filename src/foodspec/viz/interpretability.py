"""Interpretability and feature visualization helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from foodspec.viz.style import apply_style


def plot_importance_overlay(
    wavenumbers: Iterable[float],
    spectrum: Iterable[float],
    importance: Iterable[float],
    *,
    seed: int = 0,
):
    """Overlay importance weights on a spectrum plot."""
    apply_style()
    wn = np.asarray(list(wavenumbers), dtype=float)
    y = np.asarray(list(spectrum), dtype=float)
    weights = np.asarray(list(importance), dtype=float)
    if weights.shape != y.shape:
        weights = np.interp(np.linspace(0, 1, len(y)), np.linspace(0, 1, len(weights)), weights)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(wn, y, color="#1f77b4", label="spectrum")
    threshold = np.percentile(weights, 90) if weights.size else 0.0
    mask = weights >= threshold
    ax.fill_between(wn, y.min(), y.max(), where=mask, color="#ff8a65", alpha=0.3, label="importance")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Importance Overlay")
    ax.invert_xaxis()
    ax.legend()
    return fig


def plot_marker_bands(
    wavenumbers: Iterable[float],
    bands: Sequence[tuple[float, float]],
    *,
    seed: int = 0,
):
    """Plot marker bands as shaded regions."""
    apply_style()
    wn = np.asarray(list(wavenumbers), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(wn, np.zeros_like(wn), alpha=0.0)
    for start, end in bands:
        ax.axvspan(start, end, color="#2a6fdb", alpha=0.2)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_title("Marker Bands")
    ax.invert_xaxis()
    return fig


def plot_coefficient_heatmap(coefficients: np.ndarray, feature_names: Sequence[str], *, seed: int = 0):
    """Plot coefficient heatmap for linear models."""
    apply_style()
    coef = np.asarray(coefficients, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(coef, aspect="auto", cmap="coolwarm")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Class")
    ax.set_title("Coefficient Heatmap")
    return fig


def plot_feature_stability(stability_matrix: np.ndarray, feature_names: Sequence[str], *, seed: int = 0):
    """Plot feature stability across folds/bootstraps."""
    apply_style()
    mat = np.asarray(stability_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Fold")
    ax.set_title("Feature Stability Map")
    return fig


__all__ = [
    "plot_importance_overlay",
    "plot_marker_bands",
    "plot_coefficient_heatmap",
    "plot_feature_stability",
]
