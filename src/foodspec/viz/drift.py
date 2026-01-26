from __future__ import annotations

"""Drift and stability visualization helpers."""

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from foodspec.viz.style import apply_style


def plot_batch_drift(batch_names: Iterable[str], drift_scores: Iterable[float], *, seed: int = 0):
    """Plot batch drift scores as a bar chart."""
    apply_style()
    names = list(batch_names)
    scores = np.asarray(list(drift_scores), dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, len(names) * 0.6), 3))
    ax.bar(names, scores, color="#2a6fdb")
    ax.set_ylabel("Drift Score")
    ax.set_xlabel("Batch")
    ax.set_title("Batch Drift")
    ax.set_ylim(0.0, max(1.0, float(scores.max() if scores.size else 1.0)))
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_stage_difference_spectra(
    wavenumbers: Iterable[float],
    stage_spectra: Mapping[str, Iterable[float]],
    *,
    seed: int = 0,
):
    """Plot stage-wise difference spectra against a baseline stage."""
    apply_style()
    wn = np.asarray(list(wavenumbers), dtype=float)
    stages = list(stage_spectra.keys())
    if not stages:
        stages = ["baseline", "stage1"]
        stage_spectra = {
            "baseline": np.sin(wn / 200.0),
            "stage1": np.sin(wn / 200.0) + 0.05,
        }
    baseline = np.asarray(list(stage_spectra[stages[0]]), dtype=float)

    fig, ax = plt.subplots(figsize=(6, 3))
    for stage in stages[1:]:
        diff = np.asarray(list(stage_spectra[stage]), dtype=float) - baseline
        ax.plot(wn, diff, label=f"{stage} - {stages[0]}")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Delta Intensity (a.u.)")
    ax.set_title("Stage-wise Difference Spectra")
    ax.legend()
    ax.invert_xaxis()
    return fig


def plot_replicate_similarity(similarity_matrix: np.ndarray, *, seed: int = 0):
    """Plot replicate similarity matrix heatmap."""
    apply_style()
    mat = np.asarray(similarity_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax.set_title("Replicate Similarity")
    ax.set_xlabel("Replicate")
    ax.set_ylabel("Replicate")
    return fig


def plot_temporal_drift(time_points: Iterable[float], drift_values: Iterable[float], *, seed: int = 0):
    """Plot temporal drift trend."""
    apply_style()
    times = np.asarray(list(time_points), dtype=float)
    values = np.asarray(list(drift_values), dtype=float)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(times, values, marker="o", color="#2a6fdb")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drift Score")
    ax.set_title("Temporal Drift Trend")
    ax.grid(True, alpha=0.2)
    return fig


__all__ = [
    "plot_batch_drift",
    "plot_stage_difference_spectra",
    "plot_replicate_similarity",
    "plot_temporal_drift",
]
