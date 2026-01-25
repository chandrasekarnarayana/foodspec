"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

Matplotlib-only plotting utilities writing files under an ArtifactRegistry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foodspec.core.artifacts import ArtifactRegistry


def _save(fig: plt.Figure, artifacts: ArtifactRegistry, filename: str) -> Path:
    artifacts.ensure_layout()
    path = artifacts.plots_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_raw_vs_processed(
    wavenumbers: np.ndarray,
    raw: np.ndarray,
    processed: np.ndarray,
    artifacts: ArtifactRegistry,
    filename: str = "raw_vs_processed.png",
) -> Path:
    """Overlay raw and processed spectra lines."""

    wn = np.asarray(wavenumbers)
    raw = np.asarray(raw)
    proc = np.asarray(processed)
    fig, ax = plt.subplots()
    ax.plot(wn, raw, label="raw", alpha=0.7)
    ax.plot(wn, proc, label="processed", alpha=0.7)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.set_title("Raw vs Processed")
    return _save(fig, artifacts, filename)


def plot_heatmap(
    matrix: np.ndarray,
    artifacts: ArtifactRegistry,
    title: str = "Heatmap",
    filename: str = "heatmap.png",
    xlabels: Sequence[str] | None = None,
    ylabels: Sequence[str] | None = None,
) -> Path:
    """Heatmap plot for a matrix."""

    mat = np.asarray(matrix)
    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax)
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
    ax.set_title(title)
    return _save(fig, artifacts, filename)


def plot_pca_scatter(
    scores: np.ndarray,
    labels: Sequence[object],
    artifacts: ArtifactRegistry,
    filename: str = "pca_scatter.png",
) -> Path:
    """2D scatter of PCA scores colored by labels."""

    s = np.asarray(scores)
    if s.shape[1] < 2:
        raise ValueError("scores must have at least two components")
    fig, ax = plt.subplots()
    labels_arr = np.asarray(labels)
    uniq = np.unique(labels_arr)
    for val in uniq:
        mask = labels_arr == val
        ax.scatter(s[mask, 0], s[mask, 1], label=str(val), alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Scatter")
    ax.legend()
    return _save(fig, artifacts, filename)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    artifacts: ArtifactRegistry,
    filename: str = "confusion_matrix.png",
) -> Path:
    """Confusion matrix heatmap."""

    mat = np.asarray(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color="black")
    return _save(fig, artifacts, filename)


def plot_reliability_diagram(
    y_true: Sequence[int],
    proba: np.ndarray,
    artifacts: ArtifactRegistry,
    n_bins: int = 10,
    filename: str = "reliability_diagram.png",
) -> Path:
    """Reliability diagram using confidence bins (max probability)."""

    y = np.asarray(y_true)
    p = np.asarray(proba)
    if p.ndim != 2:
        raise ValueError("proba must be 2D")
    conf = p.max(axis=1)
    preds = p.argmax(axis=1)
    correct = (preds == y).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    accs = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        bin_centers.append((lo + hi) / 2)
        accs.append(float(correct[mask].mean()))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    if bin_centers:
        ax.plot(bin_centers, accs, marker="o", label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return _save(fig, artifacts, filename)


__all__ = [
    "plot_raw_vs_processed",
    "plot_heatmap",
    "plot_pca_scatter",
    "plot_confusion_matrix",
    "plot_reliability_diagram",
]
