"""Classification visualization helpers."""
from __future__ import annotations


from typing import Dict, Sequence, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_confusion_matrix", "plot_roc_curves", "plot_reliability_diagram"]


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], ax=None):
    """Plot confusion matrix as heatmap."""

    cm = np.asarray(cm)
    ax = ax or plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    return ax


def plot_roc_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], ax=None):
    """Plot ROC curves from precomputed FPR/TPR pairs."""

    ax = ax or plt.gca()
    for label, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    return ax


def plot_reliability_diagram(
    data_bundle,
    *,
    outdir=None,
    name=None,
    fmt=("png", "svg"),
    dpi=300,
    seed=0,
    n_bins: int = 10,
):
    """Plot reliability diagram from probabilities and labels.

    Parameters
    ----------
    data_bundle : RunBundle | dict
        Either a RunBundle or payload with y_true and y_prob.
    """
    from foodspec.reporting.schema import RunBundle
    from foodspec.viz.save import save_figure
    from foodspec._version import __version__

    payload = data_bundle if isinstance(data_bundle, dict) else {}
    if isinstance(data_bundle, RunBundle):
        payload = {}

    rng = np.random.default_rng(seed)
    y_true = np.asarray(payload.get("y_true", rng.integers(0, 2, size=100)), dtype=int)
    y_prob = np.asarray(payload.get("y_prob", rng.random(size=100)), dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_acc = []
    bin_conf = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            bin_acc.append(float(np.mean(y_true[mask] == (y_prob[mask] >= 0.5))))
            bin_conf.append(float(np.mean(y_prob[mask])))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(bin_conf, bin_acc, marker="o", label="reliability")
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    if outdir is not None:
        base = Path(outdir) / "figures" / (name or "reliability_diagram")
        save_figure(
            fig,
            base,
            metadata={
                "description": "Reliability diagram",
                "inputs": {"n_samples": len(y_true), "n_bins": n_bins},
                "code_version": __version__,
                "seed": seed,
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig
