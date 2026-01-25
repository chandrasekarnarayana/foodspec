"""Classification visualization helpers."""
from __future__ import annotations


from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_confusion_matrix", "plot_roc_curves"]


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
