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

Validation metrics: accuracy, macro F1, one-vs-rest AUROC, ECE, Brier.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _ensure_proba_2d(proba: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        # Binary case: interpret as positive class probability
        p = np.column_stack([1.0 - p, p])
    if p.ndim != 2:
        raise ValueError("proba must be 1D or 2D array")
    if n_classes is not None and p.shape[1] != n_classes:
        raise ValueError("proba second dimension must equal n_classes")
    # Normalize rows to sum to 1 (tolerant to tiny drift)
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return p / row_sum


def _to_labels_from_scores(y_or_scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(y_or_scores)
    if arr.ndim == 2:
        return arr.argmax(axis=1)
    return arr


def accuracy(y_true: np.ndarray, y_pred_or_proba: np.ndarray) -> float:
    """Accuracy from labels or probabilities (argmax).

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred_or_proba : ndarray
        If 1D, treated as predicted labels. If 2D, treated as class probabilities,
        and argmax is used.
    """

    y_true = np.asarray(y_true)
    y_pred = _to_labels_from_scores(y_pred_or_proba)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and predictions must have the same length")
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred_or_proba: np.ndarray) -> float:
    """Macro-averaged F1 from labels or probabilities (argmax)."""

    y_true = np.asarray(y_true)
    y_pred = _to_labels_from_scores(y_pred_or_proba)
    return float(f1_score(y_true, y_pred, average="macro"))


def auroc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    """One-vs-rest AUROC (macro-average).

    Requires class probabilities of shape (n_samples, n_classes) or (n_samples,)
    for binary (interpreted as positive-class probabilities).
    """

    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    n_classes = classes.size
    p = _ensure_proba_2d(np.asarray(proba), n_classes if n_classes > 2 else None)
    if n_classes == 2 and p.shape[1] == 2:
        # sklearn can handle labels directly with multi_class='ovr'
        return float(roc_auc_score(y_true, p[:, 1], average="macro"))
    return float(roc_auc_score(y_true, p, average="macro", multi_class="ovr"))


def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) using confidence binning.

    Strategy: uniform bins in [0, 1]. For each sample, take confidence as
    max probability and correctness based on argmax vs label.
    ECE = sum_b (n_b/N) * |acc_b - conf_b|, ignoring empty bins.
    """

    y_true = np.asarray(y_true)
    p = _ensure_proba_2d(np.asarray(proba))
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y_true).astype(float)

    # Bin edges uniform [0,1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    N = y_true.shape[0]
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge only on last bin
        if i < n_bins - 1:
            in_bin = (conf >= lo) & (conf < hi)
        else:
            in_bin = (conf >= lo) & (conf <= hi)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        acc_b = float(np.mean(correct[in_bin]))
        conf_b = float(np.mean(conf[in_bin]))
        ece_val += (count / N) * abs(acc_b - conf_b)
    return float(ece_val)


def brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score (lower is better).

    Mean squared error between one-hot labels and predicted probabilities.
    Accepts binary probabilities as shape (n_samples,) or (n_samples, 2).
    """

    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    n_classes = classes.size
    p = _ensure_proba_2d(np.asarray(proba), n_classes if n_classes > 2 else None)
    # Build one-hot
    one_hot = np.zeros((y_true.shape[0], p.shape[1]), dtype=float)
    one_hot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((one_hot - p) ** 2, axis=1)))


__all__ = ["accuracy", "macro_f1", "auroc_ovr", "ece", "brier"]
