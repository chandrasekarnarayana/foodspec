"""Automated threshold optimization for QC metrics."""
from __future__ import annotations


from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def estimate_threshold_quantile(
    scores: np.ndarray,
    percentile: float = 95,
) -> float:
    """Estimate threshold as percentile of scores."""
    return float(np.percentile(scores, percentile))


def estimate_threshold_youden(
    scores: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Estimate threshold using Youden's J-statistic."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_stats = tpr - fpr
    best_idx = np.argmax(j_stats)
    return float(thresholds[best_idx])


def estimate_threshold_f1(
    scores: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Estimate threshold by maximizing F1 score."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def estimate_threshold_elbow(
    scores: np.ndarray,
    n_clusters: int = 2,
) -> float:
    """Estimate threshold using unsupervised elbow detection."""
    scores = np.asarray(scores, dtype=float)
    sorted_scores = np.sort(scores)

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        km.fit_predict(sorted_scores.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.flatten())
        return float(np.mean(centers[:2]))
    except Exception:
        diffs = np.diff(sorted_scores)
        elbow_idx = np.argmax(diffs) + 1
        return float(sorted_scores[min(elbow_idx, len(sorted_scores) - 1)])


def validate_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute threshold validation metrics."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    y_pred = (scores >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-12)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
    }


__all__ = [
    "estimate_threshold_quantile",
    "estimate_threshold_youden",
    "estimate_threshold_f1",
    "estimate_threshold_elbow",
    "validate_threshold",
]
