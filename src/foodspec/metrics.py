"""Metrics and evaluation utilities for FoodSpec."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics as skm


__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_roc_curve",
    "compute_pr_curve",
    "compute_embedding_silhouette",
    "compute_between_within_ratio",
    "compute_between_within_stats",
    "bootstrap_metric_ci",
]


def compute_classification_metrics(
    y_true,
    y_pred,
    labels: Optional[Sequence] = None,
    average: str = "macro",
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray | float]:
    """
    Compute core classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : sequence, optional
        Class label order for confusion matrix.
    average : str, optional
        Averaging for precision/recall/F1 ('macro', 'micro', 'weighted'), by default 'macro'.
    y_scores : array-like, optional
        Probabilities or decision scores (binary) for ROC/PR.

    Returns
    -------
    dict
        accuracy, precision, recall, specificity, f1, balanced_accuracy,
        confusion_matrix, per_class metrics, optional roc/pr curves.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = _binary_counts(cm)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    per_class_precision = skm.precision_score(y_true, y_pred, labels=labels, average=None, zero_division=np.nan)
    per_class_recall = skm.recall_score(y_true, y_pred, labels=labels, average=None, zero_division=np.nan)
    per_class_f1 = skm.f1_score(y_true, y_pred, labels=labels, average=None, zero_division=np.nan)
    res: Dict[str, np.ndarray | float] = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(skm.precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, average=average, zero_division=0)),
        "specificity": specificity,
        "f1": float(skm.f1_score(y_true, y_pred, average=average, zero_division=0)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "support": cm.sum(axis=1),
    }
    if y_scores is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = skm.roc_curve(y_true, y_scores, pos_label=np.unique(y_true)[1])
        prec, rec, _ = skm.precision_recall_curve(y_true, y_scores, pos_label=np.unique(y_true)[1])
        res["roc_curve"] = (fpr, tpr)
        res["pr_curve"] = (prec, rec)
        res["auc"] = float(skm.auc(fpr, tpr))
    return res


def _binary_counts(cm: np.ndarray) -> Tuple[int, int, int, int]:
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    return tn, fp, fn, tp


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float | np.ndarray]:
    """
    Compute regression metrics: RMSE, MAE, R2, MAPE, residuals.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = float(skm.r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs(residuals / np.clip(y_true, 1e-12, None))) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "residuals": residuals}


def compute_roc_curve(y_true, y_scores):
    """
    Convenience wrapper around sklearn roc_curve for binary tasks.
    """

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_scores, pos_label=np.unique(y_true)[1])
    auc_val = float(skm.auc(fpr, tpr))
    return fpr, tpr, thresholds, auc_val


def compute_pr_curve(y_true, y_scores):
    """
    Convenience wrapper around sklearn precision_recall_curve for binary tasks.
    """

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    prec, rec, thresholds = skm.precision_recall_curve(y_true, y_scores, pos_label=np.unique(y_true)[1])
    return prec, rec, thresholds


def compute_embedding_silhouette(scores: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """Compute the silhouette score on an embedding (e.g., PCA or t-SNE scores).

    Parameters
    ----------
    scores : np.ndarray
        Embedded coordinates, shape (n_samples, n_components).
    labels : np.ndarray
        Class labels, shape (n_samples,).
    metric : str
        Distance metric for silhouette (default: euclidean).

    Returns
    -------
    float
        Silhouette score in [-1, 1]; higher means better separation.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    return skm.silhouette_score(scores, labels, metric=metric)


def compute_between_within_ratio(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute a between/within scatter ratio for an embedding.

    Ratio = mean distance between class centroids / mean distance to class centroid (within-class spread).
    Higher values suggest better separation.

    Parameters
    ----------
    scores : np.ndarray
        Embedded coordinates (n_samples, n_components).
    labels : np.ndarray
        Class labels (n_samples,).

    Returns
    -------
    float
        Between/within ratio (unitless). np.nan if only one class.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size < 2:
        return np.nan
    centroids = {c: scores[labels == c].mean(axis=0) for c in unique}
    centroid_list = list(centroids.values())
    between = []
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            between.append(np.linalg.norm(centroid_list[i] - centroid_list[j]))
    mean_between = float(np.mean(between))
    within = []
    for c in unique:
        diffs = scores[labels == c] - centroids[c]
        within.append(np.linalg.norm(diffs, axis=1).mean())
    mean_within = float(np.mean(within))
    if mean_within == 0:
        return np.inf
    return mean_between / mean_within


def compute_between_within_stats(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 0,
    random_state: int | None = None,
) -> Dict[str, float]:
    """Between/within ratio with an F-like statistic and optional permutation p-value.

    This is a descriptive analogue of an ANOVA F-statistic on an embedding (e.g., PCA
    scores). Use it alongside the silhouette score to quantify visible clustering in
    PCA/t-SNE plots.

    Parameters
    ----------
    scores : np.ndarray
        Embedded coordinates (n_samples, n_components).
    labels : np.ndarray
        Class labels (n_samples,).
    n_permutations : int, optional
        If > 0, perform a permutation test on labels to estimate p_perm. Default 0.
    random_state : int, optional
        Seed for reproducibility of the permutation test.

    Returns
    -------
    dict
        Keys: ``ratio`` (between/within), ``f_stat`` (same value), ``p_perm`` (np.nan if
        no permutations were run).
    """

    rng = np.random.default_rng(random_state)
    ratio = compute_between_within_ratio(scores, labels)
    f_stat = ratio
    p_perm = np.nan
    if n_permutations > 0:
        labels_arr = np.asarray(labels)
        perm_ratios = []
        for _ in range(n_permutations):
            perm = rng.permutation(labels_arr)
            perm_ratios.append(compute_between_within_ratio(scores, perm))
        perm_ratios = np.asarray(perm_ratios)
        p_perm = (np.sum(perm_ratios >= ratio) + 1) / (n_permutations + 1)
    return {"ratio": ratio, "f_stat": f_stat, "p_perm": p_perm}


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Dict[str, float | np.ndarray]:
    """Bootstrap confidence interval for an arbitrary metric.

    Parameters
    ----------
    y_true, y_pred
        Arrays of true and predicted values/labels.
    metric_fn
        Callable metric function ``metric_fn(y_true, y_pred) -> float``.
    n_bootstrap
        Number of bootstrap resamples.
    alpha
        Significance level (e.g., 0.05 for 95% CI).
    random_state
        Seed for reproducibility.

    Returns
    -------
    dict
        ``{"metric": observed, "ci_low": low, "ci_high": high, "samples": dist}``
        where ``samples`` is the bootstrap distribution (np.ndarray).
    """

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    dist = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        dist.append(metric_fn(y_true[idx], y_pred[idx]))
    dist = np.asarray(dist)
    lo = float(np.quantile(dist, alpha / 2))
    hi = float(np.quantile(dist, 1 - alpha / 2))
    observed = float(metric_fn(y_true, y_pred))
    return {"metric": observed, "ci_low": lo, "ci_high": hi, "samples": dist}
