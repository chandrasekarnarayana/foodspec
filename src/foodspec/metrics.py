"""
Metrics and evaluation utilities for FoodSpec.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics as skm


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
