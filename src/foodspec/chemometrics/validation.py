"""Validation utilities for chemometrics models."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate, permutation_test_score

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "cross_validate_pipeline",
    "permutation_test_score_wrapper",
]


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Compute common classification metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    average = "binary" if len(labels) == 2 else "weighted"
    pos_label = labels[0] if len(labels) == 2 else None

    results: dict[str, Any] = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(
            y_true, y_pred, zero_division=0, average=average, pos_label=pos_label
        ),
        "recall": metrics.recall_score(
            y_true, y_pred, zero_division=0, average=average, pos_label=pos_label
        ),
        "f1": metrics.f1_score(
            y_true, y_pred, zero_division=0, average=average, pos_label=pos_label
        ),
    }
    if y_proba is not None and len(labels) == 2:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 2 and y_proba.shape[1] > 1:
            pos_scores = y_proba[:, 1]
        else:
            pos_scores = y_proba
        results["roc_auc"] = metrics.roc_auc_score(y_true, pos_scores)
        results["average_precision"] = metrics.average_precision_score(y_true, pos_scores)

    return pd.DataFrame([results])


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """Compute regression metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return pd.Series({"rmse": rmse, "mae": mae, "r2": r2})


def cross_validate_pipeline(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Cross-validate a pipeline and return fold scores plus summary."""

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        return_train_score=False,
    )
    scores = cv_results["test_score"]
    rows = [{"fold": i + 1, "score": s} for i, s in enumerate(scores)]
    rows.append({"fold": "mean", "score": np.mean(scores)})
    rows.append({"fold": "std", "score": np.std(scores)})
    return pd.DataFrame(rows)


def permutation_test_score_wrapper(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = "accuracy",
    n_permutations: int = 100,
    random_state: Optional[int] = None,
):
    """Wrapper around sklearn's permutation_test_score."""

    score, perm_scores, pvalue = permutation_test_score(
        estimator,
        X,
        y,
        scoring=scoring,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return score, perm_scores, pvalue
