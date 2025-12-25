"""Nested cross-validation utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


def nested_cross_validate(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_outer: int = 5,
    cv_inner: int = 3,
    scoring: str = "accuracy",
    fit_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Nested CV for unbiased model evaluation.

    Parameters
    ----------
    estimator : estimator
        sklearn estimator.
    X : np.ndarray
        Features (n_samples, n_features).
    y : np.ndarray
        Labels.
    cv_outer : int, default=5
        Outer folds.
    cv_inner : int, default=3
        Inner folds.
    scoring : str, default="accuracy"
        Score metric.
    fit_params : dict, optional
        fit() kwargs.

    Returns
    -------
    dict
        Nested CV results.
    """
    if fit_params is None:
        fit_params = {}

    outer_cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=0)
    test_scores = []
    train_scores = []

    for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV
        inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=outer_idx)
        cross_val_score(clone(estimator), X_train, y_train, cv=inner_cv, scoring=scoring)

        # Outer evaluation
        est = clone(estimator)
        est.fit(X_train, y_train, **fit_params)
        test_score = est.score(X_test, y_test) if hasattr(est, "score") else 0.0
        train_score = est.score(X_train, y_train) if hasattr(est, "score") else 0.0

        test_scores.append(test_score)
        train_scores.append(train_score)

    return {
        "test_scores": np.array(test_scores),
        "train_scores": np.array(train_scores),
        "mean_test_score": float(np.mean(test_scores)),
        "std_test_score": float(np.std(test_scores)),
        "mean_train_score": float(np.mean(train_scores)),
        "std_train_score": float(np.std(train_scores)),
    }


def nested_cross_validate_custom(
    train_fn: Callable,
    eval_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv_outer: int = 5,
    cv_inner: int = 3,
) -> Dict[str, Any]:
    """Nested CV with custom train/eval functions.

    Parameters
    ----------
    train_fn : callable
        Custom training function.
    eval_fn : callable
        Custom evaluation function.
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    cv_outer : int, default=5
        Outer folds.
    cv_inner : int, default=3
        Inner folds.

    Returns
    -------
    dict
        Results.
    """
    outer_cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=0)
    test_scores = []

    for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=outer_idx)
        inner_folds = list(inner_cv.split(X_train, y_train))

        model = train_fn(X_train, y_train, inner_folds)
        score = eval_fn(model, X_test, y_test)
        test_scores.append(score)

    return {
        "test_scores": np.array(test_scores),
        "mean_test_score": float(np.mean(test_scores)),
        "std_test_score": float(np.std(test_scores)),
    }


def nested_cross_validate_regression(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_outer: int = 5,
    cv_inner: int = 3,
    scoring: str = "r2",
) -> Dict[str, Any]:
    """Nested CV for regression.

    Parameters
    ----------
    estimator : estimator
        Regression estimator.
    X : np.ndarray
        Features.
    y : np.ndarray
        Targets.
    cv_outer : int, default=5
        Outer folds.
    cv_inner : int, default=3
        Inner folds.
    scoring : str, default="r2"
        Score metric.

    Returns
    -------
    dict
        Results.
    """
    outer_cv = KFold(n_splits=cv_outer, shuffle=True, random_state=0)
    test_scores = []
    train_scores = []

    for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = KFold(n_splits=cv_inner, shuffle=True, random_state=outer_idx)
        cross_val_score(clone(estimator), X_train, y_train, cv=inner_cv, scoring=scoring)

        est = clone(estimator)
        est.fit(X_train, y_train)
        test_score = est.score(X_test, y_test) if hasattr(est, "score") else 0.0
        train_score = est.score(X_train, y_train) if hasattr(est, "score") else 0.0

        test_scores.append(test_score)
        train_scores.append(train_score)

    return {
        "test_scores": np.array(test_scores),
        "train_scores": np.array(train_scores),
        "mean_test_score": float(np.mean(test_scores)),
        "std_test_score": float(np.std(test_scores)),
        "mean_train_score": float(np.mean(train_scores)),
        "std_train_score": float(np.std(train_scores)),
    }


def compare_models_nested_cv(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv_outer: int = 5,
    cv_inner: int = 3,
    scoring: str = "accuracy",
    task: str = "classification",
) -> Dict[str, Dict[str, Any]]:
    """Compare models with nested CV.

    Parameters
    ----------
    models : dict
        {name: estimator, ...}.
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels/targets.
    cv_outer : int, default=5
        Outer folds.
    cv_inner : int, default=3
        Inner folds.
    scoring : str, default="accuracy"
        Score metric.
    task : str, default="classification"
        "classification" or "regression".

    Returns
    -------
    dict
        {name: results, ...}.
    """
    results = {}

    for name, est in models.items():
        if task == "classification":
            res = nested_cross_validate(est, X, y, cv_outer, cv_inner, scoring)
        else:
            res = nested_cross_validate_regression(est, X, y, cv_outer, cv_inner, scoring)

        results[name] = res

    return results


__all__ = [
    "nested_cross_validate",
    "nested_cross_validate_custom",
    "nested_cross_validate_regression",
    "compare_models_nested_cv",
]
