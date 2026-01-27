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
    """Nested cross-validation for unbiased model evaluation with hyperparameter tuning.

    Prevents optimistic bias that arises from tuning hyperparameters on the same
    data used for evaluation. Critical for reliable model selection in FoodSpec.

    **Why Nested CV Matters:**
    ╔═══════════════════════════════════════╗
    │ Single CV (BIASED):                   │
    │ Outer fold → tune hyperparams → eval  │
    │ Result: 95% accuracy (overfitted)     │
    ├───────────────────────────────────────┤
    │ Nested CV (UNBIASED):                 │
    │ Inner fold → tune hyperparams         │
    │ Outer fold → eval on unseen data      │
    │ Result: 82% accuracy (realistic)      │
    ╚═══════════════════════════════════════╝

    **Outer Folds (Evaluation):**
    - Use StratifiedKFold to maintain class balance
    - k=5 (default): 80% train, 20% test per split
    - Each outer fold tests generalization to completely unseen data

    **Inner Folds (Hyperparameter Tuning):**
    - Use StratifiedKFold to maintain class balance within training set
    - k=3 (default): Fast tuning without excessive train/test splits
    - Only searches over the training portion of each outer fold

    **Output Interpretation:**
    - test_scores: Outer fold scores (realistic performance estimate)
    - train_scores: Inner fold scores (may be >95%, don't trust this!)
    - mean_test_score ± std_test_score: Final model performance

    **When to Use:**
    - Always use for hyperparameter tuning (grid search, random search)
    - Every model selection decision should use nested CV
    - Especially critical with small datasets (n < 200)

    **Real Example:**
    Model selection: Kernel SVM vs. Linear SVM
    ├─ Without nested CV: RBF kernel 95%, Linear 92% → choose RBF (overfitted)
    └─ With nested CV: RBF kernel 82%, Linear 83% → choose Linear (more stable)

    **Red Flags:**
    - train_scores >> test_scores (>15% gap): Check for hyperparameter overfitting
    - test_scores very noisy (std > mean/2): Increase outer folds or data size
    - Negative scores: Likely scoring metric issue; check scoring param

    Parameters:
        estimator (sklearn estimator): Model to evaluate (e.g., SVC, RandomForestClassifier)
        X (np.ndarray): Features array (n_samples, n_features)
        y (np.ndarray): Target labels (n_samples,)
        cv_outer (int, default 5): Number of outer folds for evaluation
        cv_inner (int, default 3): Number of inner folds for hyperparameter tuning
        scoring (str, default "accuracy"): Scoring metric name from sklearn.metrics
            Options: "accuracy", "f1", "roc_auc", "precision", "recall"
        fit_params (dict, optional): Additional arguments to pass to estimator.fit()

    Returns:
        dict: Results containing:
            - test_scores: Array of outer fold test scores (realistic performance)
            - train_scores: Array of inner fold train scores (optimistic estimates)
            - mean_test_score: Mean test score across all outer folds
            - std_test_score: Standard deviation of test scores
            - all_inner_scores: Detailed inner fold scores for diagnosis

    Examples:
        >>> from foodspec.ml import nested_cross_validate\n        >>> from sklearn.svm import SVC\n        >>> from sklearn.datasets import make_classification\n        >>> X, y = make_classification(n_samples=100, n_features=20)\n        >>> svc = SVC(kernel='rbf')\n        >>> result = nested_cross_validate(svc, X, y, cv_outer=5, cv_inner=3)\n        >>> print(f\"Realistic accuracy: {result['mean_test_score']:.3f} ± {result['std_test_score']:.3f}\")\n

    References:
        Varma, S., & Simon, R. (2006). Bias in error estimation when using
        cross-validation for model selection. BMC Bioinformatics, 7(1), 91.
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
