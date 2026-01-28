"""Model evaluation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_model_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, float]:
    """Evaluate a model using cross-validation.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible estimator.
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector.
    cv : int
        Number of CV folds.

    Returns
    -------
    dict
        Cross-validation results.
    """
    try:
        from sklearn.model_selection import cross_validate
    except Exception:
        return {"accuracy": 0.0}

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    try:
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        return {
            "mean_test_accuracy": float(np.mean(cv_results["test_accuracy"])),
            "std_test_accuracy": float(np.std(cv_results["test_accuracy"])),
        }
    except Exception:
        return {"accuracy": 0.0}


__all__ = ["evaluate_model_cv"]
