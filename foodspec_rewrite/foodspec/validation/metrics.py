"""
Classification and calibration metrics for spectroscopy validation.

This module provides robust metrics that handle common edge cases in
spectroscopy validation workflows, particularly Leave-One-Batch-Out (LOBO)
cross-validation where some classes may be missing in certain folds.

Functions
---------
accuracy
    Classification accuracy.
macro_f1
    Macro-averaged F1 score.
precision_macro
    Macro-averaged precision.
recall_macro
    Macro-averaged recall.
auroc_macro
    Macro-averaged AUROC (one-vs-rest) with safe fallback.
expected_calibration_error
    Expected Calibration Error (ECE) with configurable bins.
compute_classification_metrics
    Compute all classification metrics at once.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities (not used for accuracy).

    Returns
    -------
    dict[str, float]
        Dictionary with 'accuracy' key.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 0])
    >>> accuracy(y_true, y_pred)
    {'accuracy': 0.75}
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D array")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be 1D array")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": float(acc)}


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute macro-averaged F1 score.

    Macro-averaging computes F1 for each class independently and averages them,
    giving equal weight to each class regardless of support.

    Handles missing classes gracefully by computing F1 only for classes present
    in y_true. This is crucial for Leave-One-Batch-Out (LOBO) validation.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities (not used for F1).

    Returns
    -------
    dict[str, float]
        Dictionary with 'macro_f1' key.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 0])
    >>> macro_f1(y_true, y_pred)
    {'macro_f1': 0.8333...}

    Notes
    -----
    Uses zero_division=0 to handle cases where a class has no predictions.
    This is common in imbalanced datasets or LOBO validation.
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D array")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be 1D array")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Get classes present in y_true
    labels = np.unique(y_true)

    # Compute F1 with zero_division=0 for robustness
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"macro_f1": float(f1)}


def precision_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute macro-averaged precision.

    Macro-averaging computes precision for each class independently and averages
    them, giving equal weight to each class regardless of support.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities (not used for precision).

    Returns
    -------
    dict[str, float]
        Dictionary with 'precision_macro' key.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 0])
    >>> precision_macro(y_true, y_pred)
    {'precision_macro': 0.75}

    Notes
    -----
    Uses zero_division=0 to handle cases where a class has no predictions.
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D array")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be 1D array")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    labels = np.unique(y_true)
    prec = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {"precision_macro": float(prec)}


def recall_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute macro-averaged recall.

    Macro-averaging computes recall for each class independently and averages
    them, giving equal weight to each class regardless of support.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities (not used for recall).

    Returns
    -------
    dict[str, float]
        Dictionary with 'recall_macro' key.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 0])
    >>> recall_macro(y_true, y_pred)
    {'recall_macro': 0.75}

    Notes
    -----
    Uses zero_division=0 to handle cases where a class has no true samples.
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D array")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be 1D array")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    labels = np.unique(y_true)
    rec = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {"recall_macro": float(rec)}


def auroc_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute macro-averaged AUROC (one-vs-rest).

    Computes AUROC for each class in a one-vs-rest fashion and averages them.
    Robust to missing classes in validation fold - only computes AUROC for
    classes present in y_true.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels (not used for AUROC).
    y_proba : np.ndarray, shape (n_samples, n_classes), optional
        Predicted probabilities. Required for AUROC computation.

    Returns
    -------
    dict[str, float]
        Dictionary with 'auroc_macro' key.

    Raises
    ------
    ValueError
        If y_proba is None or has invalid shape.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    >>> auroc_macro(y_true, None, y_proba)
    {'auroc_macro': 1.0}

    Notes
    -----
    - Only classes present in y_true are considered
    - Returns NaN if only one class is present in y_true
    - Uses one-vs-rest strategy for multiclass classification
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D array")

    if y_proba is None:
        raise ValueError(
            "y_proba is required for AUROC computation.\n"
            "Ensure your model supports predict_proba() and pass the probabilities.\n"
            "For models without probability support (e.g., LinearSVC), "
            "consider using SVCClassifier with probability=True or a different model."
        )

    if y_proba.ndim != 2:
        raise ValueError("y_proba must be 2D array with shape (n_samples, n_classes)")

    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have same length")

    # Get classes present in y_true
    classes_in_y = np.unique(y_true)

    # Need at least 2 classes for AUROC
    if len(classes_in_y) < 2:
        return {"auroc_macro": float("nan")}

    # For binary classification
    if len(classes_in_y) == 2 and y_proba.shape[1] == 2:
        try:
            # Use positive class probability
            auroc = roc_auc_score(y_true, y_proba[:, 1])
            return {"auroc_macro": float(auroc)}
        except ValueError:
            return {"auroc_macro": float("nan")}

    # For multiclass, need to compute one-vs-rest manually
    try:
        from sklearn.preprocessing import label_binarize

        # Binarize the labels for one-vs-rest
        # sklearn needs all possible classes, not just those in this fold
        all_classes = np.arange(y_proba.shape[1])
        y_bin = label_binarize(y_true, classes=all_classes)

        # If only some classes are present, y_bin will have columns for all classes
        # but only those in classes_in_y will have non-zero entries

        # Compute AUROC for each class in one-vs-rest manner
        aurocs = []
        for i, class_label in enumerate(all_classes):
            if class_label not in classes_in_y:
                # Skip classes not present in this fold
                continue

            try:
                # One-vs-rest: class i vs all others
                auroc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
                aurocs.append(auroc_i)
            except ValueError:
                # Skip if we can't compute AUROC for this class
                pass

        if len(aurocs) == 0:
            return {"auroc_macro": float("nan")}

        # Macro-average: unweighted mean
        return {"auroc_macro": float(np.mean(aurocs))}

    except Exception:
        return {"auroc_macro": float("nan")}


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual outcomes.
    Perfect calibration means that predictions with confidence p are correct
    p% of the time.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels (not used directly, but kept for API consistency).
    y_proba : np.ndarray, shape (n_samples, n_classes), optional
        Predicted probabilities. Required for ECE computation.
    n_bins : int, default 10
        Number of bins for calibration curve.

    Returns
    -------
    dict[str, float]
        Dictionary with 'ece' key.

    Raises
    ------
    ValueError
        If y_proba is None or n_bins is invalid.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    >>> expected_calibration_error(y_true, None, y_proba)
    {'ece': 0.0}

    Notes
    -----
    - Uses maximum predicted probability (confidence) for binning
    - Bins are equally spaced between 0 and 1
    - Empty bins are ignored in the ECE calculation
    - Lower ECE indicates better calibration (perfect calibration = 0)

    References
    ----------
    Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).
    Obtaining Well Calibrated Probabilities Using Bayesian Binning.
    In AAAI (pp. 2901-2907).
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D array")

    if y_proba is None:
        raise ValueError(
            "y_proba is required for Expected Calibration Error (ECE) computation.\n"
            "Ensure your model supports predict_proba() and pass the probabilities.\n"
            "For models without probability support (e.g., LinearSVC), "
            "consider using:\n"
            "  1. SVCClassifier with probability=True (enables Platt scaling)\n"
            "  2. A different model that naturally supports probabilities\n"
            "  3. CalibratedClassifierCV wrapper from sklearn\n"
            "ECE cannot be computed from hard predictions alone."
        )

    if y_proba.ndim != 2:
        raise ValueError("y_proba must be 2D array with shape (n_samples, n_classes)")

    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have same length")

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    # Get maximum predicted probability (confidence)
    confidences = np.max(y_proba, axis=1)

    # Get predicted labels from probabilities
    predicted_labels = np.argmax(y_proba, axis=1)

    # Check if predictions are correct
    accuracies = predicted_labels == y_true

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        # Skip empty bins
        if not np.any(in_bin):
            continue

        # Proportion of samples in this bin
        prop_in_bin = np.mean(in_bin)

        # Average confidence in this bin
        avg_confidence = np.mean(confidences[in_bin])

        # Average accuracy in this bin
        avg_accuracy = np.mean(accuracies[in_bin])

        # Add weighted absolute difference to ECE
        ece += prop_in_bin * np.abs(avg_confidence - avg_accuracy)

    return {"ece": float(ece)}


def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) - legacy API.

    This is a backward-compatibility wrapper that returns a single float
    instead of a dictionary.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    proba : np.ndarray, shape (n_samples, n_classes)
        Predicted probabilities.
    n_bins : int, default 10
        Number of bins for calibration curve.

    Returns
    -------
    float
        Expected Calibration Error.

    See Also
    --------
    expected_calibration_error : New API that returns dict[str, float].
    """
    pred = proba.argmax(axis=1) if proba.ndim == 2 else (proba > 0.5).astype(int)
    result = expected_calibration_error(y_true, pred, proba, n_bins=n_bins)
    return result["ece"]


def brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Compute multiclass Brier score.

    Brier score measures mean squared error between one-hot labels and
    predicted probabilities. Lower is better.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    proba : np.ndarray, shape (n_samples, n_classes)
        Predicted probabilities.

    Returns
    -------
    float
        Brier score (lower is better).

    Examples
    --------
    >>> y_true = np.array([0, 1])
    >>> proba = np.array([[0.8, 0.2], [0.3, 0.7]])
    >>> brier(y_true, proba)
    0.13

    Notes
    -----
    - Accepts binary probabilities as shape (n_samples,) or (n_samples, 2)
    - For binary case, 1D proba is interpreted as positive class probability
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    # Handle 1D probabilities (binary case)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    if proba.ndim != 2:
        raise ValueError("proba must be 1D or 2D array")

    if len(y_true) != len(proba):
        raise ValueError("y_true and proba must have same length")

    # Normalize probabilities (sum to 1)
    row_sum = proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    proba = proba / row_sum

    # Build one-hot encoding
    n_samples = len(y_true)
    n_classes = proba.shape[1]
    one_hot = np.zeros((n_samples, n_classes), dtype=float)
    one_hot[np.arange(n_samples), y_true.astype(int)] = 1.0

    # Compute mean squared error
    return float(np.mean(np.sum((one_hot - proba) ** 2, axis=1)))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    include_ece: bool = True,
    ece_bins: int = 10,
) -> Dict[str, float]:
    """Compute all classification metrics at once.

    Convenience function that computes accuracy, macro-averaged precision,
    recall, F1, AUROC, and optionally ECE.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True labels.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.
    y_proba : np.ndarray, shape (n_samples, n_classes), optional
        Predicted probabilities. Required for AUROC and ECE.
    include_ece : bool, default True
        Whether to compute ECE (requires y_proba).
    ece_bins : int, default 10
        Number of bins for ECE computation.

    Returns
    -------
    dict[str, float]
        Dictionary containing all computed metrics.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 0])
    >>> y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    >>> metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    >>> sorted(metrics.keys())
    ['accuracy', 'auroc_macro', 'ece', 'macro_f1', 'precision_macro', 'recall_macro']

    Notes
    -----
    - AUROC and ECE require y_proba
    - Missing classes are handled gracefully
    - If y_proba is None, AUROC and ECE will be skipped with warnings
    """
    metrics = {}

    # Always compute these metrics
    metrics.update(accuracy(y_true, y_pred, y_proba))
    metrics.update(precision_macro(y_true, y_pred, y_proba))
    metrics.update(recall_macro(y_true, y_pred, y_proba))
    metrics.update(macro_f1(y_true, y_pred, y_proba))

    # Compute AUROC if probabilities available
    if y_proba is not None:
        try:
            metrics.update(auroc_macro(y_true, y_pred, y_proba))
        except ValueError:
            # If AUROC computation fails, set to NaN
            metrics["auroc_macro"] = float("nan")

        # Compute ECE if requested
        if include_ece:
            try:
                metrics.update(
                    expected_calibration_error(y_true, y_pred, y_proba, n_bins=ece_bins)
                )
            except ValueError:
                metrics["ece"] = float("nan")

    return metrics


__all__ = [
    "accuracy",
    "macro_f1",
    "precision_macro",
    "recall_macro",
    "auroc_macro",
    "expected_calibration_error",
    "compute_classification_metrics",
    "ece",  # Legacy API
    "brier",  # Additional calibration metric
]
