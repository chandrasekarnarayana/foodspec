from __future__ import annotations
"""
Reliability and calibration evaluation utilities.

Provides metrics for assessing whether predicted probabilities are well-calibrated
(i.e., predicted confidence matches empirical accuracy).
"""


from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


def top_class_confidence(proba: np.ndarray) -> np.ndarray:
    """Extract maximum predicted probability per sample.

    Parameters
    ----------
    proba : ndarray, shape (n_samples, n_classes)
        Predicted class probabilities.

    Returns
    -------
    ndarray, shape (n_samples,)
        Maximum probability for each sample.

    Examples
    --------
    >>> proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> top_class_confidence(proba)
    array([0.7, 0.6])
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be 2D (n_samples, n_classes)")
    return np.max(proba, axis=1)


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Lower is better. Range [0, 1] for binary classification, [0, 2] for multiclass.

    Formula: mean of (proba[i, y_true[i]] - 1)² + sum(proba[i, j]² for j ≠ y_true[i])

    Or equivalently: mean(sum(proba[i]² for all j) - 2 * proba[i, y_true[i]] + 1)

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True class labels (0-indexed).
    proba : ndarray, shape (n_samples, n_classes)
        Predicted class probabilities, sum to 1 for each sample.

    Returns
    -------
    float
        Brier score (lower is better).

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.25, 0.75]])
    >>> brier = brier_score(y_true, proba)
    >>> np.isclose(brier, 0.014375)
    True
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if proba.ndim != 2:
        raise ValueError("proba must be 2D (n_samples, n_classes)")
    if y_true.shape[0] != proba.shape[0]:
        raise ValueError("y_true and proba must have same number of samples")

    n_samples = y_true.shape[0]
    n_classes = proba.shape[1]

    # Validate labels
    y_true_int = y_true.astype(int)
    if (y_true_int < 0).any() or (y_true_int >= n_classes).any():
        raise ValueError("y_true labels out of range for number of classes")

    # One-hot encode true labels
    y_one_hot = np.zeros_like(proba)
    y_one_hot[np.arange(n_samples), y_true_int] = 1

    # Brier score: mean((proba - y_one_hot)^2)
    return float(np.mean((proba - y_one_hot) ** 2))


def reliability_curve_data(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability curve data for calibration visualization.

    Bins samples by predicted confidence, then computes mean confidence,
    mean accuracy, and sample count in each bin.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True class labels (0-indexed).
    proba : ndarray, shape (n_samples, n_classes)
        Predicted class probabilities.
    n_bins : int, default 15
        Number of bins for confidence interval.
    strategy : {"uniform", "quantile"}, default "uniform"
        - "uniform": equal-width bins [0, 1/n_bins, 2/n_bins, ..., 1]
        - "quantile": equal-frequency bins based on confidence quantiles

    Returns
    -------
    bin_centers : ndarray, shape (n_bins,)
        Mean predicted confidence in each bin (or bin midpoint if empty).
    accuracies : ndarray, shape (n_bins,)
        Fraction of correct predictions in each bin (0 if empty).
    confidences : ndarray, shape (n_bins,)
        Mean predicted confidence in each bin (same as bin_centers).
    counts : ndarray, shape (n_bins,)
        Number of samples in each bin.

    Examples
    --------
    >>> y_true = [0, 0, 0, 1, 1, 1]
    >>> proba = np.array([
    ...     [0.8, 0.2],
    ...     [0.9, 0.1],
    ...     [0.7, 0.3],
    ...     [0.1, 0.9],
    ...     [0.2, 0.8],
    ...     [0.3, 0.7],
    ... ])
    >>> centers, accs, confs, counts = reliability_curve_data(y_true, proba, n_bins=2)
    >>> counts  # doctest: +SKIP
    array([3, 3])
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if proba.ndim != 2:
        raise ValueError("proba must be 2D (n_samples, n_classes)")
    if y_true.shape[0] != proba.shape[0]:
        raise ValueError("y_true and proba must have same number of samples")

    n_samples = y_true.shape[0]
    n_classes = proba.shape[1]

    # Validate labels
    y_true_int = y_true.astype(int)
    if (y_true_int < 0).any() or (y_true_int >= n_classes).any():
        raise ValueError("y_true labels out of range for number of classes")

    # Get predicted confidence (top class probability)
    confidence = top_class_confidence(proba)

    # Get predictions and correctness
    predictions = np.argmax(proba, axis=1)
    correct = predictions == y_true_int

    # Define bin edges
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        # Use quantiles of observed confidence to get equal-frequency bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(confidence, quantiles)
        # Ensure strictly increasing edges (handle duplicates from constant confidence)
        bin_edges = np.unique(bin_edges)
        n_bins_actual = len(bin_edges) - 1
        if n_bins_actual < n_bins:
            # Fallback to uniform if quantiles don't give enough unique bins
            bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Digitize: assign each sample to a bin
    bin_indices = np.digitize(confidence, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

    # Compute statistics per bin
    n_bins_actual = len(bin_edges) - 1
    bin_centers = np.zeros(n_bins_actual)
    accuracies = np.zeros(n_bins_actual)
    confidences_per_bin = np.zeros(n_bins_actual)
    counts = np.zeros(n_bins_actual, dtype=int)

    for i in range(n_bins_actual):
        mask = bin_indices == i
        counts[i] = int(mask.sum())

        if counts[i] > 0:
            # Mean confidence in bin
            confidences_per_bin[i] = float(np.mean(confidence[mask]))
            bin_centers[i] = confidences_per_bin[i]
            # Accuracy in bin
            accuracies[i] = float(np.mean(correct[mask]))
        else:
            # Empty bin: use bin midpoint for center
            bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            confidences_per_bin[i] = bin_centers[i]
            accuracies[i] = 0.0

    return bin_centers, accuracies, confidences_per_bin, counts


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the average absolute difference between predicted confidence
    and empirical accuracy across bins. Lower is better (0 = perfectly calibrated).

    ECE = sum_i (n_i / n) * |accuracy_i - confidence_i|

    where i ranges over bins, n_i is sample count in bin i, n is total samples.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True class labels (0-indexed).
    proba : ndarray, shape (n_samples, n_classes)
        Predicted class probabilities.
    n_bins : int, default 15
        Number of bins for calibration computation.
    strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy for confidence intervals.

    Returns
    -------
    float
        Expected Calibration Error (range [0, 1]).

    Examples
    --------
    Perfect calibration should give ECE ≈ 0:

    >>> y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    >>> proba = np.array([
    ...     [0.9, 0.1],  # Correct, high confidence
    ...     [0.1, 0.9],  # Correct, high confidence
    ...     [0.9, 0.1],  # Correct, high confidence
    ...     [0.1, 0.9],  # Correct, high confidence
    ...     [0.9, 0.1],  # Correct, high confidence
    ...     [0.1, 0.9],  # Correct, high confidence
    ...     [0.9, 0.1],  # Correct, high confidence
    ...     [0.1, 0.9],  # Correct, high confidence
    ...     [0.9, 0.1],  # Correct, high confidence
    ...     [0.1, 0.9],  # Correct, high confidence
    ... ])
    >>> ece = expected_calibration_error(y_true, proba)
    >>> ece < 0.01  # Should be very close to 0
    True
    """
    bin_centers, accuracies, confidences, counts = reliability_curve_data(
        y_true, proba, n_bins=n_bins, strategy=strategy
    )

    n_samples = len(y_true)
    weights = counts / n_samples

    # ECE: weighted average of absolute differences
    ece = float(np.sum(weights * np.abs(accuracies - confidences)))

    return ece


@dataclass
class CalibrationMetrics:
    """Summary of calibration quality metrics.

    Attributes
    ----------
    ece : float
        Expected Calibration Error (0 = perfect, higher = worse calibration).
    brier : float
        Brier score (0 = perfect, higher = worse).
    bin_centers : ndarray
        Mean predicted confidence per bin.
    accuracies : ndarray
        Empirical accuracy per bin.
    counts : ndarray
        Sample counts per bin.
    """

    ece: float
    brier: float
    bin_centers: np.ndarray
    accuracies: np.ndarray
    counts: np.ndarray


def compute_calibration_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> CalibrationMetrics:
    """Compute comprehensive calibration metrics.

    Combines ECE, Brier score, and reliability curve data into single result.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True class labels (0-indexed).
    proba : ndarray, shape (n_samples, n_classes)
        Predicted class probabilities.
    n_bins : int, default 15
        Number of bins.
    strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy.

    Returns
    -------
    CalibrationMetrics
        Object with ece, brier, and binning data.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])
    >>> metrics = compute_calibration_metrics(y_true, proba, n_bins=2)
    >>> 0 <= metrics.ece <= 1
    True
    >>> 0 <= metrics.brier <= 2
    True
    """
    ece = expected_calibration_error(y_true, proba, n_bins=n_bins, strategy=strategy)
    brier = brier_score(y_true, proba)
    bin_centers, accuracies, _, counts = reliability_curve_data(
        y_true, proba, n_bins=n_bins, strategy=strategy
    )

    return CalibrationMetrics(
        ece=ece,
        brier=brier,
        bin_centers=bin_centers,
        accuracies=accuracies,
        counts=counts,
    )


__all__ = [
    "top_class_confidence",
    "brier_score",
    "reliability_curve_data",
    "expected_calibration_error",
    "compute_calibration_metrics",
    "CalibrationMetrics",
]
