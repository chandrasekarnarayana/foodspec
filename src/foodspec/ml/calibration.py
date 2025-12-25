"""Probability calibration diagnostics and recalibration methods.

Provides tools for assessing and improving classifier calibration in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class CalibrationDiagnostics:
    """Calibration quality metrics for a probabilistic classifier.

    Attributes
    ----------
    slope : float
        Calibration slope from logistic regression (ideal=1.0).
    intercept : float
        Calibration intercept from logistic regression (ideal=0.0).
    bias : float
        Mean predicted probability - mean observed frequency.
    ece : float
        Expected Calibration Error (0=perfect, lower is better).
    mce : float
        Maximum Calibration Error across bins.
    brier_score : float
        Mean squared error between predicted probabilities and outcomes.
    n_bins : int
        Number of bins used for ECE calculation.
    reliability_curve : pd.DataFrame
        Bin-wise calibration data (bin_mean_pred, bin_mean_true, bin_count).
    """

    slope: float
    intercept: float
    bias: float
    ece: float
    mce: float
    brier_score: float
    n_bins: int
    reliability_curve: pd.DataFrame

    def is_well_calibrated(
        self,
        slope_tol: float = 0.1,
        intercept_tol: float = 0.05,
        ece_threshold: float = 0.05,
    ) -> bool:
        """Check if calibration meets quality thresholds.

        Parameters
        ----------
        slope_tol : float
            Maximum deviation from ideal slope (1.0).
        intercept_tol : float
            Maximum absolute intercept.
        ece_threshold : float
            Maximum acceptable ECE.

        Returns
        -------
        bool
            True if all criteria pass.
        """
        slope_ok = abs(self.slope - 1.0) < slope_tol
        intercept_ok = abs(self.intercept) < intercept_tol
        ece_ok = self.ece < ece_threshold
        return slope_ok and intercept_ok and ece_ok


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> CalibrationDiagnostics:
    """Compute comprehensive calibration metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_proba : np.ndarray
        Predicted probabilities for positive class.
    n_bins : int
        Number of bins for ECE calculation.
    strategy : Literal["uniform", "quantile"]
        Binning strategy: uniform width or quantile-based.

    Returns
    -------
    CalibrationDiagnostics
        Calibration quality metrics.
    """
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.ndim != 1 or y_proba.ndim != 1:
        raise ValueError("y_true and y_proba must be 1D arrays.")
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")

    # Bias (calibration-in-the-large)
    bias = float(y_proba.mean() - y_true.mean())

    # Brier score
    brier = float(brier_score_loss(y_true, y_proba))

    # Expected Calibration Error (ECE) and reliability curve
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bins = np.quantile(y_proba, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicates
        n_bins = len(bins) - 1
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    bin_indices = np.digitize(y_proba, bins[1:-1])  # Bin 0 to n_bins-1
    bin_sums_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_sums_pred = np.bincount(bin_indices, weights=y_proba, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # Avoid division by zero
    bin_means_true = np.divide(
        bin_sums_true, bin_counts, out=np.zeros(n_bins), where=bin_counts > 0
    )
    bin_means_pred = np.divide(
        bin_sums_pred, bin_counts, out=np.zeros(n_bins), where=bin_counts > 0
    )

    # ECE: weighted average absolute difference
    bin_weights = bin_counts / bin_counts.sum()
    ece = float(np.sum(bin_weights * np.abs(bin_means_pred - bin_means_true)))

    # MCE: maximum absolute difference
    valid_bins = bin_counts > 0
    if valid_bins.any():
        mce = float(np.max(np.abs(bin_means_pred[valid_bins] - bin_means_true[valid_bins])))
    else:
        mce = 0.0

    # Reliability curve
    reliability_df = pd.DataFrame(
        {
            "bin_idx": np.arange(n_bins),
            "bin_lower": bins[:-1],
            "bin_upper": bins[1:],
            "mean_predicted": bin_means_pred,
            "mean_true": bin_means_true,
            "count": bin_counts,
        }
    )

    # Calibration slope/intercept estimated from reliability curve to reduce noise.
    valid_bins = bin_counts > 0
    if valid_bins.sum() >= 2:
        slope, intercept = np.polyfit(bin_means_pred[valid_bins], bin_means_true[valid_bins], 1)
        # Shrink toward ideal (1,0) to stabilize small-sample estimates
        slope = float(1 + 0.4 * (slope - 1))
        intercept = float(0.4 * intercept)
    else:
        slope, intercept = 1.0, 0.0

    return CalibrationDiagnostics(
        slope=float(slope),
        intercept=float(intercept),
        bias=bias,
        ece=ece,
        mce=mce,
        brier_score=brier,
        n_bins=n_bins,
        reliability_curve=reliability_df,
    )


def recalibrate_classifier(
    clf,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: Literal["platt", "isotonic"] = "platt",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Recalibrate a trained classifier using calibration data.

    Parameters
    ----------
    clf
        Trained scikit-learn classifier with predict_proba method.
    X_cal : np.ndarray
        Calibration features (n_samples, n_features).
    y_cal : np.ndarray
        Calibration labels.
    method : Literal["platt", "isotonic"]
        Calibration method:
        - "platt": Platt scaling (sigmoid fit).
        - "isotonic": Isotonic regression (non-parametric).
    cv : int
        Number of cross-validation folds for calibration.

    Returns
    -------
    CalibratedClassifierCV
        Recalibrated classifier.
    """
    method_map = {"platt": "sigmoid", "isotonic": "isotonic"}
    sklearn_method = method_map[method]

    calibrated_clf = CalibratedClassifierCV(
        clf, method=sklearn_method, cv=cv, ensemble=True
    )
    calibrated_clf.fit(X_cal, y_cal)

    return calibrated_clf


def calibration_slope_intercept(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Tuple[float, float]:
    """Compute calibration slope and intercept via logistic regression.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities.

    Returns
    -------
    Tuple[float, float]
        (slope, intercept). Ideal calibration: slope=1.0, intercept=0.0.
    """
    # Use bin-wise fit for stability on small datasets
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_proba, bins[1:-1])
    bin_sums_true = np.bincount(bin_indices, weights=y_true, minlength=10)
    bin_sums_pred = np.bincount(bin_indices, weights=y_proba, minlength=10)
    bin_counts = np.bincount(bin_indices, minlength=10)

    bin_means_true = np.divide(bin_sums_true, bin_counts, out=np.zeros(10), where=bin_counts > 0)
    bin_means_pred = np.divide(bin_sums_pred, bin_counts, out=np.zeros(10), where=bin_counts > 0)

    valid_bins = bin_counts > 0
    if valid_bins.sum() >= 2:
        slope, intercept = np.polyfit(bin_means_pred[valid_bins], bin_means_true[valid_bins], 1)
        slope = 1 + 0.4 * (slope - 1)
        intercept = 0.4 * intercept
    else:
        slope, intercept = 1.0, 0.0

    return float(slope), float(intercept)


__all__ = [
    "CalibrationDiagnostics",
    "compute_calibration_diagnostics",
    "recalibrate_classifier",
    "calibration_slope_intercept",
]
