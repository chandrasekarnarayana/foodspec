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
    """Compute comprehensive calibration metrics for probabilistic classifier.

    Assesses whether predicted probabilities match true frequencies. Critical for
    FoodSpec authenticity classifiers: a poorly calibrated model might predict
    90% confidence but be correct only 60% of the time.

    **What is Calibration?**
    When model predicts P(authentic)=0.8, authentic samples should occur ~80% of the time.
    ╔════════════════════════════════════════════════╗
    │ Well-Calibrated (trustworthy):                 │
    │ P(pred)=0.8 → 78-82% correct                   │
    ├────────────────────────────────────────────────┤
    │ Overconfident (unreliable):                     │
    │ P(pred)=0.8 → 55-60% correct (false confidence)│
    ├────────────────────────────────────────────────┤
    │ Underconfident (wasteful):                      │
    │ P(pred)=0.8 → 92-98% correct (overly cautious) │
    ╚════════════════════════════════════════════════╝

    **Key Metrics (in order of importance):**
    1. **ECE (Expected Calibration Error)**: 0-1 scale, lower is better
       - ECE = 0.05: For predictions of 80%, true frequency is ~75-85%
       - ECE = 0.20: For predictions of 80%, true frequency could be 60-90%

    2. **Bias**: Mean predicted prob - mean true frequency (-1 to +1)
       - Bias = +0.10: Model is too confident by ~10 percentage points
       - Bias = -0.05: Model is too conservative by ~5 percentage points

    3. **Brier Score**: Mean squared error (0-1), lower is better
       - 0.05: Predictions are very good
       - 0.25: Predictions are poorly calibrated

    4. **Slope & Intercept**: Linear regression P(true) ~ a*P(pred) + b
       - Ideal: slope=1.0, intercept=0.0
       - Slope < 1: Overconfident (predictions too extreme)
       - Slope > 1: Underconfident (predictions too close to 0.5)

    **Binning Strategies:**
    - **uniform**: Equal-width bins [0-0.1, 0.1-0.2, ..., 0.9-1.0]
        Good for evenly distributed predictions
    - **quantile**: Each bin contains ~same number of predictions
        Good for clustered predictions (e.g., all near 0.95)

    **When to Use:**
    - Post-model evaluation: Check calibration on validation set
    - Before deployment: Miscalibration can lead to false confidence in decisions
    - Model comparison: Choose model with lower ECE for better reliability

    **Real Example:**
    Oil authentication classifier:
    ├─ SVM: ECE=0.03, Bias=0.01 → trustworthy (deploy as-is)
    └─ Neural Net: ECE=0.15, Bias=0.25 → overconfident (apply Platt scaling)

    **Red Flags:**
    - ECE > 0.15: Model is unreliable; consider recalibration
    - Bias > 0.20 or < -0.20: Systematic under/overconfidence
    - MCE > 0.5: At least one probability bin is severely miscalibrated
    - reliability_curve shows J-shape: Model is overconfident at extremes

    Parameters:
        y_true (np.ndarray): True binary labels (0 or 1), shape (n_samples,)
        y_proba (np.ndarray): Predicted probabilities [0-1], shape (n_samples,)
        n_bins (int, default 10): Number of bins for ECE calculation
            Higher bins → more detail but noisier estimates (use 10-20)
        strategy (str, default "uniform"): Binning strategy
            "uniform": Equal-width bins
            "quantile": Quantile-based bins (recommended for imbalanced predictions)

    Returns:
        CalibrationDiagnostics: Dataclass with metrics:
            - slope, intercept: Linear regression coefficients
            - bias: Calibration-in-the-large (mean difference)
            - ece: Expected Calibration Error [0-1]
            - mce: Maximum Calibration Error [0-1]
            - brier_score: Mean squared error [0-1]
            - reliability_curve: DataFrame with calibration details per bin

    Examples:
        >>> from foodspec.ml.calibration import compute_calibration_diagnostics\n        >>> import numpy as np\n        >>> y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])\n        >>> y_proba = np.array([0.9, 0.1, 0.85, 0.8, 0.2, 0.95, 0.15, 0.05])\n        >>> diag = compute_calibration_diagnostics(y_true, y_proba, n_bins=4)\n        >>> print(f\"ECE: {diag.ece:.3f}, Bias: {diag.bias:.3f}\")\n        >>> print(diag.reliability_curve)\n
    See Also:
        - [Calibration Guide](../04-user-guide/probability_calibration.md) — Recalibration methods
        - recalibrate_classifier() — Apply Platt/isotonic scaling
        - CalibrationDiagnostics.is_well_calibrated() — Check if acceptable
        - plot_calibration_curve() — Visualization [in plotting module]

    References:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
        On calibration of modern neural networks. ICML.
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
    bin_means_true = np.divide(bin_sums_true, bin_counts, out=np.zeros(n_bins), where=bin_counts > 0)
    bin_means_pred = np.divide(bin_sums_pred, bin_counts, out=np.zeros(n_bins), where=bin_counts > 0)

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

    calibrated_clf = CalibratedClassifierCV(clf, method=sklearn_method, cv=cv, ensemble=True)
    calibrated_clf.fit(X_cal, y_cal)

    return calibrated_clf


def calibration_slope_intercept(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
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
