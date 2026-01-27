from __future__ import annotations

"""Model drift detection and recalibration triggers for production monitoring.

Detects distributional shifts in features, predictions, and performance.
"""


from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr


@dataclass
class DriftReport:
    """Report of detected drift between reference and current data.

    Attributes
    ----------
    drift_detected : bool
        True if any drift metric exceeds threshold.
    psi : float
        Population Stability Index (PSI < 0.1 = stable, 0.1-0.25 = moderate, >0.25 = significant).
    kl_divergence : float
        Kullback-Leibler divergence (0 = identical distributions).
    ks_statistic : float
        Kolmogorov-Smirnov test statistic.
    ks_pvalue : float
        KS test p-value (low = significant difference).
    wasserstein_distance : float
        Earth mover's distance between distributions.
    feature_drifts : Optional[Dict[str, float]]
        Per-feature PSI scores (if feature-level analysis performed).
    recommendation : str
        Action recommendation based on drift severity.
    """

    drift_detected: bool
    psi: float
    kl_divergence: float
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    feature_drifts: Optional[Dict[str, float]] = None
    recommendation: str = ""


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-10,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI measures the shift in distribution between reference and current data.
    Common interpretation:
    - PSI < 0.1: No significant change
    - 0.1 ≤ PSI < 0.25: Moderate change, investigate
    - PSI ≥ 0.25: Significant change, recalibration recommended

    Parameters
    ----------
    reference : np.ndarray
        Reference (training) distribution.
    current : np.ndarray
        Current (production) distribution.
    bins : int
        Number of bins for discretization.
    epsilon : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        PSI score.
    """
    reference = np.asarray(reference).ravel()
    current = np.asarray(current).ravel()

    # Create bins based on reference distribution
    bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)  # Remove duplicates

    # Compute proportions
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    ref_props = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    cur_props = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

    # PSI = sum((current% - expected%) * ln(current% / expected%))
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

    return float(psi)


def kl_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-10,
) -> float:
    """Compute Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    reference : np.ndarray
        Reference distribution.
    current : np.ndarray
        Current distribution.
    bins : int
        Number of bins for discretization.
    epsilon : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        KL divergence (0 = identical, higher = more different).
    """
    reference = np.asarray(reference).ravel()
    current = np.asarray(current).ravel()

    bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)

    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    ref_probs = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    cur_probs = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))

    kl = np.sum(rel_entr(cur_probs, ref_probs))

    return float(kl)


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05,
    bins: int = 10,
    method: Literal["auto", "continuous", "categorical"] = "auto",
) -> DriftReport:
    """Detect distributional drift using multiple statistical tests.

    Parameters
    ----------
    reference : np.ndarray
        Reference (training/baseline) data.
    current : np.ndarray
        Current (production) data.
    psi_threshold : float
        PSI threshold for drift detection (default 0.25).
    ks_alpha : float
        Significance level for KS test (default 0.05).
    bins : int
        Number of bins for PSI/KL calculations.
    method : Literal["auto", "continuous", "categorical"]
        Type of data (auto-detects if "auto").

    Returns
    -------
    DriftReport
        Comprehensive drift assessment.
    """
    reference = np.asarray(reference).ravel()
    current = np.asarray(current).ravel()

    # Auto-detect data type
    if method == "auto":
        n_unique_ref = len(np.unique(reference))
        n_unique_cur = len(np.unique(current))
        if n_unique_ref < 20 and n_unique_cur < 20:
            method = "categorical"
        else:
            method = "continuous"

    # Compute drift metrics
    psi = population_stability_index(reference, current, bins=bins)
    kl = kl_divergence(reference, current, bins=bins)

    # Kolmogorov-Smirnov test (continuous data)
    if method == "continuous":
        ks_stat, ks_pval = stats.ks_2samp(reference, current)
        wasserstein = stats.wasserstein_distance(reference, current)
    else:
        # For categorical, use chi-square-like comparison
        ks_stat = 0.0
        ks_pval = 1.0
        wasserstein = 0.0

    # Drift decision
    drift_detected = (psi > psi_threshold) or (ks_pval < ks_alpha)

    # Recommendation
    if psi >= 0.25:
        recommendation = "⚠️ CRITICAL: Significant drift detected. Recalibration required."
    elif psi >= 0.1:
        recommendation = "⚡ WARNING: Moderate drift. Monitor closely and consider recalibration."
    else:
        recommendation = "✅ OK: No significant drift detected."

    return DriftReport(
        drift_detected=drift_detected,
        psi=psi,
        kl_divergence=kl,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pval),
        wasserstein_distance=float(wasserstein),
        recommendation=recommendation,
    )


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[list] = None,
    psi_threshold: float = 0.25,
) -> Dict[str, DriftReport]:
    """Detect drift for each feature in a dataset.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference (training) features.
    current : pd.DataFrame
        Current (production) features.
    features : Optional[list]
        List of feature names to analyze. If None, uses all numeric columns.
    psi_threshold : float
        PSI threshold for drift detection.

    Returns
    -------
    Dict[str, DriftReport]
        Drift report for each feature.
    """
    if features is None:
        features = reference.select_dtypes(include=[np.number]).columns.tolist()

    reports = {}
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue

        ref_vals = reference[feat].dropna().values
        cur_vals = current[feat].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        reports[feat] = detect_drift(ref_vals, cur_vals, psi_threshold=psi_threshold)

    return reports


def should_recalibrate(
    reference_performance: float,
    current_performance: float,
    performance_threshold: float = 0.05,
    drift_report: Optional[DriftReport] = None,
    psi_threshold: float = 0.25,
) -> bool:
    """Determine if model recalibration is needed based on performance and drift.

    Parameters
    ----------
    reference_performance : float
        Baseline model performance metric (e.g., accuracy, AUC).
    current_performance : float
        Current production performance metric.
    performance_threshold : float
        Maximum acceptable performance drop (default 5%).
    drift_report : Optional[DriftReport]
        Drift analysis results (if available).
    psi_threshold : float
        PSI threshold for drift-based recalibration trigger.

    Returns
    -------
    bool
        True if recalibration is recommended.
    """
    # Performance degradation check
    performance_drop = reference_performance - current_performance
    performance_degraded = performance_drop > performance_threshold

    # Drift check
    drift_detected = False
    if drift_report is not None:
        drift_detected = drift_report.psi > psi_threshold

    return performance_degraded or drift_detected


__all__ = [
    "DriftReport",
    "population_stability_index",
    "kl_divergence",
    "detect_drift",
    "detect_feature_drift",
    "should_recalibrate",
]
