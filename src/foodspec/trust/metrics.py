"""Trust and uncertainty metrics for calibration and abstention."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from foodspec.trust.reliability import brier_score, expected_calibration_error


def negative_log_likelihood(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    """Compute negative log-likelihood for multiclass probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if proba.ndim != 2:
        raise ValueError("proba must be 2D")
    if y_true.shape[0] != proba.shape[0]:
        raise ValueError("y_true and proba must have same length")
    proba = np.clip(proba, eps, 1.0)
    nll = -np.mean(np.log(proba[np.arange(len(y_true)), y_true]))
    return float(nll)


def compute_calibration_metrics(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Compute ECE, Brier, and NLL."""
    ece = expected_calibration_error(y_true, proba, n_bins=n_bins)
    brier = brier_score(y_true, proba)
    nll = negative_log_likelihood(y_true, proba)
    return {"ece": float(ece), "brier": float(brier), "nll": float(nll)}


def risk_coverage_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> dict[str, List[float]]:
    """Compute risk-coverage curve for abstention by confidence threshold."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be 2D")
    max_prob = proba.max(axis=1)
    y_pred = proba.argmax(axis=1)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    thresholds = list(thresholds)
    coverage: List[float] = []
    risk: List[float] = []
    for tau in thresholds:
        accept = max_prob >= tau
        if accept.any():
            cov = float(np.mean(accept))
            acc = float(np.mean(y_pred[accept] == y_true[accept]))
            coverage.append(cov)
            risk.append(1.0 - acc)
        else:
            coverage.append(0.0)
            risk.append(0.0)
    return {"thresholds": thresholds, "coverage": coverage, "risk": risk}


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for mean."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(float(np.mean(sample)))
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(np.mean(values)), lower, upper


def bootstrap_coverage_efficiency(
    covered: np.ndarray,
    set_sizes: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap coverage and efficiency (mean set size) summaries."""
    covered = np.asarray(covered, dtype=float)
    set_sizes = np.asarray(set_sizes, dtype=float)
    if covered.size == 0 or set_sizes.size == 0:
        return {
            "coverage": 0.0,
            "coverage_ci": (0.0, 0.0),
            "efficiency": 0.0,
            "efficiency_ci": (0.0, 0.0),
            "curve": [],
        }
    rng = np.random.default_rng(seed)
    n = len(covered)
    curves = []
    coverages = []
    efficiencies = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        cov = float(np.mean(covered[idx]))
        avg_size = float(np.mean(set_sizes[idx]))
        efficiency = float(1.0 / avg_size) if avg_size > 0 else 0.0
        coverages.append(cov)
        efficiencies.append(efficiency)
        curves.append({"coverage": cov, "efficiency": efficiency, "avg_set_size": avg_size})
    cov_ci = (float(np.quantile(coverages, 0.025)), float(np.quantile(coverages, 0.975)))
    eff_ci = (float(np.quantile(efficiencies, 0.025)), float(np.quantile(efficiencies, 0.975)))
    return {
        "coverage": float(np.mean(covered)),
        "coverage_ci": cov_ci,
        "efficiency": float(1.0 / np.mean(set_sizes)) if np.mean(set_sizes) > 0 else 0.0,
        "efficiency_ci": eff_ci,
        "curve": curves,
    }


__all__ = [
    "compute_calibration_metrics",
    "expected_calibration_error",
    "brier_score",
    "negative_log_likelihood",
    "risk_coverage_curve",
    "bootstrap_ci",
    "bootstrap_coverage_efficiency",
]
