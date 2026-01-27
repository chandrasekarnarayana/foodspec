"""
Miscellaneous metrics and diagnostics for model/QC reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy import stats


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Compute adjusted R-squared from R^2, sample size, and feature count."""
    if n_samples <= n_features + 1:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def aic_from_rss(rss: float, n_samples: int, n_params: int) -> float:
    """AIC from residual sum of squares (Gaussian assumption)."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    return float(n_samples * np.log(rss / n_samples + 1e-12) + 2 * n_params)


def bic_from_rss(rss: float, n_samples: int, n_params: int) -> float:
    """BIC from residual sum of squares (Gaussian assumption)."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    return float(n_samples * np.log(rss / n_samples + 1e-12) + n_params * np.log(n_samples))


def cronbach_alpha(data: np.ndarray) -> float:
    """Cronbach's alpha for reliability across items."""
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be 2D (n_samples, n_items).")
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    k = X.shape[1]
    if k < 2:
        return float("nan")
    if total_var == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


@dataclass
class RunsTestResult:
    n_runs: int
    z: float
    pvalue: float


def runs_test(values: Iterable[float], *, threshold: float | None = None) -> RunsTestResult:
    """Wald-Wolfowitz runs test for randomness."""
    x = np.asarray(list(values), dtype=float)
    if x.size < 2:
        raise ValueError("Runs test requires at least 2 observations.")
    if threshold is None:
        threshold = float(np.median(x))
    signs = x >= threshold
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    n1 = int(np.sum(signs))
    n2 = int(np.sum(~signs))
    if n1 == 0 or n2 == 0:
        return RunsTestResult(n_runs=int(runs), z=float("nan"), pvalue=float("nan"))
    mean_runs = (2 * n1 * n2) / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z = (runs - mean_runs) / np.sqrt(var_runs + 1e-12)
    pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
    return RunsTestResult(n_runs=int(runs), z=float(z), pvalue=float(pvalue))


def normal_tolerance_interval(
    data: Iterable[float],
    *,
    coverage: float = 0.95,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Approximate two-sided normal tolerance interval."""
    vals = np.asarray(list(data), dtype=float)
    n = vals.size
    if n < 2:
        raise ValueError("Need at least 2 samples for tolerance interval.")
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    k = stats.t.ppf(confidence, n - 1) * np.sqrt(1 + 1 / n)
    z = stats.norm.ppf((1 + coverage) / 2)
    k = k * (z / stats.norm.ppf((1 + confidence) / 2))
    return mean - k * std, mean + k * std


__all__ = [
    "adjusted_r2",
    "aic_from_rss",
    "bic_from_rss",
    "cronbach_alpha",
    "RunsTestResult",
    "runs_test",
    "normal_tolerance_interval",
]
