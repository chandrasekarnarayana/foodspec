"""
Robustness utilities: bootstrap and permutation checks for model metrics.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def bootstrap_metric(
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: tuple[float, float] = (2.5, 97.5),
    random_state: Optional[int] = None,
) -> dict:
    """
    Estimate robustness of a metric via bootstrap resampling.

    Parameters
    ----------
    metric_func : callable
        Function taking (y_true, y_pred) and returning a scalar metric.
    y_true : array-like
        True targets.
    y_pred : array-like
        Predicted targets.
    n_bootstrap : int, optional
        Number of bootstrap samples, by default 1000.
    ci : tuple, optional
        Confidence interval percentiles, by default (2.5, 97.5).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict
        Contains observed metric, bootstrap samples, and CI bounds.
    """

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples.append(metric_func(y_true[idx], y_pred[idx]))
    samples = np.asarray(samples)
    observed = metric_func(y_true, y_pred)
    ci_lower, ci_upper = np.percentile(samples, ci)
    return {
        "observed": observed,
        "bootstrap_samples": samples,
        "ci": (ci_lower, ci_upper),
    }


def permutation_test_metric(
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 1000,
    metric_higher_is_better: bool = True,
    random_state: Optional[int] = None,
) -> dict:
    """
    Permutation test for a metric by shuffling the target labels.

    Parameters
    ----------
    metric_func : callable
        Function taking (y_true, y_pred) and returning a scalar metric.
    y_true : array-like
        True targets.
    y_pred : array-like
        Predicted targets.
    n_permutations : int, optional
        Number of permutations, by default 1000.
    metric_higher_is_better : bool, optional
        If False, p-value computation is reversed (e.g., for RMSE), by default True.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict
        Contains observed metric, permutation samples, and an empirical p-value.
    """

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    observed = metric_func(y_true, y_pred)
    permuted = []
    for _ in range(n_permutations):
        permuted_true = rng.permutation(y_true)
        permuted.append(metric_func(permuted_true, y_pred))
    permuted = np.asarray(permuted)
    if metric_higher_is_better:
        p_value = np.mean(permuted >= observed)
    else:
        p_value = np.mean(permuted <= observed)
    return {"observed": observed, "permuted_samples": permuted, "p_value": float(p_value)}
