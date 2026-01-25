"""
Statistical utilities for evaluation and comparison.

Provides:
- Bootstrap confidence intervals for metrics
- ANOVA for comparing metrics across factors
- MANOVA placeholder for multi-metric comparisons
"""

from typing import Optional, Tuple, Union

import numpy as np


def bootstrap_ci(
    metric_values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.

    Supports both per-sample (from predictions table) and per-fold bootstrapping.
    Uses percentile method for CI estimation.

    Parameters
    ----------
    metric_values : np.ndarray
        Array of metric values. Can be:
        - Per-fold values (e.g., [0.85, 0.87, 0.83, 0.88, 0.86])
        - Per-sample values (e.g., correctness for each prediction)
    n_boot : int, default=2000
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Significance level for CI (e.g., 0.05 for 95% CI).
    seed : int, optional
        Random seed for deterministic resampling.

    Returns
    -------
    lower : float
        Lower bound of the confidence interval.
    median : float
        Median (50th percentile) of bootstrap distribution.
    upper : float
        Upper bound of the confidence interval.

    Examples
    --------
    >>> # Per-fold bootstrap
    >>> fold_accuracies = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
    >>> lower, median, upper = bootstrap_ci(fold_accuracies, seed=42)
    >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}], median={median:.3f}")

    >>> # Per-sample bootstrap
    >>> predictions = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])  # correctness
    >>> lower, median, upper = bootstrap_ci(predictions, seed=42)

    Notes
    -----
    - Uses stratified resampling (bootstrap with replacement)
    - Deterministic when seed is provided
    - Bounds are guaranteed to be monotonic (lower <= median <= upper)
    - For small sample sizes, consider using higher n_boot
    """
    if len(metric_values) == 0:
        raise ValueError("metric_values cannot be empty")

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    # Handle edge case: single value
    if len(metric_values) == 1:
        val = float(metric_values[0])
        return val, val, val

    # Set up random state for determinism
    rng = np.random.RandomState(seed)

    # Bootstrap resampling
    n_samples = len(metric_values)
    bootstrap_metrics = np.zeros(n_boot)

    for i in range(n_boot):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = metric_values[indices]
        # Compute metric on bootstrap sample (mean for aggregation)
        bootstrap_metrics[i] = np.mean(bootstrap_sample)

    # Compute percentile-based CI
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(bootstrap_metrics, lower_percentile)
    median = np.percentile(bootstrap_metrics, 50)
    upper = np.percentile(bootstrap_metrics, upper_percentile)

    # Ensure monotonicity (can be violated due to floating point)
    lower = min(lower, median)
    upper = max(upper, median)

    return float(lower), float(median), float(upper)


def anova_on_metric(
    metrics_per_fold: dict,
    factor: str = "model",
) -> dict:
    """
    Perform one-way ANOVA on metric values grouped by factor.

    Tests whether there are statistically significant differences in a metric
    across different levels of a factor (e.g., different models, recipes, batches).

    Parameters
    ----------
    metrics_per_fold : dict
        Dictionary mapping factor levels to arrays of metric values.
        Example: {"model_A": [0.85, 0.87, 0.83], "model_B": [0.80, 0.82, 0.81]}
    factor : str, default="model"
        Name of the factor being compared. Used for result documentation.
        Common values: "model", "recipe", "batch"

    Returns
    -------
    result : dict
        Dictionary containing:
        - "factor": Name of the factor
        - "f_statistic": F-statistic from ANOVA
        - "p_value": p-value from ANOVA
        - "df_between": Degrees of freedom between groups
        - "df_within": Degrees of freedom within groups
        - "groups": List of group names
        - "n_groups": Number of groups
        - "total_samples": Total number of samples

    Raises
    ------
    ImportError
        If scipy is not installed (optional dependency).
    ValueError
        If fewer than 2 groups provided or any group is empty.

    Examples
    --------
    >>> metrics = {
    ...     "model_A": np.array([0.85, 0.87, 0.83, 0.88, 0.86]),
    ...     "model_B": np.array([0.80, 0.82, 0.81, 0.79, 0.83]),
    ...     "model_C": np.array([0.90, 0.91, 0.89, 0.92, 0.90]),
    ... }
    >>> result = anova_on_metric(metrics, factor="model")
    >>> print(f"F={result['f_statistic']:.2f}, p={result['p_value']:.4f}")

    Notes
    -----
    - Requires scipy (optional dependency)
    - Assumes normality and equal variance (standard ANOVA assumptions)
    - For non-normal data, consider non-parametric alternatives (Kruskal-Wallis)
    - Use p < 0.05 as threshold for statistical significance
    """
    try:
        from scipy import stats
    except ImportError as e:
        raise ImportError(
            "scipy is required for ANOVA. Install with: pip install scipy"
        ) from e

    if not metrics_per_fold:
        raise ValueError("metrics_per_fold cannot be empty")

    if len(metrics_per_fold) < 2:
        raise ValueError(f"Need at least 2 groups for ANOVA, got {len(metrics_per_fold)}")

    # Validate all groups have data
    groups = list(metrics_per_fold.keys())
    samples = []
    for group_name, values in metrics_per_fold.items():
        if len(values) == 0:
            raise ValueError(f"Group '{group_name}' has no samples")
        samples.append(np.asarray(values))

    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(*samples)

    # Compute degrees of freedom
    n_groups = len(samples)
    total_samples = sum(len(s) for s in samples)
    df_between = n_groups - 1
    df_within = total_samples - n_groups

    return {
        "factor": factor,
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "groups": groups,
        "n_groups": n_groups,
        "total_samples": total_samples,
    }


def manova_placeholder(
    metrics_per_fold: dict,
    factor: str = "model",
) -> None:
    """
    MANOVA (Multivariate Analysis of Variance) placeholder.

    MANOVA extends ANOVA to multiple dependent variables (metrics) simultaneously,
    testing whether there are statistically significant differences across factor
    levels when considering all metrics together.

    This function is a placeholder and raises NotImplementedError. It documents
    the intended use case for future implementation.

    Intended Functionality
    ----------------------
    When implemented, this function would:

    1. Accept multiple metrics per factor level:
       - metrics_per_fold: Dict mapping factor levels to DataFrames
       - Each DataFrame has columns for different metrics (accuracy, F1, etc.)
       - Each row represents one fold/replicate

    2. Perform MANOVA using statsmodels or scikit-learn:
       - Test multivariate differences across factor levels
       - Account for correlations between metrics
       - Provide Wilks' Lambda, Pillai's trace, etc.

    3. Return comprehensive results:
       - Test statistics (Wilks' Lambda, Hotelling-Lawley, Pillai-Bartlett)
       - p-values for overall effect
       - Effect sizes
       - Post-hoc comparisons if requested

    Example Use Case
    ----------------
    >>> # Hypothetical usage (not yet implemented)
    >>> metrics = {
    ...     "model_A": pd.DataFrame({
    ...         "accuracy": [0.85, 0.87, 0.83],
    ...         "f1": [0.83, 0.85, 0.81],
    ...         "auroc": [0.90, 0.92, 0.88]
    ...     }),
    ...     "model_B": pd.DataFrame({
    ...         "accuracy": [0.80, 0.82, 0.81],
    ...         "f1": [0.78, 0.80, 0.79],
    ...         "auroc": [0.85, 0.87, 0.86]
    ...     })
    ... }
    >>> result = manova_placeholder(metrics, factor="model")  # Would work when implemented

    Parameters
    ----------
    metrics_per_fold : dict
        Dictionary mapping factor levels to metric DataFrames (intended structure).
    factor : str, default="model"
        Name of the factor being compared.

    Raises
    ------
    NotImplementedError
        Always raised. This is a placeholder for future implementation.

    Notes
    -----
    Implementation considerations:
    - Requires statsmodels or equivalent (statsmodels.multivariate.manova.MANOVA)
    - Should handle missing data gracefully
    - Should validate MANOVA assumptions (multivariate normality, homogeneity)
    - Consider providing diagnostic plots
    - May want to support contrast matrices for specific comparisons

    See Also
    --------
    anova_on_metric : One-way ANOVA for single metric comparisons
    """
    raise NotImplementedError(
        "MANOVA is not yet implemented. This is a placeholder for multi-metric "
        "statistical comparisons across factor levels.\n\n"
        "Intended use:\n"
        "  - Compare multiple metrics simultaneously (accuracy, F1, AUROC, etc.)\n"
        "  - Account for correlations between metrics\n"
        "  - Provide multivariate test statistics (Wilks' Lambda, etc.)\n\n"
        "When implemented, this will require statsmodels:\n"
        "  pip install statsmodels\n\n"
        "For now, use anova_on_metric() for single-metric comparisons."
    )
