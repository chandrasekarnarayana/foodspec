from __future__ import annotations

"""
Bland–Altman analysis for method comparison.
"""


from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass
class BlandAltmanResult:
    mean_diff: float
    loa_low: float
    loa_high: float


@dataclass
class PassingBablokResult:
    slope: float
    intercept: float
    slope_ci: Tuple[float, float]
    intercept_ci: Tuple[float, float]
    n_pairs: int


def passing_bablok(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.05,
) -> PassingBablokResult:
    """Passing-Bablok regression for method comparison (robust, nonparametric)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)
    slopes = np.asarray(slopes)
    if slopes.size == 0:
        raise ValueError("Passing-Bablok requires varying x values.")
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))

    slopes_sorted = np.sort(slopes)
    m = slopes_sorted.size
    z = stats.norm.ppf(1 - alpha / 2)
    k = int(np.floor((m - z * np.sqrt(m * (m - 1) / 2)) / 2))
    l = int(np.ceil((m + z * np.sqrt(m * (m - 1) / 2)) / 2))
    k = max(0, min(m - 1, k))
    l = max(0, min(m - 1, l))
    slope_ci = (float(slopes_sorted[k]), float(slopes_sorted[l]))
    intercept_ci = (float(np.median(y - slope_ci[1] * x)), float(np.median(y - slope_ci[0] * x)))
    return PassingBablokResult(
        slope=slope,
        intercept=intercept,
        slope_ci=slope_ci,
        intercept_ci=intercept_ci,
        n_pairs=int(m),
    )


def lins_concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    var_x, var_y = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    return float((2 * cov) / (var_x + var_y + (mean_x - mean_y) ** 2 + 1e-12))


def bland_altman(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> BlandAltmanResult:
    """Perform Bland–Altman method comparison analysis.

    Computes mean difference and limits of agreement (LoA) between two
    measurement methods.

    Args:
        a: Measurements from method A.
        b: Measurements from method B.
        alpha: Significance level (currently unused, reserved for CI).

    Returns:
        A `BlandAltmanResult` containing mean difference, lower LoA, and upper LoA.

    Examples:
        >>> from foodspec.stats.method_comparison import bland_altman
        >>> import numpy as np
        >>> a = np.array([1.0, 2.0, 3.0, 4.0])
        >>> b = np.array([1.1, 1.9, 3.1, 4.2])
        >>> result = bland_altman(a, b)
        >>> abs(result.mean_diff) < 0.5
        True
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low = md - 1.96 * sd
    loa_high = md + 1.96 * sd
    return BlandAltmanResult(mean_diff=md, loa_low=float(loa_low), loa_high=float(loa_high))


def bland_altman_plot(a: np.ndarray, b: np.ndarray, title: str = "Bland–Altman"):
    """Generate a Bland–Altman plot for method comparison.

    Args:
        a: Measurements from method A.
        b: Measurements from method B.
        title: Plot title.

    Returns:
        A matplotlib Figure object.

    Examples:
        >>> from foodspec.stats.method_comparison import bland_altman_plot
        >>> import numpy as np
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.1, 1.9, 3.1])
        >>> fig = bland_altman_plot(a, b)
        >>> fig is not None
        True
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    avg = 0.5 * (a + b)
    diff = a - b
    res = bland_altman(a, b)
    plt.figure(figsize=(6, 4))
    plt.scatter(avg, diff, alpha=0.7)
    plt.axhline(res.mean_diff, color="red", linestyle="--", label="mean diff")
    plt.axhline(res.loa_low, color="gray", linestyle=":", label="LoA low")
    plt.axhline(res.loa_high, color="gray", linestyle=":", label="LoA high")
    plt.xlabel("Average of methods")
    plt.ylabel("Difference (A - B)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def passing_bablok_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Passing-Bablok",
) -> plt.Figure:
    """Plot Passing-Bablok regression with identity line."""
    res = passing_bablok(x, y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, alpha=0.7)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    plt.plot(xs, res.slope * xs + res.intercept, color="#2a6fdb", label="Passing-Bablok")
    plt.plot(xs, xs, color="gray", linestyle="--", label="Identity")
    plt.xlabel("Method A")
    plt.ylabel("Method B")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


__all__ = [
    "BlandAltmanResult",
    "PassingBablokResult",
    "bland_altman",
    "bland_altman_plot",
    "passing_bablok",
    "passing_bablok_plot",
    "lins_concordance_correlation",
]
