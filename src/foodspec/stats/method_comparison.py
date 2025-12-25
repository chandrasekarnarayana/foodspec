"""
Bland–Altman analysis for method comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BlandAltmanResult:
    mean_diff: float
    loa_low: float
    loa_high: float


def bland_altman(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> BlandAltmanResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low = md - 1.96 * sd
    loa_high = md + 1.96 * sd
    return BlandAltmanResult(mean_diff=md, loa_low=float(loa_low), loa_high=float(loa_high))


def bland_altman_plot(a: np.ndarray, b: np.ndarray, title: str = "Bland–Altman"):
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
