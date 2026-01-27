from __future__ import annotations

"""
Distribution fitting and probability diagnostics for QC metrics.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats

_DISTRIBUTIONS: Dict[str, object] = {
    "normal": stats.norm,
    "lognorm": stats.lognorm,
    "weibull": stats.weibull_min,
    "gamma": stats.gamma,
    "beta": stats.beta,
    "chi2": stats.chi2,
}


@dataclass
class DistributionFit:
    name: str
    params: Tuple[float, ...]
    loglik: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    notes: str = ""

    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "params": list(self.params),
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "ks_stat": self.ks_stat,
            "ks_pvalue": self.ks_pvalue,
            "notes": self.notes,
        }


def _loglik(dist, data: np.ndarray, params: Tuple[float, ...]) -> float:
    pdf = dist.pdf(data, *params)
    pdf = np.clip(pdf, 1e-12, None)
    return float(np.sum(np.log(pdf)))


def fit_distribution(
    data: Iterable[float],
    dist_name: str,
    *,
    auto_scale_beta: bool = True,
) -> DistributionFit:
    """Fit a named distribution and compute GOF diagnostics."""
    values = np.asarray(data, dtype=float)
    if values.size < 3:
        raise ValueError("Need at least 3 samples to fit a distribution.")
    if dist_name not in _DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution '{dist_name}'.")
    dist = _DISTRIBUTIONS[dist_name]
    notes = ""

    if dist_name == "beta" and auto_scale_beta:
        vmin, vmax = float(values.min()), float(values.max())
        if vmin == vmax:
            raise ValueError("Beta fit requires non-constant data.")
        values = (values - vmin) / (vmax - vmin)
        notes = "beta_auto_scaled"

    params = dist.fit(values)
    ll = _loglik(dist, values, params)
    k = len(params)
    n = values.size
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    ks_stat, ks_p = stats.kstest(values, dist.name, args=params)
    return DistributionFit(
        name=dist_name,
        params=tuple(float(p) for p in params),
        loglik=float(ll),
        aic=float(aic),
        bic=float(bic),
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_p),
        notes=notes,
    )


def compare_distributions(data: Iterable[float], dist_names: Iterable[str]) -> List[DistributionFit]:
    """Fit multiple distributions and sort by AIC."""
    fits = [fit_distribution(data, name) for name in dist_names]
    return sorted(fits, key=lambda f: f.aic)


def probability_plot_data(
    data: Iterable[float],
    dist_name: str = "normal",
) -> Dict[str, np.ndarray]:
    """Return data for probability plots (theoretical vs ordered)."""
    values = np.asarray(data, dtype=float)
    if dist_name not in _DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution '{dist_name}'.")
    dist = _DISTRIBUTIONS[dist_name]
    (osm, osr), (slope, intercept, r) = stats.probplot(values, dist=dist, plot=None)
    return {
        "theoretical": np.asarray(osm, dtype=float),
        "ordered": np.asarray(osr, dtype=float),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
    }


__all__ = [
    "DistributionFit",
    "fit_distribution",
    "compare_distributions",
    "probability_plot_data",
]
