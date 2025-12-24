from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from foodspec.core.time import TimeSpectrumSet


@dataclass
class ShelfLifeEstimate:
    entity: str
    t_star: float
    ci_low: float
    ci_high: float
    slope: float
    intercept: float


def estimate_remaining_shelf_life(
    ds: TimeSpectrumSet,
    value_col: str,
    threshold: float,
) -> pd.DataFrame:
    if ds.entity_col is None:
        raise ValueError("entity_col must be set in TimeSpectrumSet.")
    results: list[Dict[str, float]] = []
    groups = ds.groups_by_entity()
    for ent, idx in groups.items():
        t = ds.metadata.loc[idx, ds.time_col].to_numpy(dtype=float)
        y = ds.metadata.loc[idx, value_col].to_numpy(dtype=float)
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        X = sm.add_constant(t)
        model = sm.OLS(y, X)
        fit = model.fit()
        b0 = float(fit.params[0])
        b1 = float(fit.params[1])
        cov = np.asarray(fit.cov_params())
        t_star = (threshold - b0) / b1
        grad = np.array([-1.0 / b1, -(threshold - b0) / (b1 * b1)], dtype=float)
        var_t = float(grad.T @ cov @ grad)
        se_t = float(np.sqrt(max(var_t, 0.0)))
        z = 1.959963984540054
        ci_low = float(t_star - z * se_t)
        ci_high = float(t_star + z * se_t)
        results.append(
            {
                "entity": ent,
                "t_star": float(t_star),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "slope": b1,
                "intercept": b0,
            }
        )
    return pd.DataFrame(results)


__all__ = ["ShelfLifeEstimate", "estimate_remaining_shelf_life"]
