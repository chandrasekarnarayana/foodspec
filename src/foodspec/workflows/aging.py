from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from foodspec.core.time import TimeSpectrumSet
from foodspec.stats.time_metrics import linear_slope, quadratic_acceleration


@dataclass
class TrajectoryFit:
    method: Literal["linear", "spline"]
    params: Dict[str, float]
    times: np.ndarray
    values: np.ndarray
    fitted: np.ndarray


@dataclass
class AgingResult:
    fits: Dict[str, TrajectoryFit]
    metrics: pd.DataFrame
    stages: pd.DataFrame


def compute_degradation_trajectories(
    ds: TimeSpectrumSet,
    value_col: str,
    method: Literal["linear", "spline"] = "linear",
) -> AgingResult:
    if ds.entity_col is None:
        raise ValueError("entity_col must be set in TimeSpectrumSet.")
    groups = ds.groups_by_entity()
    fits: Dict[str, TrajectoryFit] = {}
    rows: List[Dict[str, float]] = []
    stage_records: List[Dict[str, object]] = []
    for ent, idx in groups.items():
        times = ds.metadata.loc[idx, ds.time_col].to_numpy(dtype=float)
        values = ds.metadata.loc[idx, value_col].to_numpy(dtype=float)
        order = np.argsort(times)
        times = times[order]
        values = values[order]
        if method == "linear":
            m, b = linear_slope(times, values)
            fitted = m * times + b
            fits[ent] = TrajectoryFit(method="linear", params={"slope": m, "intercept": b}, times=times, values=values, fitted=fitted)
        elif method == "spline":
            s = UnivariateSpline(times, values, s=0.0)
            fitted = s(times)
            acc = float(np.mean(s.derivative(n=2)(times)))
            fits[ent] = TrajectoryFit(method="spline", params={"acceleration_mean": acc}, times=times, values=values, fitted=fitted)
        else:
            raise ValueError("method must be 'linear' or 'spline'.")
        slope, _ = linear_slope(times, values)
        accel = quadratic_acceleration(times, values)
        rows.append({"entity": ent, "slope": float(slope), "acceleration": float(accel)})
        n = len(times)
        for i in range(n):
            frac = (i + 1) / n
            stage = _stage_from_fraction(frac)
            stage_records.append({"entity": ent, "time": float(times[i]), "stage": stage})
    metrics = pd.DataFrame(rows)
    stages = pd.DataFrame(stage_records)
    return AgingResult(fits=fits, metrics=metrics, stages=stages)


def _stage_from_fraction(frac: float) -> str:
    if frac <= 1.0 / 3.0:
        return "early"
    if frac <= 2.0 / 3.0:
        return "mid"
    return "late"


__all__ = ["AgingResult", "TrajectoryFit", "compute_degradation_trajectories"]
