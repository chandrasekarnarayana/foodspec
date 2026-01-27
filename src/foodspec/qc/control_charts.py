from __future__ import annotations

"""
Control chart utilities for QC monitoring.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
_D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

_A3 = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975}
_B3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.03, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284}
_B4 = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.97, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}


@dataclass
class ControlChartResult:
    chart: str
    points: np.ndarray
    center: float
    ucl: float
    lcl: float
    signals: List[int]
    run_signals: List[int]
    meta: Dict[str, float]


@dataclass
class ControlChartGroup:
    xbar: ControlChartResult
    variability: ControlChartResult


def _flatten_to_subgroups(data: np.ndarray, subgroup_size: int) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        if x.size % subgroup_size != 0:
            raise ValueError("Data length must be divisible by subgroup_size.")
        x = x.reshape(-1, subgroup_size)
    return x


def _signal_indices(points: np.ndarray, ucl: float, lcl: float) -> List[int]:
    return [int(i) for i, v in enumerate(points) if v > ucl or v < lcl]


def _run_rule(points: np.ndarray, center: float, run_length: int = 7) -> List[int]:
    signs = points > center
    run_indices = []
    streak = 1
    for i in range(1, len(points)):
        if signs[i] == signs[i - 1]:
            streak += 1
        else:
            streak = 1
        if streak >= run_length:
            run_indices.append(i)
    return run_indices


def xbar_r_chart(data: np.ndarray, subgroup_size: int) -> ControlChartGroup:
    """X-bar and R chart for subgrouped data."""
    if subgroup_size not in _A2:
        raise ValueError("Unsupported subgroup_size for X-bar/R chart.")
    subgroups = _flatten_to_subgroups(data, subgroup_size)
    xbar = subgroups.mean(axis=1)
    ranges = subgroups.max(axis=1) - subgroups.min(axis=1)
    xbar_bar = float(xbar.mean())
    r_bar = float(ranges.mean())
    ucl_x = xbar_bar + _A2[subgroup_size] * r_bar
    lcl_x = xbar_bar - _A2[subgroup_size] * r_bar
    ucl_r = _D4[subgroup_size] * r_bar
    lcl_r = _D3[subgroup_size] * r_bar
    return ControlChartGroup(
        xbar=ControlChartResult(
            chart="xbar",
            points=xbar,
            center=xbar_bar,
            ucl=float(ucl_x),
            lcl=float(lcl_x),
            signals=_signal_indices(xbar, ucl_x, lcl_x),
            run_signals=_run_rule(xbar, xbar_bar),
            meta={"subgroup_size": float(subgroup_size)},
        ),
        variability=ControlChartResult(
            chart="r",
            points=ranges,
            center=r_bar,
            ucl=float(ucl_r),
            lcl=float(lcl_r),
            signals=_signal_indices(ranges, ucl_r, lcl_r),
            run_signals=_run_rule(ranges, r_bar),
            meta={"subgroup_size": float(subgroup_size)},
        ),
    )


def xbar_s_chart(data: np.ndarray, subgroup_size: int) -> ControlChartGroup:
    """X-bar and S chart for subgrouped data."""
    if subgroup_size not in _A3:
        raise ValueError("Unsupported subgroup_size for X-bar/S chart.")
    subgroups = _flatten_to_subgroups(data, subgroup_size)
    xbar = subgroups.mean(axis=1)
    s_vals = subgroups.std(axis=1, ddof=1)
    xbar_bar = float(xbar.mean())
    s_bar = float(s_vals.mean())
    ucl_x = xbar_bar + _A3[subgroup_size] * s_bar
    lcl_x = xbar_bar - _A3[subgroup_size] * s_bar
    ucl_s = _B4[subgroup_size] * s_bar
    lcl_s = _B3[subgroup_size] * s_bar
    return ControlChartGroup(
        xbar=ControlChartResult(
            chart="xbar",
            points=xbar,
            center=xbar_bar,
            ucl=float(ucl_x),
            lcl=float(lcl_x),
            signals=_signal_indices(xbar, ucl_x, lcl_x),
            run_signals=_run_rule(xbar, xbar_bar),
            meta={"subgroup_size": float(subgroup_size)},
        ),
        variability=ControlChartResult(
            chart="s",
            points=s_vals,
            center=s_bar,
            ucl=float(ucl_s),
            lcl=float(lcl_s),
            signals=_signal_indices(s_vals, ucl_s, lcl_s),
            run_signals=_run_rule(s_vals, s_bar),
            meta={"subgroup_size": float(subgroup_size)},
        ),
    )


def individuals_mr_chart(values: np.ndarray) -> ControlChartGroup:
    """Individuals and Moving Range chart."""
    x = np.asarray(values, dtype=float)
    mr = np.abs(np.diff(x))
    mr_bar = float(np.mean(mr)) if mr.size else 0.0
    sigma = mr_bar / 1.128 if mr_bar > 0 else float(np.std(x, ddof=1))
    center = float(np.mean(x))
    ucl_x = center + 3 * sigma
    lcl_x = center - 3 * sigma
    ucl_mr = 3.267 * mr_bar
    lcl_mr = 0.0
    return ControlChartGroup(
        xbar=ControlChartResult(
            chart="individuals",
            points=x,
            center=center,
            ucl=float(ucl_x),
            lcl=float(lcl_x),
            signals=_signal_indices(x, ucl_x, lcl_x),
            run_signals=_run_rule(x, center),
            meta={"sigma": float(sigma)},
        ),
        variability=ControlChartResult(
            chart="moving_range",
            points=mr,
            center=mr_bar,
            ucl=float(ucl_mr),
            lcl=float(lcl_mr),
            signals=_signal_indices(mr, ucl_mr, lcl_mr),
            run_signals=_run_rule(mr, mr_bar),
            meta={},
        ),
    )


def cusum_chart(values: np.ndarray, *, target: Optional[float] = None, k: float = 0.5, h: float = 5.0) -> Dict:
    """CUSUM chart for detecting small shifts."""
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x)) if target is None else float(target)
    pos = []
    neg = []
    c_pos = 0.0
    c_neg = 0.0
    signals = []
    for i, val in enumerate(x):
        c_pos = max(0.0, c_pos + val - mean - k)
        c_neg = min(0.0, c_neg + val - mean + k)
        pos.append(c_pos)
        neg.append(c_neg)
        if c_pos > h or abs(c_neg) > h:
            signals.append(i)
    return {"pos": np.asarray(pos), "neg": np.asarray(neg), "center": mean, "h": h, "signals": signals}


def ewma_chart(values: np.ndarray, *, lam: float = 0.2, l: float = 3.0) -> Dict:
    """EWMA chart for smoothing and shift detection."""
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    ewma = []
    z = mean
    for val in x:
        z = lam * val + (1 - lam) * z
        ewma.append(z)
    ewma = np.asarray(ewma)
    limits = []
    for i in range(1, len(x) + 1):
        sigma_z = sigma * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * i)))
        limits.append((mean - l * sigma_z, mean + l * sigma_z))
    lcl = np.asarray([lo for lo, _ in limits])
    ucl = np.asarray([hi for _, hi in limits])
    signals = [int(i) for i, val in enumerate(ewma) if val < lcl[i] or val > ucl[i]]
    return {"ewma": ewma, "center": mean, "lcl": lcl, "ucl": ucl, "signals": signals}


def p_chart(defect_counts: np.ndarray, sample_sizes: np.ndarray) -> ControlChartResult:
    """P chart for proportions."""
    d = np.asarray(defect_counts, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    p = d / n
    center = float(np.sum(d) / np.sum(n))
    sigma = np.sqrt(center * (1 - center) / n)
    ucl = center + 3 * sigma
    lcl = np.maximum(0.0, center - 3 * sigma)
    signals = [int(i) for i, val in enumerate(p) if val < lcl[i] or val > ucl[i]]
    return ControlChartResult(
        chart="p",
        points=p,
        center=center,
        ucl=float(np.max(ucl)),
        lcl=float(np.min(lcl)),
        signals=signals,
        run_signals=_run_rule(p, center),
        meta={"ucl": ucl.tolist(), "lcl": lcl.tolist()},
    )


def np_chart(defect_counts: np.ndarray, sample_size: int) -> ControlChartResult:
    """NP chart for counts with constant sample size."""
    d = np.asarray(defect_counts, dtype=float)
    n = float(sample_size)
    pbar = float(np.mean(d) / n)
    center = n * pbar
    sigma = np.sqrt(n * pbar * (1 - pbar))
    ucl = center + 3 * sigma
    lcl = max(0.0, center - 3 * sigma)
    return ControlChartResult(
        chart="np",
        points=d,
        center=center,
        ucl=float(ucl),
        lcl=float(lcl),
        signals=_signal_indices(d, ucl, lcl),
        run_signals=_run_rule(d, center),
        meta={"sample_size": float(sample_size)},
    )


def c_chart(counts: np.ndarray) -> ControlChartResult:
    """C chart for defect counts."""
    c = np.asarray(counts, dtype=float)
    center = float(np.mean(c))
    sigma = np.sqrt(center)
    ucl = center + 3 * sigma
    lcl = max(0.0, center - 3 * sigma)
    return ControlChartResult(
        chart="c",
        points=c,
        center=center,
        ucl=float(ucl),
        lcl=float(lcl),
        signals=_signal_indices(c, ucl, lcl),
        run_signals=_run_rule(c, center),
        meta={},
    )


def u_chart(counts: np.ndarray, sample_sizes: np.ndarray) -> ControlChartResult:
    """U chart for defect counts per unit."""
    c = np.asarray(counts, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    u = c / n
    center = float(np.sum(c) / np.sum(n))
    sigma = np.sqrt(center / n)
    ucl = center + 3 * sigma
    lcl = np.maximum(0.0, center - 3 * sigma)
    signals = [int(i) for i, val in enumerate(u) if val < lcl[i] or val > ucl[i]]
    return ControlChartResult(
        chart="u",
        points=u,
        center=center,
        ucl=float(np.max(ucl)),
        lcl=float(np.min(lcl)),
        signals=signals,
        run_signals=_run_rule(u, center),
        meta={"ucl": ucl.tolist(), "lcl": lcl.tolist()},
    )


def levey_jennings(values: np.ndarray, *, mean: Optional[float] = None, sd: Optional[float] = None) -> ControlChartResult:
    """Levey-Jennings chart (mean +/- 3 SD)."""
    x = np.asarray(values, dtype=float)
    center = float(np.mean(x)) if mean is None else float(mean)
    sigma = float(np.std(x, ddof=1)) if sd is None else float(sd)
    ucl = center + 3 * sigma
    lcl = center - 3 * sigma
    return ControlChartResult(
        chart="levey_jennings",
        points=x,
        center=center,
        ucl=float(ucl),
        lcl=float(lcl),
        signals=_signal_indices(x, ucl, lcl),
        run_signals=_run_rule(x, center),
        meta={"sigma": sigma},
    )


def capability_analysis(values: np.ndarray, lsl: float, usl: float) -> Dict[str, float]:
    """Capability indices Cp/Cpk for a process."""
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    cp = (usl - lsl) / (6 * sigma) if sigma > 0 else float("nan")
    cpu = (usl - mean) / (3 * sigma) if sigma > 0 else float("nan")
    cpl = (mean - lsl) / (3 * sigma) if sigma > 0 else float("nan")
    cpk = min(cpu, cpl)
    return {"cp": float(cp), "cpk": float(cpk), "cpu": float(cpu), "cpl": float(cpl)}


def pareto_counts(categories: List[str]) -> Dict[str, int]:
    """Compute Pareto-ordered counts for categorical defects."""
    values, counts = np.unique(categories, return_counts=True)
    order = np.argsort(counts)[::-1]
    return {str(values[i]): int(counts[i]) for i in order}


__all__ = [
    "ControlChartResult",
    "ControlChartGroup",
    "xbar_r_chart",
    "xbar_s_chart",
    "individuals_mr_chart",
    "cusum_chart",
    "ewma_chart",
    "p_chart",
    "np_chart",
    "c_chart",
    "u_chart",
    "levey_jennings",
    "capability_analysis",
    "pareto_counts",
]
