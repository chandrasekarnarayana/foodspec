"""QC system: health scoring, outlier detection, drift monitoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from foodspec.core.dataset import FoodSpectrumSet


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _hampel_spike_fraction(row: np.ndarray, window: int = 5, n_sigma: float = 3.0) -> float:
    n = len(row)
    if n < window * 2 + 1:
        return 0.0
    spikes = 0
    for i in range(window, n - window):
        segment = row[i - window : i + window + 1]
        median = np.median(segment)
        mad = np.median(np.abs(segment - median)) + 1e-9
        if abs(row[i] - median) > n_sigma * 1.4826 * mad:
            spikes += 1
    return spikes / n


def _snr_ratio(row: np.ndarray) -> float:
    signal = np.percentile(row, 95) - np.percentile(row, 5)
    noise = np.std(np.diff(row)) + 1e-9
    return float(signal / noise)


def _saturation_fraction(row: np.ndarray, clip_tol: float = 1e-6) -> float:
    rmax, rmin = row.max(), row.min()
    upper = np.mean(row >= rmax - clip_tol)
    lower = np.mean(row <= rmin + clip_tol)
    return float(upper + lower)


def _baseline_lowfreq_energy(row: np.ndarray) -> float:
    fft = np.abs(np.fft.rfft(row))
    cutoff = max(1, int(0.05 * fft.shape[0]))
    return float(np.mean(fft[:cutoff]))


def _axis_coverage(wavenumbers: np.ndarray, reference_grid: Optional[np.ndarray]) -> float:
    if reference_grid is None:
        return 1.0
    ref_min, ref_max = reference_grid.min(), reference_grid.max()
    cur_min, cur_max = wavenumbers.min(), wavenumbers.max()
    overlap = max(0.0, min(cur_max, ref_max) - max(cur_min, ref_min))
    full = ref_max - ref_min + 1e-9
    return float(overlap / full)


def _replicate_distance(row: np.ndarray, batch_mean: np.ndarray) -> float:
    return float(np.linalg.norm(row - batch_mean) / (np.linalg.norm(batch_mean) + 1e-9))


@dataclass
class HealthResult:
    table: pd.DataFrame
    aggregates: Dict[str, float]


def compute_health_scores(
    ds: FoodSpectrumSet,
    reference_grid: Optional[np.ndarray] = None,
    batch_col: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> HealthResult:
    X = _ensure_2d(ds.x)
    wn = ds.wavenumbers
    meta = ds.metadata if isinstance(ds.metadata, pd.DataFrame) else pd.DataFrame(index=np.arange(X.shape[0]))
    batch_col = batch_col or ds.batch_col
    weights = weights or {
        "snr": 0.25,
        "spike": 0.15,
        "saturation": 0.1,
        "baseline": 0.15,
        "axis": 0.1,
        "replicate": 0.25,
    }

    batches = meta[batch_col].fillna("batch0") if batch_col and batch_col in meta else pd.Series(["batch0"] * len(meta))
    batch_means: Dict[Any, np.ndarray] = {}
    for b in batches.unique():
        mask = batches == b
        batch_means[b] = np.mean(X[mask.values], axis=0)

    rows = []
    for i, row in enumerate(X):
        snr = _snr_ratio(row)
        spike = _hampel_spike_fraction(row)
        saturation = _saturation_fraction(row)
        baseline = _baseline_lowfreq_energy(row)
        axis_cov = _axis_coverage(wn, reference_grid)
        bmean = batch_means[batches.iloc[i]]
        replicate = _replicate_distance(row, bmean)

        # Map to penalty in [0,1]
        snr_score = 1.0 / (1.0 + math.exp(-0.2 * (snr - 10)))  # higher snr better
        spike_pen = min(1.0, spike * 10)
        sat_pen = min(1.0, saturation)
        baseline_pen = min(1.0, baseline / (baseline + 10.0))
        replicate_pen = min(1.0, replicate)
        axis_pen = 1.0 - axis_cov

        penalty = (
            weights["snr"] * (1 - snr_score)
            + weights["spike"] * spike_pen
            + weights["saturation"] * sat_pen
            + weights["baseline"] * baseline_pen
            + weights["axis"] * axis_pen
            + weights["replicate"] * replicate_pen
        )
        health = float(max(0.0, min(1.0, 1.0 - penalty)))

        rows.append(
            {
                "sample_index": i,
                "snr": snr,
                "spike_fraction": spike,
                "saturation_fraction": saturation,
                "baseline_lowfreq": baseline,
                "axis_coverage": axis_cov,
                "replicate_distance": replicate,
                "health_score": health,
            }
        )

    table = pd.DataFrame(rows)
    aggregates = {
        "health_mean": float(table["health_score"].mean()),
        "health_min": float(table["health_score"].min()),
        "health_p10": float(table["health_score"].quantile(0.1)),
    }
    return HealthResult(table=table, aggregates=aggregates)


@dataclass
class OutlierResult:
    labels: np.ndarray
    scores: np.ndarray
    method: str


def detect_outliers(
    ds: FoodSpectrumSet,
    method: str = "robust_z",
    pca_components: int = 5,
    contamination: float = 0.1,
) -> OutlierResult:
    X = _ensure_2d(ds.x)

    if method == "robust_z":
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) + 1e-9
        z = np.max(np.abs((X - med) / (1.4826 * mad)), axis=1)
        thr = 4.0
        labels = (z > thr).astype(int)
        scores = z
    elif method == "mahalanobis":
        pca = PCA(n_components=min(pca_components, X.shape[1], X.shape[0] - 1))
        Z = pca.fit_transform(X)
        cov = EmpiricalCovariance().fit(Z)
        dist = cov.mahalanobis(Z)
        thr = np.percentile(dist, 95)
        labels = (dist > thr).astype(int)
        scores = dist
    elif method == "isolation_forest":
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(X)
        labels = (preds == -1).astype(int)
        scores = -clf.score_samples(X)
    elif method == "lof":
        lof = LocalOutlierFactor(contamination=contamination)
        preds = lof.fit_predict(X)
        labels = (preds == -1).astype(int)
        scores = -lof.negative_outlier_factor_
    else:
        raise ValueError(f"Unknown outlier method {method}")

    return OutlierResult(labels=labels, scores=scores, method=method)


@dataclass
class DriftResult:
    drift_score: float
    trend_slope: float
    method: str
    details: Dict[str, Any]


def detect_drift(
    ds: FoodSpectrumSet,
    time_col: Optional[str] = None,
    reference: Optional[np.ndarray] = None,
    method: str = "pca_cusum",
) -> DriftResult:
    X = _ensure_2d(ds.x)
    meta = ds.metadata if isinstance(ds.metadata, pd.DataFrame) else pd.DataFrame(index=np.arange(X.shape[0]))

    if time_col and time_col in meta:
        order = np.argsort(meta[time_col].to_numpy())
        X = X[order]
    reference if reference is not None else np.mean(X, axis=0)

    if method == "pca_cusum":
        pca = PCA(n_components=min(3, X.shape[0], X.shape[1]))
        scores = pca.fit_transform(X)[:, 0]
        mean_score = np.mean(scores)
        cusum = np.cumsum(scores - mean_score)
        drift_score = float(np.max(np.abs(cusum)))
        slope = float(np.polyfit(np.arange(len(scores)), scores, 1)[0])
        details = {"cusum_max": drift_score, "score_trend_slope": slope}
    elif method == "batch_shift":
        batches = meta.get(ds.batch_col, pd.Series(["batch0"] * len(meta))) if ds.batch_col else pd.Series(["batch0"] * len(meta))
        means = []
        for b in batches.unique():
            means.append(np.mean(X[batches == b], axis=0))
        if len(means) >= 2:
            last, prev = means[-1], means[0]
            cos = float(np.dot(last, prev) / (np.linalg.norm(last) * np.linalg.norm(prev) + 1e-9))
            drift_score = float(1 - cos)
        else:
            drift_score = 0.0
        slope = 0.0
        details = {"cosine_shift": drift_score}
    else:
        raise ValueError(f"Unknown drift method {method}")

    return DriftResult(drift_score=drift_score, trend_slope=slope, method=method, details=details)


@dataclass
class QCReport:
    health: HealthResult
    outliers: OutlierResult
    drift: DriftResult
    recommendations: str


def generate_qc_report(
    ds: FoodSpectrumSet,
    reference_grid: Optional[np.ndarray] = None,
    batch_col: Optional[str] = None,
    time_col: Optional[str] = None,
    outlier_method: str = "robust_z",
) -> QCReport:
    health = compute_health_scores(ds, reference_grid=reference_grid, batch_col=batch_col)
    outliers = detect_outliers(ds, method=outlier_method)
    drift = detect_drift(ds, time_col=time_col)

    rec = "ok"
    if health.aggregates["health_mean"] < 0.6:
        rec = "recollect"
    if drift.drift_score > 1.0 or abs(drift.trend_slope) > 0.01:
        rec = "recalibrate"
    if outliers.labels.mean() > 0.1:
        rec = "exclude"

    return QCReport(health=health, outliers=outliers, drift=drift, recommendations=rec)


__all__ = [
    "compute_health_scores",
    "detect_outliers",
    "detect_drift",
    "generate_qc_report",
    "HealthResult",
    "OutlierResult",
    "DriftResult",
    "QCReport",
]
