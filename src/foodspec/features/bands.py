"""Band integration utilities."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from foodspec.features.schema import BandSpec, FeatureInfo, normalize_assignment

__all__ = ["integrate_bands", "compute_band_features", "extract_band_features"]


def _apply_baseline_matrix(sub_x: np.ndarray, rule: str) -> np.ndarray:
    rule = (rule or "none").lower()
    if rule in {"none", "off"}:
        return sub_x
    if rule == "linear":
        start = sub_x[:, :1]
        end = sub_x[:, -1:]
        t = np.linspace(0.0, 1.0, sub_x.shape[1])[None, :]
        baseline = start + (end - start) * t
        return sub_x - baseline
    if rule == "min":
        return sub_x - np.nanmin(sub_x, axis=1, keepdims=True)
    if rule == "median":
        return sub_x - np.nanmedian(sub_x, axis=1, keepdims=True)
    raise ValueError(f"Unsupported baseline rule: {rule}")


def compute_band_features(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    bands: Sequence[Tuple[str, float, float]] | Sequence[BandSpec],
    metrics: Iterable[str] = ("integral",),
) -> pd.DataFrame:
    """Compute band-level features (integral/mean/max/slope).

    Args:
        X: 2D array of spectra (samples × wavenumbers).
        wavenumbers: 1D array of wavenumber values matching X columns.
        bands: Sequence of (label, min_wn, max_wn) tuples or BandSpec objects defining bands.
        metrics: Feature types to compute per band ("integral", "mean", "max", "slope").

    Returns:
        DataFrame with one row per sample and columns for each band × metric.

    Raises:
        ValueError: If X is not 2D or wavenumbers shape mismatches X columns.
        ValueError: If any band has invalid range (min_wn >= max_wn).
    """

    X = np.asarray(X, dtype=float)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if wavenumbers.ndim != 1 or wavenumbers.shape[0] != X.shape[1]:
        raise ValueError("wavenumbers must be 1D and match number of columns in X.")

    metrics = list(metrics)
    single_integral = len(metrics) == 1 and metrics[0] == "integral"
    data = {}
    for item in bands:
        baseline = "none"
        if isinstance(item, BandSpec):
            label = item.name
            min_wn = item.start
            max_wn = item.end
            baseline = item.baseline
        else:
            label, min_wn, max_wn = item
        if min_wn >= max_wn:
            raise ValueError(f"Band {label} has invalid range.")
        mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
        if not np.any(mask):
            for m in metrics:
                col_name = label if (m == "integral" and single_integral) else f"{label}_{m}"
                data[col_name] = np.full(X.shape[0], np.nan)
            continue
        sub_x = X[:, mask]
        sub_w = wavenumbers[mask]
        if baseline and str(baseline).lower() not in {"none", "off"}:
            sub_x = _apply_baseline_matrix(sub_x, str(baseline))
        if "integral" in metrics:
            col_name = label if single_integral else f"{label}_integral"
            data[col_name] = np.trapezoid(sub_x, x=sub_w, axis=1)
        if "mean" in metrics:
            data[f"{label}_mean"] = np.mean(sub_x, axis=1)
        if "max" in metrics:
            data[f"{label}_max"] = np.max(sub_x, axis=1)
        if "slope" in metrics:
            data[f"{label}_slope"] = (sub_x[:, -1] - sub_x[:, 0]) / (sub_w[-1] - sub_w[0] + 1e-12)

    return pd.DataFrame(data)


def extract_band_features(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    bands: Sequence[BandSpec],
    *,
    metrics: Iterable[str] = ("integral",),
) -> tuple[pd.DataFrame, list[FeatureInfo]]:
    """Extract band features and return feature descriptors."""

    df = compute_band_features(X, wavenumbers, bands, metrics=metrics)
    info: list[FeatureInfo] = []
    metrics = tuple(metrics)
    single_integral = len(metrics) == 1 and metrics[0] == "integral"
    for band in bands:
        for metric in metrics:
            col = band.name if (metric == "integral" and single_integral) else f"{band.name}_{metric}"
            description = f"{metric} for band {band.name} ({band.start:.0f}-{band.end:.0f} cm^-1)."
            info.append(
                FeatureInfo(
                    name=col,
                    ftype="band",
                    assignment=normalize_assignment(band.assignment),
                    description=description,
                    params={"start": band.start, "end": band.end, "metric": metric},
                )
            )
    return df, info


def integrate_bands(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    bands: Sequence[Tuple[str, float, float]],
) -> pd.DataFrame:
    """Backwards-compatible wrapper: band integrals only.

    Args:
        X: 2D array of spectra (samples × wavenumbers).
        wavenumbers: 1D array of wavenumber values matching X columns.
        bands: Sequence of (label, min_wn, max_wn) tuples defining bands.

    Returns:
        DataFrame with one row per sample and one column per band (integral only).
    """

    return compute_band_features(X, wavenumbers, bands, metrics=("integral",))
