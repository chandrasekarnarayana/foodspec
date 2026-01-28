"""Peak detection and feature extraction utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, TransformerMixin

from foodspec.features.schema import FeatureInfo, PeakSpec, normalize_assignment

__all__ = ["detect_peaks", "PeakFeatureExtractor", "extract_peak_features"]


def _apply_baseline(y: np.ndarray, rule: str) -> np.ndarray:
    rule = (rule or "none").lower()
    if rule in {"none", "off"}:
        return y
    if rule == "linear":
        baseline = np.linspace(float(y[0]), float(y[-1]), y.size)
        return y - baseline
    if rule == "min":
        return y - np.nanmin(y)
    if rule == "median":
        return y - np.nanmedian(y)
    raise ValueError(f"Unsupported baseline rule: {rule}")


def extract_peak_features(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    peaks: Sequence[PeakSpec] | Sequence[float],
    *,
    metrics: Iterable[str] = ("height", "area"),
    default_window: float = 5.0,
    default_baseline: str = "none",
) -> tuple[pd.DataFrame, list[FeatureInfo]]:
    """Extract peak features with optional baseline correction.

    Args:
        X: Spectra matrix (n_samples, n_wavenumbers).
        wavenumbers: Wavenumber axis (n_wavenumbers,).
        peaks: Peak specifications or centers.
        metrics: Metrics to compute ("height", "area", "width", "centroid", "symmetry").
        default_window: Default half-window in cm^-1 when peak specs omit window.
        default_baseline: Baseline rule when peak specs omit baseline.

    Returns:
        DataFrame of features and a list of FeatureInfo descriptors.
    """

    X = np.asarray(X, dtype=float)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_wavenumbers).")
    if wavenumbers.ndim != 1 or wavenumbers.shape[0] != X.shape[1]:
        raise ValueError("wavenumbers must be 1D and match X columns.")

    metrics = tuple(metrics)
    peak_specs: list[PeakSpec] = []
    for item in peaks:
        if isinstance(item, PeakSpec):
            peak_specs.append(item)
        else:
            center = float(item)
            peak_specs.append(
                PeakSpec(
                    name=f"peak_{center:.0f}",
                    center=center,
                    window=default_window,
                    baseline=default_baseline,
                )
            )

    n_samples = X.shape[0]
    data: dict[str, np.ndarray] = {}
    info: list[FeatureInfo] = []

    for spec in peak_specs:
        window = spec.window if spec.window else default_window
        if window <= 0:
            raise ValueError(f"Peak window must be positive for {spec.name}.")
        mask = (wavenumbers >= spec.center - window) & (wavenumbers <= spec.center + window)
        if not np.any(mask):
            for metric in metrics:
                col = spec.name if metric == "height" else f"{spec.name}_{metric}"
                data[col] = np.full(n_samples, np.nan, dtype=float)
                description = f"{metric} for peak {spec.name} near {spec.center:.0f} cm^-1."
                info.append(
                    FeatureInfo(
                        name=col,
                        ftype="peak",
                        assignment=normalize_assignment(spec.assignment),
                        description=description,
                        params={"center": spec.center, "window": window, "metric": metric},
                    )
                )
            continue

        local_w = wavenumbers[mask]
        arrays: dict[str, np.ndarray] = {metric: np.full(n_samples, np.nan, dtype=float) for metric in metrics}
        for i in range(n_samples):
            local_y = X[i, mask]
            if local_y.size == 0:
                continue
            corrected = _apply_baseline(local_y, spec.baseline or default_baseline)
            height = float(np.nanmax(corrected))
            area = float(np.trapezoid(corrected, x=local_w))
            width = centroid = symmetry = np.nan
            if "width" in metrics or "centroid" in metrics or "symmetry" in metrics:
                half = height / 2.0
                above = corrected >= half
                if np.any(above):
                    idx = np.where(above)[0]
                    width = float(local_w[idx[-1]] - local_w[idx[0]])
                total = float(np.sum(corrected))
                centroid = float(np.sum(local_w * corrected) / (total + 1e-12))
                left_mask = local_w <= centroid
                right_mask = local_w >= centroid
                left_area = (
                    float(np.trapezoid(corrected[left_mask], x=local_w[left_mask])) if np.any(left_mask) else 0.0
                )
                right_area = (
                    float(np.trapezoid(corrected[right_mask], x=local_w[right_mask])) if np.any(right_mask) else 0.0
                )
                denom = left_area + right_area + 1e-12
                symmetry = 1.0 - abs(left_area - right_area) / denom

            if "height" in metrics:
                arrays["height"][i] = height
            if "area" in metrics:
                arrays["area"][i] = area
            if "width" in metrics:
                arrays["width"][i] = width
            if "centroid" in metrics:
                arrays["centroid"][i] = centroid
            if "symmetry" in metrics:
                arrays["symmetry"][i] = symmetry

        for metric in metrics:
            col = spec.name if metric == "height" else f"{spec.name}_{metric}"
            data[col] = arrays[metric]
            description = f"{metric} for peak {spec.name} near {spec.center:.0f} cm^-1."
            info.append(
                FeatureInfo(
                    name=col,
                    ftype="peak",
                    assignment=normalize_assignment(spec.assignment),
                    description=description,
                    params={"center": spec.center, "window": window, "metric": metric},
                )
            )

    return pd.DataFrame(data), info


def detect_peaks(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    prominence: float = 0.0,
    width: Optional[float] = None,
) -> pd.DataFrame:
    """Detect peaks and return their properties.

    Args:
        x: 1D intensity array.
        wavenumbers: 1D axis array aligned with `x`.
        prominence: Minimum prominence passed to `scipy.signal.find_peaks`.
        width: Optional width parameter for `find_peaks`.

    Returns:
        A DataFrame with columns: `peak_index`, `peak_wavenumber`,
        `peak_intensity`, `prominence`, `width`.

    Raises:
        ValueError: If `x` and `wavenumbers` are not 1D or have mismatched lengths.

    Examples:
        >>> import numpy as np
        >>> from foodspec.features.peaks import detect_peaks
        >>> x = np.array([0, 1, 2, 1, 0, 3, 0])
        >>> wn = np.linspace(1000, 1600, 7)
        >>> peaks_df = detect_peaks(x, wn, prominence=0.5)
        >>> peaks_df.shape[0] > 0
        True
    """

    x = np.asarray(x, dtype=float)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if x.ndim != 1 or wavenumbers.ndim != 1:
        raise ValueError("x and wavenumbers must be 1D.")
    if x.shape[0] != wavenumbers.shape[0]:
        raise ValueError("x and wavenumbers must have the same length.")

    peak_indices, props = find_peaks(x, prominence=prominence, width=width)
    prominences = props.get("prominences", np.full_like(peak_indices, np.nan, dtype=float))
    widths = props.get("widths", np.full_like(peak_indices, np.nan, dtype=float))
    return pd.DataFrame(
        {
            "peak_index": peak_indices,
            "peak_wavenumber": wavenumbers[peak_indices],
            "peak_intensity": x[peak_indices],
            "prominence": prominences,
            "width": widths,
        }
    )


class PeakFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract peak height and area features around expected peaks."""

    def __init__(
        self,
        expected_peaks: Sequence[float],
        tolerance: float = 5.0,
        features: Sequence[str] = ("height", "area"),
    ):
        self.expected_peaks = list(expected_peaks)
        self.tolerance = tolerance
        self.features = tuple(features)
        self.feature_names_: list[str] = []

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, wavenumbers: Optional[np.ndarray] = None
    ) -> "PeakFeatureExtractor":
        self._build_feature_names()
        return self

    def transform(self, X: np.ndarray, wavenumbers: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D with shape (n_samples, n_wavenumbers).")
        if wavenumbers is None:
            raise ValueError("wavenumbers is required to extract peak features.")
        wavenumbers = np.asarray(wavenumbers, dtype=float)
        if wavenumbers.shape[0] != X.shape[1]:
            raise ValueError("wavenumbers length must match X columns.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")

        self._build_feature_names()
        feats = np.zeros((X.shape[0], len(self.feature_names_)), dtype=float)
        for i, spectrum in enumerate(X):
            col = 0
            for peak_center in self.expected_peaks:
                mask = (wavenumbers >= peak_center - self.tolerance) & (wavenumbers <= peak_center + self.tolerance)
                height = area = width = centroid = symmetry = np.nan
                if np.any(mask):
                    local_w = wavenumbers[mask]
                    local_y = spectrum[mask]
                    local_max_idx = np.argmax(local_y)
                    height = local_y[local_max_idx]
                    area = np.trapezoid(local_y, x=local_w)

                    half = height / 2.0
                    above = local_y >= half
                    if np.any(above):
                        idx = np.where(above)[0]
                        width = local_w[idx[-1]] - local_w[idx[0]]
                    centroid = float(np.sum(local_w * local_y) / (np.sum(local_y) + 1e-12))
                    left_area = (
                        np.trapezoid(local_y[local_w <= centroid], x=local_w[local_w <= centroid])
                        if np.any(local_w <= centroid)
                        else 0.0
                    )
                    right_area = (
                        np.trapezoid(local_y[local_w >= centroid], x=local_w[local_w >= centroid])
                        if np.any(local_w >= centroid)
                        else 0.0
                    )
                    denom = left_area + right_area + 1e-12
                    symmetry = 1.0 - abs(left_area - right_area) / denom

                if "height" in self.features:
                    feats[i, col] = height
                    col += 1
                if "area" in self.features:
                    feats[i, col] = area
                    col += 1
                if "width" in self.features:
                    feats[i, col] = width
                    col += 1
                if "centroid" in self.features:
                    feats[i, col] = centroid
                    col += 1
                if "symmetry" in self.features:
                    feats[i, col] = symmetry
                    col += 1

        return feats

    def get_feature_names_out(self, input_features=None):
        self._build_feature_names()
        return np.array(self.feature_names_, dtype=str)

    def _build_feature_names(self) -> None:
        names: list[str] = []
        for peak in self.expected_peaks:
            if "height" in self.features:
                names.append(f"peak_{peak}_height")
            if "area" in self.features:
                names.append(f"peak_{peak}_area")
            if "width" in self.features:
                names.append(f"peak_{peak}_width")
            if "centroid" in self.features:
                names.append(f"peak_{peak}_centroid")
            if "symmetry" in self.features:
                names.append(f"peak_{peak}_symmetry")
        self.feature_names_ = names
