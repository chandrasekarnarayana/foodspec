"""Peak detection and feature extraction utilities."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["detect_peaks", "PeakFeatureExtractor"]


def detect_peaks(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    prominence: float = 0.0,
    width: Optional[float] = None,
) -> pd.DataFrame:
    """Detect peaks and return their properties.

    Parameters
    ----------
    x :
        1D intensity array.
    wavenumbers :
        1D axis array aligned with ``x``.
    prominence :
        Minimum prominence passed to ``scipy.signal.find_peaks``.
    width :
        Optional width parameter for ``find_peaks``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``peak_index``, ``peak_wavenumber``, ``peak_intensity``,
        ``prominence``, ``width``.
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
                    left_area = np.trapezoid(local_y[local_w <= centroid], x=local_w[local_w <= centroid]) if np.any(local_w <= centroid) else 0.0
                    right_area = np.trapezoid(local_y[local_w >= centroid], x=local_w[local_w >= centroid]) if np.any(local_w >= centroid) else 0.0
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
