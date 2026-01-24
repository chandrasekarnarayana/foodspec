"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

Peak-based feature extractors: heights, areas, ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PeakRatios:
    """Extract ratios between peak intensities or areas at wavenumber pairs.

    For each pair (w1, w2), compute ratio of peak values: peak(w1) / peak(w2).
    Supports both height-based and area-based ratios.

    Parameters
    ----------
    pairs : sequence of (float, float)
        Target wavenumber pairs (numerator, denominator) in cm^-1.
    method : {"height", "area"}, default "height"
        Ratio computation method:
        - "height": Use local maximum intensity in window
        - "area": Integrate intensity in window
    window : float or None, default None
        Half-window size in wavenumber units (cm^-1). If None, uses nearest point.
    eps : float, default 1e-12
        Small constant added to denominator to avoid division by zero.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1000, 1100, 101)
    >>> def gauss(x, mu, amp=1.0, sigma=2.0):
    ...     return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    >>> y = gauss(x, 1030, amp=2.0) + gauss(x, 1050, amp=1.0)
    >>> X = np.vstack([y, y * 0 + gauss(x, 1030, amp=1.0) + gauss(x, 1050, amp=2.0)])
    >>> feats = PeakRatios(pairs=[(1030, 1050)], method="height", window=5.0)
    >>> df = feats.compute(X, x)
    >>> df.shape
    (2, 1)
    >>> 1.5 < df.iloc[0, 0] < 2.1
    True
    >>> 0.4 < df.iloc[1, 0] < 0.7
    True
    """

    pairs: Sequence[Tuple[float, float]]
    method: str = "height"
    window: float | None = None
    eps: float = 1e-12

    def __post_init__(self):
        """Validate parameters."""
        if not self.pairs:
            raise ValueError("pairs must be non-empty sequence")
        if self.method not in ("height", "area"):
            raise ValueError(f"method must be 'height' or 'area', got '{self.method}'")
        if self.window is not None and self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")
        
        # Validate pairs
        for pair in self.pairs:
            if len(pair) != 2:
                raise ValueError(f"Each pair must have exactly 2 elements, got {pair}")
            if np.isnan(pair[0]) or np.isnan(pair[1]):
                raise ValueError(f"Pair values cannot be NaN: {pair}")

    def _find_window_indices(self, x: np.ndarray, target: float) -> Tuple[int, int]:
        """Find start and end indices for window around target wavenumber."""
        if self.window is None:
            idx = int(np.argmin(np.abs(x - target)))
            return idx, idx + 1
        
        mask = np.abs(x - target) <= self.window
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            idx = int(np.argmin(np.abs(x - target)))
            return idx, idx + 1
        
        return int(indices[0]), int(indices[-1] + 1)

    def _compute_peak_value(self, row: np.ndarray, x: np.ndarray, start: int, end: int) -> float:
        """Compute peak value using configured method."""
        segment = row[start:end]
        if segment.size == 0:
            return 0.0
        
        if self.method == "height":
            return float(np.max(segment))
        elif self.method == "area":
            x_segment = x[start:end]
            return float(np.trapezoid(segment, x_segment))
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute(self, X: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        """Compute peak ratios for all samples.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_wavenumbers)
            Spectral intensity matrix.
        x : ndarray, shape (n_wavenumbers,)
            Wavenumber grid in cm^-1.

        Returns
        -------
        DataFrame
            One column per ratio. Column names are of the form
            ``ratio@<w1>/<w2>`` (wavenumbers rounded to nearest integer).
        
        Raises
        ------
        ValueError
            If X is not 2D, x doesn't match X columns, or contains NaN.
        """
        X = np.asarray(X, dtype=float)
        x = np.asarray(x, dtype=float)
        
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        if x.ndim != 1 or x.shape[0] != X.shape[1]:
            raise ValueError("x must be 1D and match X columns")
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isnan(x)):
            raise ValueError("x contains NaN values")

        # Precompute window indices for performance
        window_pairs = [
            (self._find_window_indices(x, w1), self._find_window_indices(x, w2))
            for (w1, w2) in self.pairs
        ]

        columns = [f"ratio@{int(round(w1))}/{int(round(w2))}" for (w1, w2) in self.pairs]
        out = np.zeros((X.shape[0], len(self.pairs)), dtype=float)

        for i, row in enumerate(X):
            for j, ((start1, end1), (start2, end2)) in enumerate(window_pairs):
                v1 = self._compute_peak_value(row, x, start1, end1)
                v2 = self._compute_peak_value(row, x, start2, end2)
                ratio = v1 / (v2 + self.eps)
                out[i, j] = ratio

        return pd.DataFrame(out, columns=columns)


@dataclass
class PeakHeights:
    """Extract peak heights at specified wavenumbers.

    For each target wavenumber, find the intensity within a window around
    the nearest grid point using either the maximum or mean.

    Parameters
    ----------
    peaks : sequence of float
        Target wavenumbers (cm^-1) where peaks are expected.
    window : float or None, default None
        Half-window size in wavenumber units (cm^-1). If None, uses nearest point only.
    method : {"max", "mean"}, default "max"
        Aggregation method within window: "max" for local maximum, "mean" for average.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1000, 1100, 101)
    >>> def gauss(x, mu, amp=1.0, sigma=2.0):
    ...     return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    >>> y = gauss(x, 1030, amp=5.0) + gauss(x, 1070, amp=3.0)
    >>> X = np.vstack([y, y * 0.5])
    >>> extractor = PeakHeights(peaks=[1030, 1070], window=5.0, method="max")
    >>> features = extractor.compute(X, x)
    >>> features.shape
    (2, 2)
    >>> features.columns.tolist()
    ['height@1030', 'height@1070']
    """

    peaks: Sequence[float]
    window: float | None = None
    method: str = "max"

    def __post_init__(self):
        """Validate parameters."""
        if not self.peaks:
            raise ValueError("peaks must be non-empty sequence")
        if self.method not in ("max", "mean"):
            raise ValueError(f"method must be 'max' or 'mean', got '{self.method}'")
        if self.window is not None and self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")
        if any(np.isnan(p) for p in self.peaks):
            raise ValueError("peaks cannot contain NaN values")

    def _find_window_indices(self, x: np.ndarray, target: float) -> Tuple[int, int]:
        """Find start and end indices for window around target wavenumber.
        
        Returns (start, end) where end is exclusive (Python slice convention).
        """
        if self.window is None:
            # Just nearest point
            idx = int(np.argmin(np.abs(x - target)))
            return idx, idx + 1
        
        # Find all points within window
        mask = np.abs(x - target) <= self.window
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            # Fall back to nearest point if no points in window
            idx = int(np.argmin(np.abs(x - target)))
            return idx, idx + 1
        
        return int(indices[0]), int(indices[-1] + 1)

    def _peak_value(self, row: np.ndarray, start: int, end: int) -> float:
        """Extract peak value from segment using configured method."""
        segment = row[start:end]
        if segment.size == 0:
            return 0.0
        
        if self.method == "max":
            return float(np.max(segment))
        elif self.method == "mean":
            return float(np.mean(segment))
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute(self, X: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        """Compute peak heights for all samples.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_wavenumbers)
            Spectral intensity matrix.
        x : ndarray, shape (n_wavenumbers,)
            Wavenumber grid in cm^-1.

        Returns
        -------
        features : DataFrame
            Columns: height@<wavenumber> for each peak location.
        
        Raises
        ------
        ValueError
            If X is not 2D, x doesn't match X columns, or contains NaN.
        """
        X = np.asarray(X, dtype=float)
        x = np.asarray(x, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        if x.ndim != 1 or x.shape[0] != X.shape[1]:
            raise ValueError("x must be 1D and match X columns")
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isnan(x)):
            raise ValueError("x contains NaN values")

        # Precompute window indices for each peak
        window_indices = [self._find_window_indices(x, w) for w in self.peaks]
        columns = [f"height@{int(round(w))}" for w in self.peaks]
        
        n_samples = X.shape[0]
        n_peaks = len(self.peaks)
        results = np.zeros((n_samples, n_peaks), dtype=float)

        for i, row in enumerate(X):
            for j, (start, end) in enumerate(window_indices):
                results[i, j] = self._peak_value(row, start, end)

        return pd.DataFrame(results, columns=columns)


@dataclass
class PeakAreas:
    """Extract peak areas by integrating spectral bands.

    For each band (low, high), integrate intensity over that wavenumber range
    using trapezoidal rule, optionally subtracting a linear baseline.

    Parameters
    ----------
    bands : sequence of (float, float) or (float, float, str)
        Band definitions as (low_wn, high_wn) or (low_wn, high_wn, label).
        Wavenumbers in cm^-1. Low < high required.
    baseline : {"none", "linear"}, default "linear"
        Baseline correction method. "linear" fits line between endpoints;
        "none" performs no baseline correction.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1000, 1100, 101)
    >>> def gauss(x, mu, amp=1.0, sigma=2.0):
    ...     return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    >>> y = gauss(x, 1050, amp=10.0, sigma=5.0)
    >>> X = np.vstack([y, y * 2])
    >>> extractor = PeakAreas(bands=[(1030, 1070)], baseline="linear")
    >>> features = extractor.compute(X, x)
    >>> features.iloc[1, 0] / features.iloc[0, 0]  # Second sample 2x area
    2.0
    """

    bands: Sequence[Tuple[float, ...]]
    baseline: str = "linear"

    def __post_init__(self):
        """Validate parameters."""
        if not self.bands:
            raise ValueError("bands must be non-empty sequence")
        if self.baseline not in ("none", "linear"):
            raise ValueError(f"baseline must be 'none' or 'linear', got '{self.baseline}'")
        
        # Validate each band
        for band in self.bands:
            if len(band) < 2:
                raise ValueError(f"Each band must have at least 2 elements (low, high), got {band}")
            low, high = band[0], band[1]
            if np.isnan(low) or np.isnan(high):
                raise ValueError(f"Band bounds cannot be NaN: {band}")
            if low >= high:
                raise ValueError(f"Band low must be < high, got ({low}, {high})")

    def _find_band_indices(self, x: np.ndarray, low: float, high: float) -> Tuple[int, int]:
        """Find start and end indices for band [low, high].
        
        Returns (start, end) where end is exclusive.
        """
        mask = (x >= low) & (x <= high)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            # No points in range - find nearest point to midpoint
            mid = (low + high) / 2
            idx = int(np.argmin(np.abs(x - mid)))
            return idx, idx + 1
        
        return int(indices[0]), int(indices[-1] + 1)

    def compute(self, X: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        """Compute peak areas for all samples.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_wavenumbers)
            Spectral intensity matrix.
        x : ndarray, shape (n_wavenumbers,)
            Wavenumber grid in cm^-1.

        Returns
        -------
        features : DataFrame
            Columns: area@<low>-<high> or area@<label> for each band.
        
        Raises
        ------
        ValueError
            If X is not 2D, x doesn't match X columns, or contains NaN.
        """
        X = np.asarray(X, dtype=float)
        x = np.asarray(x, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        if x.ndim != 1 or x.shape[0] != X.shape[1]:
            raise ValueError("x must be 1D and match X columns")
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isnan(x)):
            raise ValueError("x contains NaN values")

        # Precompute band indices and column names
        band_indices = []
        columns = []
        for band in self.bands:
            low, high = band[0], band[1]
            label = band[2] if len(band) > 2 else None
            
            band_indices.append(self._find_band_indices(x, low, high))
            
            if label:
                columns.append(f"area@{label}")
            else:
                columns.append(f"area@{int(round(high))}-{int(round(low))}")
        
        n_samples = X.shape[0]
        n_bands = len(self.bands)
        results = np.zeros((n_samples, n_bands), dtype=float)

        for i, row in enumerate(X):
            for j, (start, end) in enumerate(band_indices):
                x_segment = x[start:end]
                y_segment = row[start:end]

                if self.baseline == "linear" and len(y_segment) > 1:
                    # Linear baseline from endpoints
                    y0, y1 = y_segment[0], y_segment[-1]
                    baseline = np.linspace(y0, y1, len(y_segment))
                    y_segment = y_segment - baseline

                area = float(np.trapezoid(y_segment, x_segment))
                results[i, j] = area

        return pd.DataFrame(results, columns=columns)


__all__ = ["PeakRatios", "PeakHeights", "PeakAreas"]
