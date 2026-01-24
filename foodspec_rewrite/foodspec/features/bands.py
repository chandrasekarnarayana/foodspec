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

Band integration feature extractor for spectral regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class BandIntegration:
    """Extract features from spectral bands using integration or averaging.

    For each band (w_start, w_end), compute either:
    - Trapezoidal integration of area under the curve
    - Mean intensity over the band

    Parameters
    ----------
    bands : sequence of (float, float) or (float, float, str)
        Spectral bands as (start_cm1, end_cm1) or (start_cm1, end_cm1, label) tuples.
        Wavenumbers in cm^-1. Start must be < end.
    method : {"trapz", "mean"}, default "trapz"
        Integration method:
        - "trapz": Trapezoidal integration (area under curve)
        - "mean": Average intensity over band
    baseline : {"none", "linear"}, default "none"
        Baseline correction within band:
        - "none": No baseline correction
        - "linear": Subtract linear baseline from endpoints

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1000, 2000, 1001)
    >>> # Gaussian peak in band [1400, 1600]
    >>> def gauss(x, mu, amp=1.0, sigma=50.0):
    ...     return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    >>> y = gauss(x, 1500, amp=10.0)
    >>> X = np.vstack([y, y * 2])  # Two samples
    >>> bands = [(1400, 1600), (1700, 1800)]
    >>> extractor = BandIntegration(bands=bands, method="trapz", baseline="none")
    >>> features = extractor.compute(X, x)
    >>> features.shape
    (2, 2)
    >>> features.iloc[1, 0] / features.iloc[0, 0]  # Second sample has 2x intensity
    2.0
    """

    bands: Sequence[Tuple[float, ...]]
    method: str = "trapz"
    baseline: str = "none"

    def __post_init__(self):
        """Validate parameters."""
        if not self.bands:
            raise ValueError("bands must be non-empty sequence")
        if self.method not in ("trapz", "mean"):
            raise ValueError(f"method must be 'trapz' or 'mean', got '{self.method}'")
        if self.baseline not in ("none", "linear"):
            raise ValueError(f"baseline must be 'none' or 'linear', got '{self.baseline}'")
        
        # Validate each band
        for band in self.bands:
            if len(band) < 2:
                raise ValueError(f"Each band must have at least 2 elements (start, end), got {band}")
            start, end = band[0], band[1]
            if np.isnan(start) or np.isnan(end):
                raise ValueError(f"Band bounds cannot be NaN: {band}")
            if start >= end:
                raise ValueError(f"Band start must be < end, got ({start}, {end})")

    def compute(self, X: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        """Compute band integration for all samples.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_wavenumbers)
            Spectral intensity matrix.
        x : ndarray, shape (n_wavenumbers,)
            Wavenumber grid in cm^-1.

        Returns
        -------
        features : DataFrame
            One column per band with name ``band_<method>@<start>-<end>`` or
            ``band_<method>@<label>`` if label provided.
        
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

        n_samples = X.shape[0]
        n_bands = len(self.bands)
        results = np.zeros((n_samples, n_bands), dtype=float)
        column_names: List[str] = []

        for band_idx, band in enumerate(self.bands):
            start, end = band[0], band[1]
            label = band[2] if len(band) > 2 else None

            # Find indices for this band
            mask = (x >= start) & (x <= end)
            if not mask.any():
                # No points in band - use nearest points
                mid = (start + end) / 2
                idx = int(np.argmin(np.abs(x - mid)))
                indices = np.array([idx])
            else:
                indices = np.where(mask)[0]
            
            x_band = x[indices]

            for sample_idx in range(n_samples):
                y_band = X[sample_idx, indices].copy()

                if self.baseline == "linear" and len(y_band) > 1:
                    # Linear baseline from endpoints
                    y0, y1 = y_band[0], y_band[-1]
                    baseline_vals = np.linspace(y0, y1, len(y_band))
                    y_band = y_band - baseline_vals

                # Compute feature based on method
                if self.method == "trapz":
                    # Use numpy.trapezoid to avoid deprecation warnings
                    value = float(np.trapezoid(y_band, x_band))
                elif self.method == "mean":
                    value = float(np.mean(y_band))
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                
                results[sample_idx, band_idx] = value

            # Generate column name
            if label:
                column_names.append(f"band_{self.method}@{label}")
            else:
                column_names.append(f"band_{self.method}@{int(round(end))}-{int(round(start))}")

        return pd.DataFrame(results, columns=column_names)


__all__ = ["BandIntegration"]
