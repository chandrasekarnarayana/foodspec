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

Peak ratios feature extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PeakRatios:
    """Extract ratios between peak intensities near given wavenumber pairs.

    For each pair (w1, w2), the extractor:
      1) Finds the index in `x` nearest to each target wavenumber.
      2) Uses a +/- window (in indices) around that index to locate the local
         maximum intensity for each sample.
      3) Returns the ratio of the two maxima: peak(w1) / peak(w2).

    Parameters
    ----------
    peak_pairs : sequence of (float, float)
        Target wavenumber pairs (numerators over denominators).
    window : int, default 15
        Half-window size in indices used around the nearest index to search
        for the peak maximum. Effective slice is [i - window, i + window].
    eps : float, default 1e-12
        Small constant added to denominator to avoid division by zero.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> x = np.linspace(1000, 1100, 101)
    >>> # Two simple peaks near 1030 and 1050
    >>> def gauss(x, mu, amp=1.0, sigma=2.0):
    ...     return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    >>> y = gauss(x, 1030, amp=2.0) + gauss(x, 1050, amp=1.0)
    >>> X = np.vstack([y, y * 0 + gauss(x, 1030, amp=1.0) + gauss(x, 1050, amp=2.0)])
    >>> feats = PeakRatios(peak_pairs=[(1030, 1050)], window=10)
    >>> df = feats.compute(X, x)
    >>> df.shape
    (2, 1)
    >>> 1.5 < df.iloc[0, 0] < 2.1
    True
    >>> 0.4 < df.iloc[1, 0] < 0.7
    True
    """

    peak_pairs: Sequence[Tuple[float, float]]
    window: int = 15
    eps: float = 1e-12

    def _nearest_index(self, x: np.ndarray, w: float) -> int:
        idx = int(np.argmin(np.abs(x - w)))
        return idx

    def _peak_value(self, row: np.ndarray, idx: int) -> float:
        start = max(0, idx - self.window)
        end = min(row.shape[0], idx + self.window + 1)
        segment = row[start:end]
        # Guard against empty segments (shouldn't happen with valid window)
        if segment.size == 0:
            return float(row[idx])
        return float(np.max(segment))

    def compute(self, X: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        """Compute peak ratios for all samples.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_wavenumbers)
            Spectral intensity matrix.
        x : ndarray, shape (n_wavenumbers,)
            Wavenumber grid aligned with columns of X.

        Returns
        -------
        DataFrame
            One column per ratio in the order of `peak_pairs`. Column names are
            of the form ``ratio_<w1>_<w2>`` (rounded to nearest integer).
        """

        X = np.asarray(X, dtype=float)
        x = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")
        if x.ndim != 1 or x.shape[0] != X.shape[1]:
            raise ValueError("x must be 1D and match X columns")
        if not self.peak_pairs:
            raise ValueError("peak_pairs must be a non-empty sequence of pairs")

        # Precompute nearest indices for performance
        idx_pairs: List[Tuple[int, int]] = [
            (self._nearest_index(x, w1), self._nearest_index(x, w2))
            for (w1, w2) in self.peak_pairs
        ]

        columns = [f"ratio_{int(round(w1))}_{int(round(w2))}" for (w1, w2) in self.peak_pairs]
        out = np.zeros((X.shape[0], len(self.peak_pairs)), dtype=float)

        for i, row in enumerate(X):
            values = []
            for (idx1, idx2) in idx_pairs:
                v1 = self._peak_value(row, idx1)
                v2 = self._peak_value(row, idx2)
                ratio = v1 / (v2 + self.eps)
                values.append(ratio)
            out[i, :] = values

        return pd.DataFrame(out, columns=columns)


__all__ = ["PeakRatios"]
