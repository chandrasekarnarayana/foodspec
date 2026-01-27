from __future__ import annotations

"""
Cosmic ray (spike) detection and correction as a standalone step.

Detect spikes via robust z-score and correct by local median interpolation.
Reports per-spectrum spike counts.
"""


from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


def _detect_spikes(y: np.ndarray, window: int = 5, zscore_thresh: float = 8.0) -> np.ndarray:
    median = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    diff = y - median
    mad = np.median(np.abs(diff)) + 1e-12
    z = np.abs(diff) / mad
    return z > zscore_thresh


def _correct_spikes(y: np.ndarray, window: int = 5) -> np.ndarray:
    median = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    return median


@dataclass
class CosmicRayReport:
    total_spikes: int
    spikes_per_spectrum: List[int]


def correct_cosmic_rays(X: np.ndarray, window: int = 5, zscore_thresh: float = 8.0) -> (np.ndarray, CosmicRayReport):
    """Detect and correct cosmic ray spikes in spectra.

    Uses robust z-score computed from local median and MAD (median absolute
    deviation) to detect spikes. Corrects by replacing spike values with local
    median.

    Args:
        X: Spectral data array (n_samples, n_wavenumbers).
        window: Rolling window size for spike detection.
        zscore_thresh: Threshold for spike detection (default 8.0).

    Returns:
        A tuple `(X_corrected, report)` where `report` is a `CosmicRayReport`
        containing total spike count and per-spectrum spike counts.

    Examples:
        >>> import numpy as np
        >>> from foodspec.preprocess.spikes import correct_cosmic_rays
        >>> X = np.ones((2, 50))
        >>> X[0, 25] = 100  # simulate spike
        >>> X_corr, report = correct_cosmic_rays(X, window=5, zscore_thresh=5.0)
        >>> report.total_spikes > 0
        True
    """
    Xc = X.copy()
    per_spec: List[int] = []
    for i in range(Xc.shape[0]):
        y = Xc[i]
        mask = _detect_spikes(y, window=window, zscore_thresh=zscore_thresh)
        per_spec.append(int(mask.sum()))
        if mask.any():
            y_corr = _correct_spikes(y, window=window)
            Xc[i, mask] = y_corr[mask]
    return Xc, CosmicRayReport(total_spikes=int(sum(per_spec)), spikes_per_spectrum=per_spec)
