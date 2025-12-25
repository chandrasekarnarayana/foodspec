"""
Cosmic ray (spike) detection and correction as a standalone step.

Detect spikes via robust z-score and correct by local median interpolation.
Reports per-spectrum spike counts.
"""
from __future__ import annotations

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
