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

Spectral QC metrics (SNR, clipping, entropy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodspec.qc.base import QCMetric


class SpectralQC(QCMetric):
    """Compute SNR, clipping fraction, and normalized spectral entropy."""

    def __init__(self, clip_epsilon: float = 1e-12) -> None:
        self.clip_epsilon = clip_epsilon

    def compute(self, X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers)")

        snr_list = []
        clip_list = []
        entropy_list = []

        for row in X:
            # SNR: signal std / noise std where noise is high-frequency residual
            signal_std = float(np.std(row))
            noise = row[1:] - row[:-1]
            noise_std = float(np.std(noise)) if noise.size > 0 else 0.0
            snr = signal_std / noise_std if noise_std > 0 else np.inf

            # Clipping: fraction at min or max
            rmin, rmax = float(np.min(row)), float(np.max(row))
            clip = float(np.mean((np.isclose(row, rmin, atol=self.clip_epsilon)) | (np.isclose(row, rmax, atol=self.clip_epsilon))))

            # Spectral entropy: normalized Shannon entropy of magnitude spectrum
            mag = np.abs(np.fft.rfft(row))
            mag = mag / (np.sum(mag) + 1e-12)
            entropy = -np.sum(mag * np.log2(mag + 1e-12))
            entropy_norm = entropy / np.log2(len(mag))

            snr_list.append(snr)
            clip_list.append(clip)
            entropy_list.append(float(entropy_norm))

        return pd.DataFrame({"snr": snr_list, "clip_frac": clip_list, "entropy": entropy_list})


__all__ = ["SpectralQC"]
