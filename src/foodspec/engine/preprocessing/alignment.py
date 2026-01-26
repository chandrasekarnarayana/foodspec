"""Spectral alignment helpers (engine namespace)."""
from __future__ import annotations

import numpy as np

def align_spectra(
    reference_wavenumbers: np.ndarray,
    target_wavenumbers: np.ndarray,
    target_spectra: np.ndarray,
) -> np.ndarray:
    """Align target spectra to a reference wavenumber grid via interpolation."""
    target = np.asarray(target_wavenumbers, dtype=float)
    ref = np.asarray(reference_wavenumbers, dtype=float)
    X = np.asarray(target_spectra, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    aligned = np.vstack([np.interp(ref, target, row) for row in X])
    return aligned


__all__ = ["align_spectra"]
