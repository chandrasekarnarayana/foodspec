"""Spectral alignment helpers (engine namespace)."""
from __future__ import annotations

import numpy as np

from foodspec.data_objects.spectral_dataset import harmonize_datasets


def align_spectra(reference_wavenumbers: np.ndarray, target_wavenumbers: np.ndarray, target_spectra: np.ndarray):
    """Align target spectra to a reference wavenumber grid.

    Uses the existing harmonization helper to interpolate spectra.
    """

    aligned = harmonize_datasets(reference_wavenumbers, target_wavenumbers, target_spectra)
    return aligned


__all__ = ["align_spectra"]
