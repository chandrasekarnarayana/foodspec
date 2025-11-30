"""Cropping utilities for spectral ranges."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from foodspec.core.dataset import FoodSpectrumSet
from sklearn.base import BaseEstimator

__all__ = ["RangeCropper", "crop_spectrum_set"]


class RangeCropper(BaseEstimator):
    """Crop spectra to a specified wavenumber range."""

    def __init__(self, min_wn: float, max_wn: float):
        if min_wn >= max_wn:
            raise ValueError("min_wn must be less than max_wn.")
        self.min_wn = min_wn
        self.max_wn = max_wn

    def fit(self, X: np.ndarray, y=None, wavenumbers: np.ndarray | None = None):
        return self

    def transform(
        self, X: np.ndarray, wavenumbers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        wavenumbers = np.asarray(wavenumbers, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        if wavenumbers.ndim != 1 or wavenumbers.shape[0] != X.shape[1]:
            raise ValueError("wavenumbers must be 1D and match columns of X.")

        mask = (wavenumbers >= self.min_wn) & (wavenumbers <= self.max_wn)
        if not np.any(mask):
            raise ValueError("No wavenumbers within the specified range.")
        return X[:, mask], wavenumbers[mask]


def crop_spectrum_set(spectra: FoodSpectrumSet, min_wn: float, max_wn: float) -> FoodSpectrumSet:
    """Crop a FoodSpectrumSet to a wavenumber range."""

    cropper = RangeCropper(min_wn=min_wn, max_wn=max_wn)
    x_cropped, wn_cropped = cropper.transform(spectra.x, spectra.wavenumbers)
    return FoodSpectrumSet(
        x=x_cropped,
        wavenumbers=wn_cropped,
        metadata=spectra.metadata.copy(),
        modality=spectra.modality,
    )
