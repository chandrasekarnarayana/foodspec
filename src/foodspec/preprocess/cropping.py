"""Cropping utilities for spectral ranges."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator

from foodspec.core.dataset import FoodSpectrumSet

__all__ = ["RangeCropper", "crop_spectrum_set"]


class RangeCropper(BaseEstimator):
    """Crop spectra to a specified wavenumber range.

    Args:
        min_wn: Minimum wavenumber (inclusive).
        max_wn: Maximum wavenumber (inclusive).

    Raises:
        ValueError: If `min_wn >= max_wn` or no points fall in the range.

    Examples:
        >>> from foodspec.preprocess import RangeCropper
        >>> import numpy as np
        >>> X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> wn = np.array([1000, 1100, 1200, 1300])
        >>> cropper = RangeCropper(min_wn=1050, max_wn=1250)
        >>> X_crop, wn_crop = cropper.transform(X, wn)
        >>> wn_crop.tolist()
        [1100, 1200]
    """

    def __init__(self, min_wn: float, max_wn: float):
        if min_wn >= max_wn:
            raise ValueError("min_wn must be less than max_wn.")
        self.min_wn = min_wn
        self.max_wn = max_wn

    def fit(self, X: np.ndarray, y=None, wavenumbers: np.ndarray | None = None):
        return self

    def transform(self, X: np.ndarray, wavenumbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    """Crop a `FoodSpectrumSet` to a wavenumber range.

    Args:
        spectra: Input dataset.
        min_wn: Minimum wavenumber (inclusive).
        max_wn: Maximum wavenumber (inclusive).

    Returns:
        A new `FoodSpectrumSet` with cropped spectra and wavenumbers.

    Raises:
        ValueError: If no wavenumbers fall within the range.

    Examples:
        >>> from foodspec.preprocess import crop_spectrum_set
        >>> from foodspec.core.dataset import FoodSpectrumSet
        >>> import numpy as np
        >>> import pandas as pd
        >>> spectra = FoodSpectrumSet(
        ...     x=np.ones((2, 4)),
        ...     wavenumbers=np.array([1000, 1100, 1200, 1300]),
        ...     metadata=pd.DataFrame({"sample_id": ["a", "b"]}),
        ...     modality="raman"
        ... )
        >>> cropped = crop_spectrum_set(spectra, 1050, 1250)
        >>> len(cropped.wavenumbers)
        2
    """

    cropper = RangeCropper(min_wn=min_wn, max_wn=max_wn)
    x_cropped, wn_cropped = cropper.transform(spectra.x, spectra.wavenumbers)
    return FoodSpectrumSet(
        x=x_cropped,
        wavenumbers=wn_cropped,
        metadata=spectra.metadata.copy(),
        modality=spectra.modality,
    )
