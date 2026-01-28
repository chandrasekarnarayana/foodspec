"""Spectral alignment methods: cross-correlation and DTW."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["CrossCorrelationAligner", "DynamicTimeWarpingAligner", "align_spectra"]


class CrossCorrelationAligner(BaseEstimator, TransformerMixin):
    """Align spectra using FFT-based cross-correlation."""

    def __init__(self, max_shift: int = 50, reference_idx: int = 0):
        """
        Initialize cross-correlation aligner.

        Parameters
        ----------
        max_shift : int, default=50
            Maximum allowed shift in wavenumber indices.
        reference_idx : int, default=0
            Index of reference spectrum (row in X).
        """
        self.max_shift = max_shift
        self.reference_idx = reference_idx
        self.reference_ = None
        self.shifts_ = None

    def fit(self, X: np.ndarray) -> CrossCorrelationAligner:
        """
        Fit aligner on reference spectrum.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_wavenumbers)
            Spectral data.

        Returns
        -------
        self
        """
        self.reference_ = X[self.reference_idx].copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Align spectra to reference using cross-correlation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_wavenumbers)
            Spectra to align.

        Returns
        -------
        X_aligned : np.ndarray, shape (n_samples, n_wavenumbers)
            Aligned spectra.
        """
        if self.reference_ is None:
            raise ValueError("Must call fit() first")

        X_aligned = np.zeros_like(X)
        self.shifts_ = []

        for i, spectrum in enumerate(X):
            # Compute cross-correlation
            xcorr = signal.correlate(spectrum, self.reference_, mode="same")

            # Find peak
            center = len(xcorr) // 2
            start = max(0, center - self.max_shift)
            end = min(len(xcorr), center + self.max_shift + 1)

            peak_idx = np.argmax(xcorr[start:end]) + start
            shift = peak_idx - center
            self.shifts_.append(shift)

            # Apply shift
            X_aligned[i] = np.roll(spectrum, shift)

        return X_aligned

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : ignored
            Present for sklearn compatibility.

        Returns
        -------
        X_transformed : np.ndarray
            Aligned spectra.
        """
        return self.fit(X).transform(X)


class DynamicTimeWarpingAligner(BaseEstimator, TransformerMixin):
    """Align spectra using Dynamic Time Warping (DTW)."""

    def __init__(self, window: int = 50, reference_idx: int = 0):
        """
        Initialize DTW aligner.

        Parameters
        ----------
        window : int, default=50
            Sakoe-Chiba band width (local window for DTW).
        reference_idx : int, default=0
            Index of reference spectrum.
        """
        self.window = window
        self.reference_idx = reference_idx
        self.reference_ = None
        self.warping_paths_ = None

    def fit(self, X: np.ndarray) -> DynamicTimeWarpingAligner:
        """Fit aligner on reference."""
        self.reference_ = X[self.reference_idx].copy()
        return self

    @staticmethod
    def _dtw_distance(s1: np.ndarray, s2: np.ndarray, window: int) -> Tuple[float, np.ndarray]:
        """
        Compute DTW distance with Sakoe-Chiba band.

        Parameters
        ----------
        s1, s2 : np.ndarray
            Sequences to compare.
        window : int
            Band width.

        Returns
        -------
        distance : float
            DTW distance.
        warping_path : np.ndarray
            Optimal warping path.
        """
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if abs(i - j) <= window:
                    cost = abs(s1[i - 1] - s2[j - 1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1],
                    )

        # Backtrack to find warping path
        i, j = n, m
        path = [(i - 1, j - 1)]
        while i > 1 or j > 1:
            tb = np.argmin(
                [
                    dtw_matrix[i - 1, j - 1] if i > 1 and j > 1 else np.inf,
                    dtw_matrix[i - 1, j] if i > 1 else np.inf,
                    dtw_matrix[i, j - 1] if j > 1 else np.inf,
                ]
            )
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                i -= 1
            else:
                j -= 1
            path.append((i - 1, j - 1))

        path = np.array(path[::-1])
        return dtw_matrix[n, m], path

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Align spectra to reference using DTW.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_wavenumbers)
            Spectra to align.

        Returns
        -------
        X_aligned : np.ndarray, shape (n_samples, n_wavenumbers)
            Warped spectra.
        """
        if self.reference_ is None:
            raise ValueError("Must call fit() first")

        X_aligned = np.zeros_like(X)
        self.warping_paths_ = []

        for i, spectrum in enumerate(X):
            # Compute DTW and warping path
            _, path = self._dtw_distance(spectrum, self.reference_, self.window)
            self.warping_paths_.append(path)

            # Resample spectrum along warping path
            resampled = np.interp(
                np.arange(len(self.reference_)),
                np.linspace(0, len(spectrum) - 1, len(spectrum)),
                spectrum,
            )
            X_aligned[i] = resampled

        return X_aligned

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : ignored
            Present for sklearn compatibility.

        Returns
        -------
        X_transformed : np.ndarray
            Warped spectra.
        """
        return self.fit(X).transform(X)


def align_spectra(
    X: np.ndarray,
    method: str = "xcorr",
    reference_idx: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Align spectra using specified method.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavenumbers)
        Spectra to align.
    method : {"xcorr", "dtw"}, default="xcorr"
        Alignment method.
    reference_idx : int, default=0
        Index of reference spectrum.
    **kwargs
        Method-specific keyword arguments.

    Returns
    -------
    X_aligned : np.ndarray
        Aligned spectra.

    Examples
    --------
    >>> X_raw = np.random.randn(10, 500)
    >>> X_aligned = align_spectra(X_raw, method="dtw", window=50)
    """
    if method == "xcorr":
        aligner = CrossCorrelationAligner(reference_idx=reference_idx, **kwargs)
    elif method == "dtw":
        aligner = DynamicTimeWarpingAligner(reference_idx=reference_idx, **kwargs)
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    return aligner.fit_transform(X)
