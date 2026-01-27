"""Additional Raman and FTIR-specific preprocessing operators."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import median_filter

from foodspec.data_objects.spectra_set import FoodSpectrumSet
from foodspec.engine.preprocessing.engine import Step


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("Expected 2D array")
    return arr


def _clone_ds(ds: FoodSpectrumSet, x: np.ndarray, wavenumbers: Optional[np.ndarray] = None) -> FoodSpectrumSet:
    """Clone FoodSpectrumSet with new data."""
    meta = ds.metadata.copy() if hasattr(ds.metadata, "copy") else ds.metadata
    return ds.__class__(x=x, wavenumbers=wavenumbers or ds.wavenumbers, metadata=meta, modality=ds.modality)


# ==================== Raman-Specific Operators ====================


class DespikeOperator(Step):
    """Remove cosmic ray spikes using median filtering.

    Parameters
    ----------
    window : int
        Median filter window size (default 5).
    threshold : float
        Z-score threshold for spike detection (default 5.0).
    """

    def __init__(self, window: int = 5, threshold: float = 5.0):
        super().__init__(name="despike", config={"window": window, "threshold": threshold})
        self.window = window
        self.threshold = threshold

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        """Apply despike to each spectrum independently."""
        X = _ensure_2d(ds.x)
        X_despike = np.zeros_like(X)

        for i, spectrum in enumerate(X):
            # Apply median filter
            median = median_filter(spectrum, size=self.window)
            # Compute residuals and z-scores
            residuals = spectrum - median
            z_scores = np.abs((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10))
            # Replace spikes
            spike_mask = z_scores > self.threshold
            X_despike[i] = np.where(spike_mask, median, spectrum)

        return _clone_ds(ds, X_despike)


class FluorescenceRemovalOperator(Step):
    """Remove Raman fluorescence background.

    Uses polynomial fitting or ALS baseline to estimate fluorescence.

    Parameters
    ----------
    method : {'poly', 'als'}
        Method for fluorescence estimation (default 'poly').
    poly_order : int
        Polynomial order if method='poly' (default 2).
    """

    def __init__(self, method: str = "poly", poly_order: int = 2):
        super().__init__(name="fluorescence_removal", config={"method": method, "poly_order": poly_order})
        self.method = method
        self.poly_order = poly_order

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        """Remove fluorescence background."""
        X = _ensure_2d(ds.x)
        X_corrected = np.zeros_like(X)

        for i, spectrum in enumerate(X):
            if self.method == "poly":
                x = np.linspace(0, 1, len(spectrum))
                coef = np.polyfit(x, spectrum, self.poly_order)
                background = np.polyval(coef, x)
            else:  # ALS
                # Simple ALS baseline
                background = self._als_baseline(spectrum)

            X_corrected[i] = spectrum - background

        return _clone_ds(ds, X_corrected)

    @staticmethod
    def _als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.001, niter: int = 10) -> np.ndarray:
        """Asymmetric least squares baseline."""
        L = len(y)
        D = np.diff(np.eye(L), 2, axis=0)
        w = np.ones(L)
        for _ in range(niter):
            W = np.diag(w)
            Z = W + lam * (D.T @ D)
            z = np.linalg.solve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z


# ==================== FTIR-Specific Operators ====================


class EMSCOperator(Step):
    """Extended Multiplicative Scatter Correction (EMSC).

    Parameters
    ----------
    reference : np.ndarray | None
        Reference spectrum. If None, uses mean spectrum.
    order : int
        Polynomial order for correction (default 2).
    """

    def __init__(self, reference: Optional[np.ndarray] = None, order: int = 2):
        super().__init__(name="emsc", config={"order": order, "has_ref": reference is not None})
        self.reference = reference
        self.order = order

    def fit(self, ds: FoodSpectrumSet) -> EMSCOperator:
        """Learn reference from mean spectrum if not provided."""
        if self.reference is None:
            X = _ensure_2d(ds.x)
            self.reference = np.mean(X, axis=0)
        return self

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        """Apply EMSC correction."""
        X = _ensure_2d(ds.x)

        if self.reference is None:
            self.reference = np.mean(X, axis=0)

        X_corrected = np.zeros_like(X)
        ref = np.asarray(self.reference, dtype=float)

        for i, spectrum in enumerate(X):
            # Build design matrix: [ref, 1, x, x^2, ...]
            x_vals = np.linspace(0, 1, len(spectrum))
            design = np.column_stack([ref] + [x_vals**j for j in range(self.order + 1)])

            # Solve least squares: design @ b = spectrum
            try:
                b = np.linalg.lstsq(design, spectrum, rcond=None)[0]
                corrected = spectrum / (b[0] * ref + np.polyval(b[1:], x_vals))
            except np.linalg.LinAlgError:
                corrected = spectrum

            X_corrected[i] = corrected

        return _clone_ds(ds, X_corrected)


class MSCOperator(Step):
    """Multiplicative Scatter Correction (MSC).

    Parameters
    ----------
    reference : np.ndarray | None
        Reference spectrum. If None, uses mean spectrum.
    """

    def __init__(self, reference: Optional[np.ndarray] = None):
        super().__init__(name="msc", config={"has_ref": reference is not None})
        self.reference = reference

    def fit(self, ds: FoodSpectrumSet) -> MSCOperator:
        """Learn reference from mean spectrum if not provided."""
        if self.reference is None:
            X = _ensure_2d(ds.x)
            self.reference = np.mean(X, axis=0)
        return self

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        """Apply MSC correction."""
        X = _ensure_2d(ds.x)

        if self.reference is None:
            self.reference = np.mean(X, axis=0)

        X_corrected = np.zeros_like(X)
        ref = np.asarray(self.reference, dtype=float)

        for i, spectrum in enumerate(X):
            # Fit: spectrum = a * reference + b
            A = np.column_stack([ref, np.ones_like(ref)])
            try:
                coef = np.linalg.lstsq(A, spectrum, rcond=None)[0]
                a, b = coef[0], coef[1]
                corrected = (spectrum - b) / (a + 1e-10)
            except np.linalg.LinAlgError:
                corrected = spectrum

            X_corrected[i] = corrected

        return _clone_ds(ds, X_corrected)


# ==================== Additional Shared Operators ====================


class AtmosphericCorrectionOperator(Step):
    """Remove common atmospheric absorption lines (CO2 around 2350, H2O around 1800-1600).

    Parameters
    ----------
    co2_window : int
        Window width around CO2 line in wavenumber units (default 50).
    water_window : int
        Window width around water lines (default 100).
    """

    def __init__(self, co2_window: int = 50, water_window: int = 100):
        super().__init__(name="atmospheric_correction", config={"co2_window": co2_window, "water_window": water_window})
        self.co2_window = co2_window
        self.water_window = water_window

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        """Apply atmospheric correction."""
        X = _ensure_2d(ds.x)
        wavenumbers = ds.wavenumbers

        # Typical atmospheric lines
        co2_line = 2350  # cm-1
        water_lines = [1600, 1800]  # cm-1

        X_corrected = X.copy()

        # Create mask for atmospheric regions
        mask = np.ones_like(wavenumbers, dtype=bool)

        # Exclude CO2 region
        co2_mask = np.abs(wavenumbers - co2_line) < self.co2_window
        mask = mask & ~co2_mask

        # Exclude water regions
        for w_line in water_lines:
            w_mask = np.abs(wavenumbers - w_line) < self.water_window
            mask = mask & ~w_mask

        # Apply mask
        if np.any(~mask):
            X_corrected[:, ~mask] = np.nan

        return _clone_ds(ds, X_corrected)


class InterpolationOperator(Step):
    """Interpolate to reference wavenumber grid.

    Parameters
    ----------
    target_grid : np.ndarray | None
        Target wavenumber grid. If None, uses original grid.
    method : {'linear', 'cubic'}
        Interpolation method (default 'linear').
    """

    def __init__(self, target_grid: Optional[np.ndarray] = None, method: str = "linear"):
        super().__init__(name="interpolation", config={"method": method, "has_grid": target_grid is not None})
        self.target_grid = target_grid
        self.method = method

    def transform(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        """Apply interpolation to target grid."""
        if self.target_grid is None:
            return ds

        X = _ensure_2d(ds.x)
        source_grid = ds.wavenumbers
        target = np.asarray(self.target_grid, dtype=float)

        X_interp = np.zeros((X.shape[0], len(target)))

        for i, spectrum in enumerate(X):
            X_interp[i] = np.interp(target, source_grid, spectrum, left=np.nan, right=np.nan)

        return _clone_ds(ds, X_interp, wavenumbers=target)


__all__ = [
    "DespikeOperator",
    "FluorescenceRemovalOperator",
    "EMSCOperator",
    "MSCOperator",
    "AtmosphericCorrectionOperator",
    "InterpolationOperator",
]
