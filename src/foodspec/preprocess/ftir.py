"""Simplified FTIR-specific corrections."""
from __future__ import annotations


from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from foodspec.preprocess.base import WavenumberAwareMixin

__all__ = ["AtmosphericCorrector", "SimpleATRCorrector"]


class AtmosphericCorrector(WavenumberAwareMixin, BaseEstimator, TransformerMixin):
    """Atmospheric correction for FTIR spectra.

    Uses synthetic or user-provided water/CO₂ basis functions to remove
    atmospheric absorption features. This is a simplified approach; for
    production, consider vendor-provided atmospheric references.

    Args:
        alpha_water: Scaling factor for water basis (default 1.0).
        alpha_co2: Scaling factor for CO₂ basis (default 1.0).
        water_center: Center wavenumber for water absorption (default 1900 cm⁻¹).
        co2_center: Center wavenumber for CO₂ absorption (default 2350 cm⁻¹).
        width: Width of Gaussian basis functions (default 30 cm⁻¹).
        water_basis: Optional explicit water basis array.
        co2_basis: Optional explicit CO₂ basis array.
        normalize_bases: Whether to normalize bases to unit norm (default True).

    Examples:
        >>> from foodspec.preprocess.ftir import AtmosphericCorrector
        >>> import numpy as np
        >>> X = np.random.randn(5, 100)
        >>> wn = np.linspace(1000, 3000, 100)
        >>> corrector = AtmosphericCorrector()
        >>> X_corr = corrector.fit_transform(X, wavenumbers=wn)
        >>> X_corr.shape == X.shape
        True
    """

    def __init__(
        self,
        alpha_water: float = 1.0,
        alpha_co2: float = 1.0,
        water_center: float = 1900.0,
        co2_center: float = 2350.0,
        width: float = 30.0,
        water_basis: Optional[np.ndarray] = None,
        co2_basis: Optional[np.ndarray] = None,
        normalize_bases: bool = True,
    ):
        self.alpha_water = alpha_water
        self.alpha_co2 = alpha_co2
        self.water_center = water_center
        self.co2_center = co2_center
        self.width = width
        self.water_basis = water_basis
        self.co2_basis = co2_basis
        self.normalize_bases = normalize_bases

    def fit(self, X, y=None, wavenumbers: Optional[np.ndarray] = None):
        if wavenumbers is not None:
            self.set_wavenumbers(wavenumbers)
        self._assert_wavenumbers_set()
        bases = self._build_bases(self.wavenumbers_)
        if self.normalize_bases:
            norms = np.linalg.norm(bases, axis=0, keepdims=True)
            norms = np.maximum(norms, np.finfo(float).eps)
            bases = bases / norms
        self._bases = bases
        return self

    def transform(self, X):
        self._assert_wavenumbers_set()
        X = np.asarray(X, dtype=float)
        bases = self._bases
        BtB = bases.T @ bases
        pseudo = np.linalg.pinv(BtB) @ bases.T
        corrected = []
        for spectrum in X:
            coeffs = pseudo @ spectrum
            resid = spectrum - bases @ coeffs
            corrected.append(resid)
        return np.vstack(corrected)

    def _build_bases(self, wn: np.ndarray) -> np.ndarray:
        if self.water_basis is not None or self.co2_basis is not None:
            parts = []
            if self.water_basis is not None:
                parts.append(np.asarray(self.water_basis, dtype=float))
            if self.co2_basis is not None:
                parts.append(np.asarray(self.co2_basis, dtype=float))
            return np.column_stack(parts)
        water = self.alpha_water * np.exp(-0.5 * ((wn - self.water_center) / self.width) ** 2)
        co2 = self.alpha_co2 * np.exp(-0.5 * ((wn - self.co2_center) / self.width) ** 2)
        return np.vstack([water, co2]).T


class SimpleATRCorrector(WavenumberAwareMixin, BaseEstimator, TransformerMixin):
    """Approximate ATR (Attenuated Total Reflectance) correction.

    Applies a heuristic scaling based on refractive indices and angle of
    incidence to compensate for wavelength-dependent penetration depth in
    ATR-FTIR.

    Args:
        refractive_index_sample: Sample refractive index (default 1.5).
        refractive_index_crystal: Crystal refractive index (default 2.4).
        angle_of_incidence: Angle of incidence in degrees (default 45.0).
        wavenumber_scale: Scaling model (default "linear").

    Examples:
        >>> from foodspec.preprocess.ftir import SimpleATRCorrector
        >>> import numpy as np
        >>> X = np.random.randn(3, 80)
        >>> wn = np.linspace(600, 4000, 80)
        >>> corrector = SimpleATRCorrector()
        >>> X_corr = corrector.fit_transform(X, wavenumbers=wn)
        >>> X_corr.shape == X.shape
        True
    """

    def __init__(
        self,
        refractive_index_sample: float = 1.5,
        refractive_index_crystal: float = 2.4,
        angle_of_incidence: float = 45.0,
        wavenumber_scale: str = "linear",
    ):
        self.refractive_index_sample = refractive_index_sample
        self.refractive_index_crystal = refractive_index_crystal
        self.angle_of_incidence = angle_of_incidence
        self.wavenumber_scale = wavenumber_scale

    def fit(self, X, y=None, wavenumbers: Optional[np.ndarray] = None):
        if wavenumbers is not None:
            self.set_wavenumbers(wavenumbers)
        self._assert_wavenumbers_set()
        self._scale = self._compute_scale(self.wavenumbers_)
        return self

    def transform(self, X):
        self._assert_wavenumbers_set()
        X = np.asarray(X, dtype=float)
        return X * self._scale

    def _compute_scale(self, wn: np.ndarray) -> np.ndarray:
        ratio = self.refractive_index_sample / self.refractive_index_crystal
        angle_factor = 1.0 + 0.01 * (self.angle_of_incidence - 45.0)
        scale = 1.0 / (1.0 + angle_factor * ratio * (wn / wn.max()))
        return scale
