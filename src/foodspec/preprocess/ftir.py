"""Simplified FTIR-specific corrections."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from foodspec.preprocess.base import WavenumberAwareMixin

__all__ = ["AtmosphericCorrector", "SimpleATRCorrector"]


class AtmosphericCorrector(WavenumberAwareMixin, BaseEstimator, TransformerMixin):
    """Atmospheric correction using synthetic or user-provided water/CO2 bases.

    This is a simplified approach (not vendor-grade). You may supply explicit
    water/co2 basis arrays (shape n_points x n_bases); otherwise, broad Gaussian
    bases are generated at typical water/CO2 positions.
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
    """Approximate ATR correction using heuristic scaling."""

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
