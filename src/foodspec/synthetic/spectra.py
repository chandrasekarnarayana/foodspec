"""Utilities to generate simple synthetic Raman and FTIR spectra for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


@dataclass
class PeakSpec:
    """Specification for a synthetic peak or band."""

    position: float
    amplitude: float = 1.0
    width: float = 10.0
    shape: str = "gaussian"  # "gaussian" or "lorentzian"


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _lorentzian(x: np.ndarray, mu: float, gamma: float) -> np.ndarray:
    return gamma**2 / ((x - mu) ** 2 + gamma**2)


def generate_synthetic_raman_spectrum(
    peaks: Sequence[PeakSpec] | None = None,
    noise_level: float = 0.01,
    baseline_slope: float = 0.0,
    wavenumber_min: float = 400.0,
    wavenumber_max: float = 1800.0,
    num_points: int = 1401,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic Raman-like spectrum.

    Parameters
    ----------
    peaks:
        Iterable of PeakSpec defining peak positions/intensities/widths. If None, a few
        default peaks typical of edible oils are used.
    noise_level:
        Standard deviation of additive Gaussian noise.
    baseline_slope:
        Linear baseline slope (simulates fluorescence background).
    wavenumber_min, wavenumber_max, num_points:
        Grid definition for the spectrum.

    Returns
    -------
    wavenumbers, intensity : np.ndarray, np.ndarray
        Arrays of shape (num_points,) suitable for plotting or building a FoodSpectrumSet.
    """
    wn = np.linspace(wavenumber_min, wavenumber_max, num_points)
    if peaks is None:
        peaks = [
            PeakSpec(717, 0.8, 8),
            PeakSpec(1265, 1.0, 10),
            PeakSpec(1440, 1.2, 12),
            PeakSpec(1655, 1.0, 14),
        ]
    y = baseline_slope * (wn - wn.min())
    for p in peaks:
        if p.shape == "lorentzian":
            y += p.amplitude * _lorentzian(wn, p.position, p.width)
        else:
            y += p.amplitude * _gaussian(wn, p.position, p.width)
    if noise_level > 0:
        y += np.random.normal(scale=noise_level, size=wn.shape)
    return wn, y


def generate_synthetic_ftir_spectrum(
    bands: Sequence[PeakSpec] | None = None,
    noise_level: float = 0.003,
    baseline_amp: float = 0.02,
    wavenumber_min: float = 800.0,
    wavenumber_max: float = 3600.0,
    num_points: int = 2801,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic FTIR-like spectrum with broad bands.

    Parameters
    ----------
    bands:
        Iterable of PeakSpec defining band positions/intensities/widths. Defaults to
        common food bands (O-H, C-H, C=O, fingerprint).
    noise_level:
        Standard deviation of additive Gaussian noise.
    baseline_amp:
        Amplitude of a gentle cosine baseline.
    wavenumber_min, wavenumber_max, num_points:
        Grid definition for the spectrum.

    Returns
    -------
    wavenumbers, intensity : np.ndarray, np.ndarray
        Arrays of shape (num_points,) suitable for plotting or building a FoodSpectrumSet.
    """
    wn = np.linspace(wavenumber_min, wavenumber_max, num_points)
    if bands is None:
        bands = [
            PeakSpec(3300, 0.6, 80),  # O-H stretch
            PeakSpec(2925, 0.8, 60),  # C-H stretch
            PeakSpec(1740, 0.5, 40),  # C=O stretch (esters)
            PeakSpec(1450, 0.7, 35),  # CH2 bend
            PeakSpec(1050, 0.6, 30),  # C–O stretch
        ]
    baseline = baseline_amp * np.cos(wn / 200.0)
    y = baseline.copy()
    for b in bands:
        y += b.amplitude * _gaussian(wn, b.position, b.width)
    if noise_level > 0:
        y += np.random.normal(scale=noise_level, size=wn.shape)
    return wn, y
