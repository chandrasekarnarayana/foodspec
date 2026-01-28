"""
Digital Twin Simulator for Spectroscopy.

This module provides realistic simulation of spectroscopic data, including
noise models, instrument response functions, and domain shift generators.
Enables validation, benchmarking, and synthetic data augmentation.

Key Features:
- Multiple noise models (Gaussian, Poisson, multiplicative, mixed)
- Instrument response simulation (resolution, baseline, drift)
- Domain shift generators (temperature, concentration, instrument transfer)
- Realistic spectral generation from pure components
- Validation dataset creation

References:
    [1] Rinnan et al. (2009). Review of the most common pre-processing
        techniques for near-infrared spectra. TrAC Trends Anal. Chem., 28(10).
    [2] Barnes et al. (1989). Standard normal variate transformation and
        de-trending of near-infrared diffuse reflectance spectra.
        Applied Spectroscopy, 43(5), 772-777.
    [3] Feudale et al. (2002). Transfer of multivariate calibration models:
        a review. Chemometrics and Intelligent Laboratory Systems, 64(2), 181-192.

Example:
    >>> from foodspec.simulation import SpectraSimulator, NoiseModel
    >>> import numpy as np
    >>>
    >>> # Create simulator with realistic noise
    >>> sim = SpectraSimulator(n_wavelengths=200, random_state=42)
    >>> sim.add_noise_model(NoiseModel('gaussian', std=0.01))
    >>> sim.add_noise_model(NoiseModel('poisson', scale=0.005))
    >>>
    >>> # Generate synthetic dataset
    >>> X, y, metadata = sim.generate_mixture_dataset(
    ...     n_samples=100, n_components=3
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class NoiseModel:
    """
    Noise model configuration.

    Parameters
    ----------
    noise_type : {'gaussian', 'poisson', 'multiplicative', 'mixed'}
        Type of noise to apply.

    std : float, optional
        Standard deviation (for Gaussian noise).

    scale : float, optional
        Scaling factor (for Poisson or multiplicative noise).

    snr_db : float, optional
        Target signal-to-noise ratio in dB.

    Example
    -------
    >>> noise = NoiseModel('gaussian', std=0.01)
    >>> noise_mixed = NoiseModel('mixed', std=0.01, scale=0.005)
    """

    noise_type: Literal["gaussian", "poisson", "multiplicative", "mixed"]
    std: Optional[float] = None
    scale: Optional[float] = None
    snr_db: Optional[float] = None


@dataclass
class InstrumentModel:
    """
    Instrument response configuration.

    Parameters
    ----------
    resolution : float, default=1.0
        Spectral resolution (FWHM in wavelength units).

    baseline_drift : float, default=0.0
        Linear baseline drift coefficient.

    baseline_curve : float, default=0.0
        Quadratic baseline curvature.

    wavelength_shift : float, default=0.0
        Systematic wavelength shift.

    intensity_scale : float, default=1.0
        Overall intensity scaling factor.

    Example
    -------
    >>> instrument = InstrumentModel(
    ...     resolution=2.0,  # 2 nm FWHM
    ...     baseline_drift=0.01,
    ...     wavelength_shift=0.5
    ... )
    """

    resolution: float = 1.0
    baseline_drift: float = 0.0
    baseline_curve: float = 0.0
    wavelength_shift: float = 0.0
    intensity_scale: float = 1.0


class SpectraSimulator:
    """
    Realistic spectral simulator with noise, instrument response, and domain shifts.

    Parameters
    ----------
    n_wavelengths : int, default=200
        Number of wavelength points.

    wavelength_range : tuple, optional
        (start, end) wavelength range in nm. If None, uses indices.

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    wavelengths : ndarray
        Wavelength array.

    noise_models : list
        Active noise models.

    instrument_model : InstrumentModel
        Instrument response configuration.

    Example
    -------
    >>> sim = SpectraSimulator(n_wavelengths=200, wavelength_range=(400, 2500))
    >>> X_train, y_train, meta = sim.generate_mixture_dataset(n_samples=100)
    """

    def __init__(
        self,
        n_wavelengths: int = 200,
        wavelength_range: Optional[Tuple[float, float]] = None,
        random_state: Optional[int] = None,
    ):
        self.n_wavelengths = n_wavelengths
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if wavelength_range is not None:
            self.wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wavelengths)
        else:
            self.wavelengths = np.arange(n_wavelengths)

        self.noise_models: List[NoiseModel] = []
        self.instrument_model = InstrumentModel()

    def add_noise_model(self, noise_model: NoiseModel):
        """Add a noise model to the simulator."""
        self.noise_models.append(noise_model)

    def set_instrument_model(self, instrument_model: InstrumentModel):
        """Set instrument response model."""
        self.instrument_model = instrument_model

    def generate_pure_component(
        self,
        peak_positions: List[float],
        peak_intensities: Optional[List[float]] = None,
        peak_widths: Optional[List[float]] = None,
        baseline: float = 0.0,
    ) -> np.ndarray:
        """
        Generate a pure component spectrum with Gaussian peaks.

        Parameters
        ----------
        peak_positions : list of float
            Peak positions (in wavelength units or indices).

        peak_intensities : list of float, optional
            Peak intensities. If None, random intensities in [0.5, 1.0].

        peak_widths : list of float, optional
            Peak widths (FWHM). If None, random widths in [2, 5].

        baseline : float, default=0.0
            Baseline offset.

        Returns
        -------
        spectrum : ndarray of shape (n_wavelengths,)
            Pure component spectrum.
        """
        n_peaks = len(peak_positions)

        if peak_intensities is None:
            peak_intensities = self.rng.uniform(0.5, 1.0, n_peaks)

        if peak_widths is None:
            peak_widths = self.rng.uniform(2.0, 5.0, n_peaks)

        spectrum = np.full(self.n_wavelengths, baseline)

        for pos, intensity, width in zip(peak_positions, peak_intensities, peak_widths):
            # Gaussian peak: I * exp(-(λ - λ0)² / (2σ²))
            sigma = width / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma
            gaussian = intensity * np.exp(-((self.wavelengths - pos) ** 2) / (2 * sigma**2))
            spectrum += gaussian

        return spectrum

    def generate_mixture_dataset(
        self,
        n_samples: int,
        n_components: int = 3,
        concentration_range: Tuple[float, float] = (0.0, 1.0),
        normalize_concentrations: bool = True,
        apply_noise: bool = True,
        apply_instrument: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate synthetic mixture dataset.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        n_components : int, default=3
            Number of pure components.

        concentration_range : tuple, default=(0.0, 1.0)
            Range for concentration values.

        normalize_concentrations : bool, default=True
            Whether to normalize concentrations to sum to 1.

        apply_noise : bool, default=True
            Whether to apply noise models.

        apply_instrument : bool, default=True
            Whether to apply instrument response.

        Returns
        -------
        X : ndarray of shape (n_samples, n_wavelengths)
            Mixture spectra.

        y : ndarray of shape (n_samples, n_components)
            True concentrations.

        metadata : dict
            Generation metadata (pure_spectra, noise_levels, etc.).
        """
        # Generate pure component spectra
        pure_spectra = []
        for i in range(n_components):
            # Random peak positions across wavelength range
            n_peaks = self.rng.integers(2, 6)
            peak_positions = self.rng.uniform(self.wavelengths.min() + 20, self.wavelengths.max() - 20, n_peaks)
            spectrum = self.generate_pure_component(peak_positions)
            pure_spectra.append(spectrum)

        pure_spectra = np.array(pure_spectra)  # Shape: (n_components, n_wavelengths)

        # Generate concentration profiles
        concentrations = self.rng.uniform(concentration_range[0], concentration_range[1], (n_samples, n_components))

        if normalize_concentrations:
            concentrations = concentrations / concentrations.sum(axis=1, keepdims=True)

        # Generate mixtures via Beer-Lambert law
        X = concentrations @ pure_spectra  # Shape: (n_samples, n_wavelengths)

        # Apply instrument response
        if apply_instrument:
            X = self._apply_instrument_response(X)

        # Apply noise
        if apply_noise:
            X = self._apply_noise(X)

        metadata = {
            "pure_spectra": pure_spectra,
            "concentrations": concentrations,
            "noise_models": self.noise_models,
            "instrument_model": self.instrument_model,
        }

        return X, concentrations, metadata

    def _apply_instrument_response(self, X: np.ndarray) -> np.ndarray:
        """Apply instrument response function."""
        X_inst = X.copy()
        n_samples = X.shape[0]

        # Resolution (Gaussian smoothing)
        if self.instrument_model.resolution > 0:
            sigma = self.instrument_model.resolution / (2 * np.sqrt(2 * np.log(2)))
            for i in range(n_samples):
                X_inst[i] = signal.gaussian_filter1d(X_inst[i], sigma)

        # Baseline drift (linear + quadratic)
        if self.instrument_model.baseline_drift != 0 or self.instrument_model.baseline_curve != 0:
            x = np.arange(self.n_wavelengths)
            x_norm = (x - x.mean()) / x.std()
            baseline = self.instrument_model.baseline_drift * x_norm + self.instrument_model.baseline_curve * x_norm**2
            X_inst += baseline

        # Wavelength shift (not implemented - would require interpolation)
        # Intensity scaling
        X_inst *= self.instrument_model.intensity_scale

        return X_inst

    def _apply_noise(self, X: np.ndarray) -> np.ndarray:
        """Apply all configured noise models."""
        X_noisy = X.copy()

        for noise_model in self.noise_models:
            X_noisy = self._apply_single_noise(X_noisy, noise_model)

        return X_noisy

    def _apply_single_noise(self, X: np.ndarray, noise_model: NoiseModel) -> np.ndarray:
        """Apply a single noise model."""
        if noise_model.noise_type == "gaussian":
            std = noise_model.std if noise_model.std is not None else 0.01
            noise = self.rng.normal(0, std, X.shape)
            return X + noise

        elif noise_model.noise_type == "poisson":
            # Poisson noise: σ² = μ (shot noise)
            scale = noise_model.scale if noise_model.scale is not None else 0.01
            # Ensure non-negative for Poisson
            X_scaled = np.maximum(X, 0) / scale
            X_poisson = self.rng.poisson(X_scaled) * scale
            return X_poisson

        elif noise_model.noise_type == "multiplicative":
            # Multiplicative noise: X * (1 + ε)
            scale = noise_model.scale if noise_model.scale is not None else 0.01
            noise = self.rng.normal(1.0, scale, X.shape)
            return X * noise

        elif noise_model.noise_type == "mixed":
            # Combination of Gaussian + Poisson
            std = noise_model.std if noise_model.std is not None else 0.01
            scale = noise_model.scale if noise_model.scale is not None else 0.005

            X_gaussian = X + self.rng.normal(0, std, X.shape)
            X_scaled = np.maximum(X_gaussian, 0) / scale
            X_mixed = self.rng.poisson(X_scaled) * scale

            return X_mixed

        else:
            return X


class DomainShiftGenerator:
    """
    Generate domain shifts for transfer learning evaluation.

    Simulates realistic domain shifts between training and test data:
    - Temperature variations
    - Concentration range shifts
    - Instrument-to-instrument transfer
    - Time-dependent drift

    Parameters
    ----------
    shift_type : {'temperature', 'concentration', 'instrument', 'time'}
        Type of domain shift to generate.

    magnitude : float, default=1.0
        Magnitude of the shift (interpretation depends on shift_type).

    random_state : int, optional
        Random seed.

    Example
    -------
    >>> from foodspec.simulation import DomainShiftGenerator
    >>>
    >>> # Generate temperature-shifted spectra
    >>> shift_gen = DomainShiftGenerator('temperature', magnitude=10.0)
    >>> X_shifted = shift_gen.apply_shift(X_original, wavelengths)
    """

    def __init__(
        self,
        shift_type: Literal["temperature", "concentration", "instrument", "time"],
        magnitude: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.shift_type = shift_type
        self.magnitude = magnitude
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def apply_shift(
        self,
        X: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply domain shift to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Original spectra.

        wavelengths : ndarray, optional
            Wavelength array (required for some shift types).

        Returns
        -------
        X_shifted : ndarray
            Domain-shifted spectra.
        """
        if self.shift_type == "temperature":
            return self._apply_temperature_shift(X, wavelengths)
        elif self.shift_type == "concentration":
            return self._apply_concentration_shift(X)
        elif self.shift_type == "instrument":
            return self._apply_instrument_shift(X)
        elif self.shift_type == "time":
            return self._apply_time_drift(X)
        else:
            raise ValueError(f"Unknown shift type: {self.shift_type}")

    def _apply_temperature_shift(
        self,
        X: np.ndarray,
        wavelengths: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Temperature shift (peak position shift + intensity change).

        Temperature affects molecular vibrations, causing peak shifts.
        """
        if wavelengths is None:
            wavelengths = np.arange(X.shape[1])

        # Peak shift: ~0.1 nm per 10°C
        wavelength_shift = self.magnitude * 0.01

        # Interpolate to shifted wavelengths
        X_shifted = np.zeros_like(X)
        for i in range(X.shape[0]):
            wavelengths_shifted = wavelengths + wavelength_shift
            X_shifted[i] = np.interp(wavelengths, wavelengths_shifted, X[i])

        # Intensity change (~1% per 10°C)
        intensity_factor = 1.0 + self.magnitude * 0.001
        X_shifted *= intensity_factor

        return X_shifted

    def _apply_concentration_shift(self, X: np.ndarray) -> np.ndarray:
        """
        Concentration range shift (scaling).

        Test samples have systematically different concentration ranges.
        """
        # Scale intensities
        scale_factor = 1.0 + self.magnitude * 0.1
        return X * scale_factor

    def _apply_instrument_shift(self, X: np.ndarray) -> np.ndarray:
        """
        Instrument-to-instrument transfer (baseline + scale).

        Different instruments have systematic biases.
        """
        # Additive baseline shift
        baseline = self.magnitude * 0.01 * self.rng.uniform(-1, 1, X.shape[1])

        # Multiplicative scale
        scale = 1.0 + self.magnitude * 0.05 * self.rng.uniform(-1, 1, X.shape[1])

        X_shifted = X * scale + baseline

        return X_shifted

    def _apply_time_drift(self, X: np.ndarray) -> np.ndarray:
        """
        Time-dependent drift (gradual baseline change).

        Simulates instrument aging or environmental drift.
        """
        n_samples, n_features = X.shape

        # Linear drift over sample index
        drift = np.linspace(0, self.magnitude * 0.01, n_samples)
        drift = drift[:, np.newaxis]

        return X + drift
