"""Reusable validation helpers for foodspec datasets."""

from __future__ import annotations

import sys
import platform
import warnings

import numpy as np

from foodspec.core.dataset import FoodSpectrumSet

__all__ = [
    "ValidationError",
    "validate_spectrum_set",
    "validate_public_evoo_sunflower",
]


class ValidationError(Exception):
    """Custom exception raised when validation checks fail."""


def _format_context(message: str) -> str:
    return f"{message} (Python {sys.version_info.major}.{sys.version_info.minor} on {platform.platform()})"


def validate_spectrum_set(
    fs: FoodSpectrumSet,
    *,
    check_monotonic: bool = True,
    allow_nan: bool = False,
) -> None:
    """Validate the structural integrity of a FoodSpectrumSet.

    Parameters
    ----------
    fs :
        Dataset to validate.
    check_monotonic :
        If True, enforce strictly increasing wavenumber axis.
    allow_nan :
        If False, NaN values in ``fs.x`` trigger a validation error.

    Raises
    ------
    ValidationError
        If any structural issue is detected.
    """

    if fs.x.ndim != 2 or fs.x.size == 0:
        raise ValidationError(_format_context("x must be a non-empty 2D array"))
    if fs.wavenumbers.ndim != 1:
        raise ValidationError(_format_context("wavenumbers must be a 1D array"))
    if fs.x.shape[1] != fs.wavenumbers.shape[0]:
        raise ValidationError(
            _format_context(
                "wavenumber axis length does not match x columns "
                f"({fs.wavenumbers.shape[0]} != {fs.x.shape[1]})"
            )
        )
    if check_monotonic and not np.all(np.diff(fs.wavenumbers) > 0):
        raise ValidationError(_format_context("wavenumbers must be strictly increasing"))
    if not allow_nan and np.isnan(fs.x).any():
        raise ValidationError(_format_context("NaN values detected in spectra"))
    if len(fs.metadata) != fs.x.shape[0]:
        raise ValidationError(
            _format_context(
                f"metadata length {len(fs.metadata)} does not match number of spectra {fs.x.shape[0]}"
            )
        )


def validate_public_evoo_sunflower(fs: FoodSpectrumSet) -> None:
    """Validate the EVOO–sunflower mixture dataset metadata and ranges.

    Ensures that ``mixture_fraction_evoo`` exists and values lie in [0, 1]
    or [0, 100]. Values outside these bounds raise ``ValidationError``.
    """

    if "mixture_fraction_evoo" not in fs.metadata.columns:
        raise ValidationError(
            _format_context("metadata must include 'mixture_fraction_evoo'")
        )
    series = fs.metadata["mixture_fraction_evoo"]
    finite_vals = series.dropna()
    if finite_vals.empty:
        warnings.warn(
            "mixture_fraction_evoo is empty or NaN; downstream regression may fail.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    min_val = finite_vals.min()
    max_val = finite_vals.max()
    within_unit = 0 <= min_val <= max_val <= 1
    within_percent = 0 <= min_val <= max_val <= 100

    if within_percent and not within_unit and max_val > 1:
        warnings.warn(
            "mixture_fraction_evoo appears to be in percent (0–100); consider scaling to 0–1.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not (within_unit or within_percent):
        raise ValidationError(
            _format_context(
                "mixture_fraction_evoo values must be within [0,1] or [0,100]; "
                f"observed min={min_val}, max={max_val}"
            )
        )
