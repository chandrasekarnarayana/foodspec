"""Validation helpers for FoodSpec IO."""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Set

import numpy as np

from foodspec.core.errors import FoodSpecValidationError

_UNIT_MAP = {
    "cm^-1": "cm-1",
    "cm-1": "cm-1",
    "1/cm": "cm-1",
    "cm⁻¹": "cm-1",
    "nm": "nm",
    "um": "um",
}


def validate_wavenumber_axis(
    x: np.ndarray,
    *,
    allow_descending: bool = False,
    min_value: float = 50.0,
    max_value: float = 50000.0,
) -> None:
    """Validate spectral axis values."""
    axis = np.asarray(x, dtype=float).ravel()
    if axis.size < 2:
        raise FoodSpecValidationError("Wavenumber axis must contain at least 2 points.")
    if not np.isfinite(axis).all():
        raise FoodSpecValidationError("Wavenumber axis contains NaN or infinite values.")

    diffs = np.diff(axis)
    if allow_descending:
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            raise FoodSpecValidationError("Wavenumber axis must be strictly monotonic.")
    else:
        if not np.all(diffs > 0):
            raise FoodSpecValidationError("Wavenumber axis must be strictly increasing.")

    if min_value is not None and axis.min() < min_value:
        raise FoodSpecValidationError(f"Wavenumber axis below expected range ({min_value}).")
    if max_value is not None and axis.max() > max_value:
        raise FoodSpecValidationError(f"Wavenumber axis above expected range ({max_value}).")


def validate_intensity_array(y: np.ndarray) -> None:
    """Validate intensity values."""
    arr = np.asarray(y, dtype=float)
    if not np.isfinite(arr).all():
        raise FoodSpecValidationError("Intensity array contains NaN or infinite values.")

    # Optional clipping detection (warn only)
    if arr.size > 0:
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if max_val > min_val:
            tol = (max_val - min_val) * 1e-3
            frac_min = float(np.mean(arr <= min_val + tol))
            frac_max = float(np.mean(arr >= max_val - tol))
            if frac_min > 0.02 or frac_max > 0.02:
                warnings.warn(
                    "Intensity values appear clipped near min/max; check detector saturation.",
                    RuntimeWarning,
                )


def validate_metadata(
    meta: Mapping[str, object],
    required: Set[str],
    optional: Set[str],
) -> None:
    """Validate metadata fields and normalize units when present."""
    if not isinstance(meta, Mapping):
        raise FoodSpecValidationError("Metadata must be a mapping/dict.")

    missing = [key for key in required if key not in meta]
    if missing:
        raise FoodSpecValidationError(f"Missing required metadata keys: {missing}")

    for key in optional:
        if key not in meta:
            continue
        value = meta[key]
        if key in {"instrument", "source", "operator"} and value is not None and not isinstance(value, str):
            raise FoodSpecValidationError(f"Metadata field '{key}' must be a string.")
        if key in {"unit", "x_unit", "axis_unit"} and value is not None:
            unit = str(value)
            normalized = _UNIT_MAP.get(unit, unit)
            if normalized not in _UNIT_MAP.values():
                raise FoodSpecValidationError(f"Unrecognized unit '{unit}'.")
            if isinstance(meta, dict):
                meta[key] = normalized


def validate_spectrum_schema(spectrum: object) -> None:
    """Validate a Spectrum-like object with axis, intensity, and metadata."""
    axis = None
    intensity = None
    metadata = None

    if hasattr(spectrum, "wavenumbers"):
        axis = getattr(spectrum, "wavenumbers")
        intensity = getattr(spectrum, "spectra", None) or getattr(spectrum, "x", None)
    else:
        axis = getattr(spectrum, "x", None)
        intensity = getattr(spectrum, "y", None)

    if axis is None or intensity is None:
        raise FoodSpecValidationError("Spectrum object missing axis or intensity arrays.")

    validate_wavenumber_axis(np.asarray(axis))
    validate_intensity_array(np.asarray(intensity))

    metadata = getattr(spectrum, "metadata", None)
    if isinstance(metadata, Mapping):
        validate_metadata(metadata, required=set(), optional={"instrument", "source", "unit", "x_unit", "axis_unit"})


def validate_input(path: str, *, deep: bool = False) -> Dict[str, List[str]]:
    """Validate input path and inferred format.

    Returns dict with "errors" and "warnings" lists.
    """

    errors: List[str] = []
    warnings: List[str] = []

    from foodspec.io.core import detect_format

    fmt = detect_format(path)
    if fmt == "unknown":
        errors.append("Unsupported or unknown input format.")
    elif fmt == "txt":
        warnings.append("Plain text inputs may require format hints.")

    if deep and not errors:
        from foodspec.io.core import read_spectra

        try:
            spectrum = read_spectra(path)
            validate_spectrum_schema(spectrum)
        except FoodSpecValidationError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"Failed to validate input: {exc}")

    return {"errors": errors, "warnings": warnings}


__all__ = [
    "validate_input",
    "validate_wavenumber_axis",
    "validate_intensity_array",
    "validate_metadata",
    "validate_spectrum_schema",
]
