"""Dataset validation helpers for modeling workflows."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when a dataset fails validation checks."""


def validate_public_evoo_sunflower(data, allow_nan: bool = True) -> bool:
    """Validator for public EVOO/Sunflower dataset."""
    meta = getattr(data, "metadata", None)
    if meta is None:
        raise ValidationError("Dataset metadata missing.")
    if not allow_nan and meta.isna().any().any():
        raise ValidationError("NaN values not allowed.")
    if "mixture_fraction_evoo" in meta.columns:
        if (meta["mixture_fraction_evoo"] > 100).any() or (meta["mixture_fraction_evoo"] < 0).any():
            raise ValidationError("mixture_fraction_evoo must be between 0 and 100.")
    return True


def validate_spectrum_set(
    dataset,
    allow_nan: bool = False,
    check_monotonic: bool = True,
) -> bool:
    """Validate a FoodSpectrumSet (shape consistency, monotonic wn, NaNs)."""
    wn = getattr(dataset, "wavenumbers", None)
    spectra = getattr(dataset, "spectra", getattr(dataset, "x", None))
    if wn is None or spectra is None:
        raise ValidationError("Spectrum set missing wavenumbers or spectra.")
    if check_monotonic and np.any(np.diff(wn) <= 0):
        raise ValidationError("Wavenumbers must be strictly increasing.")
    if not allow_nan and np.isnan(spectra).any():
        raise ValidationError("NaN values not allowed.")
    return True


def validate_dataset(
    dataset: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    class_col: Optional[str] = None,
    min_classes: int = 2,
) -> Dict[str, List[str]]:
    """Generic dataset validation returning diagnostics instead of raising."""
    errors: List[str] = []
    warnings: List[str] = []
    required_cols = required_cols or []
    missing = [c for c in required_cols if c not in dataset.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if class_col and class_col in dataset.columns:
        nunique = dataset[class_col].nunique(dropna=True)
        if nunique < min_classes:
            warnings.append(f"Class column '{class_col}' has {nunique} classes; discrimination limited.")
    const_cols = [c for c in dataset.columns if dataset[c].nunique(dropna=True) <= 1]
    if const_cols:
        warnings.append(f"constant columns detected: {', '.join(const_cols)}")
    return {"errors": errors, "warnings": warnings}


__all__ = [
    "ValidationError",
    "validate_public_evoo_sunflower",
    "validate_spectrum_set",
    "validate_dataset",
]

