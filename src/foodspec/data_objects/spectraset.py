"""Deprecated alias for spectra_set (kept for compatibility)."""

from __future__ import annotations

from foodspec.data_objects.spectra_set import FoodSpectrumSet, SpectraSet
from foodspec.utils.deprecation import warn_deprecated_import

warn_deprecated_import("foodspec.data_objects.spectraset", "foodspec.data_objects.spectra_set")

__all__ = ["FoodSpectrumSet", "SpectraSet"]
