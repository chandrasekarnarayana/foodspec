"""Legacy import path for FoodSpectrumSet (deprecated)."""

from __future__ import annotations

from foodspec.data_objects.spectra_set import FoodSpectrumSet, from_sklearn, to_sklearn
from foodspec.utils.deprecation import warn_deprecated_import

warn_deprecated_import("foodspec.core.dataset", "foodspec.data_objects.spectra_set")

__all__ = ["FoodSpectrumSet", "from_sklearn", "to_sklearn"]
