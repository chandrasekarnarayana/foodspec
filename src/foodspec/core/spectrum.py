"""Legacy import path for Spectrum (deprecated)."""
from __future__ import annotations

from foodspec.utils.deprecation import warn_deprecated_import
from foodspec.data_objects.spectrum import Spectrum

warn_deprecated_import("foodspec.core.spectrum", "foodspec.data_objects.spectrum")

__all__ = ["Spectrum"]

