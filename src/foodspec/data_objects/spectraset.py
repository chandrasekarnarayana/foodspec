"""SpectraSet data object (shim to foodspec.core.dataset/FoodSpectrumSet)."""
from __future__ import annotations

from foodspec.core.dataset import FoodSpectrumSet

SpectraSet = FoodSpectrumSet

__all__ = ["FoodSpectrumSet", "SpectraSet"]

