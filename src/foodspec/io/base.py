"""
Base abstractions for IO loaders.

Loaders are simple functions that accept a path and return a FoodSpectrumSet or
an intermediate DataFrame that can be converted downstream.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet


class SpectraLoader(Protocol):
    def __call__(self, path: str, **kwargs) -> pd.DataFrame | FoodSpectrumSet: ...
