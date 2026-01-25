"""Shared typing aliases for FoodSpec."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional in some environments
    pd = None

ArrayLike = np.ndarray
VectorLike = np.ndarray
MatrixLike = np.ndarray

MetadataFrame = "pd.DataFrame" if pd is not None else object

FloatArray = np.ndarray
IntArray = np.ndarray

SequenceLike = Sequence
IterableLike = Iterable
MappingLike = Mapping

