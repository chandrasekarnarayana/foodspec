"""
Ratio-Quality (RQ) Engine Package
==================================

Lightweight, dependency-minimal analysis of Raman/FTIR peak ratios for oils/chips.

Example
-------
>>> import pandas as pd
>>> from foodspec.features.rq import (
...     PeakDefinition, RatioDefinition, RQConfig, RatioQualityEngine
... )
>>> df = pd.DataFrame({
...     "sample_id": [1, 2, 3, 4],
...     "oil_type": ["A", "A", "B", "B"],
...     "matrix": ["oil", "oil", "oil", "oil"],
...     "heating_stage": [0, 1, 0, 1],
...     "I_1742": [10, 9, 6, 5],
...     "I_1652": [4, 3, 8, 7],
... })
>>> peaks = [
...     PeakDefinition(name="I_1742", column="I_1742"),
...     PeakDefinition(name="I_1652", column="I_1652"),
... ]
>>> ratios = [RatioDefinition(name="1742/1652", numerator="I_1742", denominator="I_1652")]
>>> cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage")
>>> engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg)
>>> results = engine.run_all(df)
>>> print(results.text_report[:120])
"""

from .engine import RatioQualityEngine
from .types import PeakDefinition, RatioDefinition, RatioQualityResult, RQConfig

__all__ = [
    "PeakDefinition",
    "RatioDefinition",
    "RQConfig",
    "RatioQualityEngine",
    "RatioQualityResult",
]
