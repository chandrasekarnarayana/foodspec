from __future__ import annotations

"""
RQ Engine Data Types
====================

Dataclass definitions for the Ratio-Quality (RQ) engine.
"""


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PeakDefinition:
    """Named peak/intensity column."""

    name: str
    column: str
    wavenumber: Optional[float] = None
    window: Optional[Tuple[float, float]] = None  # Optional metadata for UI/reporting
    mode: str = "max"  # max | area


@dataclass
class RatioDefinition:
    """Numerator / denominator ratio built from PeakDefinition names or raw columns."""

    name: str
    numerator: str
    denominator: str


@dataclass
class RQConfig:
    """Column naming conventions and options."""

    oil_col: str = "Oil_Name"
    matrix_col: str = "matrix"
    heating_col: str = "Heating_Stage"
    replicate_col: str = "replicate_id"
    sample_col: str = "sample_id"
    # Whether to include classifier-based feature importance
    compute_feature_importance: bool = True
    random_state: int = 0
    n_splits: int = 5
    normalization_modes: List[str] = field(default_factory=lambda: ["reference"])
    minimal_panel_target_accuracy: float = 0.9
    enable_clustering: bool = True
    adjust_p_values: bool = True
    max_features: Optional[int] = None


@dataclass
class RatioQualityResult:
    """Result container for RQ analysis."""

    ratio_table: pd.DataFrame
    stability_summary: pd.DataFrame
    discriminative_summary: pd.DataFrame
    feature_importance: Optional[pd.DataFrame]
    heating_trend_summary: pd.DataFrame
    oil_vs_chips_summary: pd.DataFrame
    normalization_comparison: Optional[pd.DataFrame]
    minimal_panel: Optional[pd.DataFrame]
    clustering_metrics: Optional[Dict]
    warnings: List[str]
    text_report: str
