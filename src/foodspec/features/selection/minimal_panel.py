"""Minimal marker panel helpers (shim to RatioQualityEngine)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from foodspec.features.rq.engine import RatioQualityEngine


def compute_minimal_panel(
    engine: RatioQualityEngine,
    df_ratios: pd.DataFrame,
    feature_importances: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute a minimal marker panel using an existing RatioQualityEngine."""

    return engine.compute_minimal_panel(df_ratios, feature_importances)


__all__ = ["compute_minimal_panel"]

