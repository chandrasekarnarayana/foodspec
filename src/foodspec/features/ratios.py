"""Ratio feature utilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["compute_ratios", "RatioFeatureGenerator"]


def compute_ratios(df: pd.DataFrame, ratio_def: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """Compute ratios of specified columns.

    Parameters
    ----------
    df :
        DataFrame containing numerator and denominator columns.
    ratio_def :
        Mapping from new column name to (numerator_col, denominator_col).

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with additional ratio columns.
    """

    result = df.copy()
    for name, (num_col, denom_col) in ratio_def.items():
        if num_col not in result.columns or denom_col not in result.columns:
            raise ValueError(f"Columns {num_col} and {denom_col} must exist in DataFrame.")
        denom = result[denom_col].to_numpy()
        num = result[num_col].to_numpy()
        ratio = np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float), where=denom != 0)
        result[name] = ratio
    return result


class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate ratio features for use in pipelines."""

    def __init__(self, ratio_def: Dict[str, Tuple[str, str]]):
        self.ratio_def = ratio_def

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "RatioFeatureGenerator":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("RatioFeatureGenerator expects a pandas DataFrame.")
        return compute_ratios(X, self.ratio_def)
