from __future__ import annotations
"""
RQ Engine Utility Functions
===========================

Helper functions for ratio analysis and statistics.
"""


from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def _cv(series: pd.Series) -> Dict[str, float]:
    """Compute coefficient of variation and related statistics."""
    vals = series.dropna().astype(float)
    if len(vals) == 0:
        return {"mean": np.nan, "std": np.nan, "cv_percent": np.nan, "mad": np.nan, "mad_over_mean": np.nan}
    mean = vals.mean()
    std = vals.std(ddof=1) if len(vals) > 1 else 0.0
    mad = (vals - mean).abs().median()
    return {
        "mean": mean,
        "std": std,
        "cv_percent": (std / mean * 100) if mean != 0 else np.nan,
        "mad": mad,
        "mad_over_mean": (mad / mean) if mean != 0 else np.nan,
    }


def _safe_group_vectors(df: pd.DataFrame, group_col: str, feature: str) -> List[np.ndarray]:
    """Extract feature vectors grouped by a column."""
    vectors = []
    for _, sub in df.groupby(group_col):
        vec = sub[feature].dropna().to_numpy(dtype=float)
        if len(vec) > 1:
            vectors.append(vec)
    return vectors


def _monotonic_label(slope: float, p_value: float, alpha: float = 0.05) -> str:
    """Generate human-readable monotonicity label."""
    if np.isnan(p_value):
        return "no clear trend"
    if p_value >= alpha:
        return "no clear trend"
    return "increases with heating" if slope > 0 else "decreases with heating"


def _rf_accuracy(df: pd.DataFrame, features: List[str], label_col: str, random_state: int, n_splits: int) -> float:
    """Compute cross-validated Random Forest accuracy."""
    X = df[features].astype(float)
    y = df[label_col].astype(str)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]
    class_counts = y.value_counts(dropna=True)
    if len(class_counts) < 2 or X.empty:
        return np.nan
    min_class = class_counts.min()
    if min_class < 2:
        return np.nan
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    cv_splits = max(2, min(int(min_class), len(y), len(class_counts), n_splits))
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    scores = cross_val_score(rf, Xz, y, cv=cv)
    return float(scores.mean())
