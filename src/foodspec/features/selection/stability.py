"""Stability selection helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foodspec.features.metrics import feature_stability_by_group


def stability_selection(
    df: pd.DataFrame,
    labels: Iterable,
    *,
    n_resamples: int = 30,
    sample_fraction: float = 0.75,
    seed: int = 0,
    max_features: Optional[int] = None,
) -> pd.DataFrame:
    """Estimate feature stability via subsampling and L1 logistic selection."""

    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be within (0, 1].")
    y = np.asarray(list(labels))
    if len(y) != len(df):
        raise ValueError("labels length must match df rows.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")

    X = df.to_numpy(dtype=float)
    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    counts = np.zeros(n_features, dtype=float)
    coef_sums = np.zeros(n_features, dtype=float)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)),
        ]
    )

    for _ in range(n_resamples):
        size = max(2, int(round(sample_fraction * n_samples)))
        idx = rng.choice(n_samples, size=size, replace=False)
        pipeline.fit(X[idx], y[idx])
        coef = pipeline.named_steps["clf"].coef_
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        coef_sums += coef
        if max_features:
            selected = np.argsort(coef)[::-1][:max_features]
        else:
            selected = np.where(coef > 1e-8)[0]
        counts[selected] += 1

    # Clip to the valid probability range to guard against any floating jitter.
    frequency = np.clip(counts / float(n_resamples), 0.0, 1.0)
    mean_coef = coef_sums / float(n_resamples)

    out = pd.DataFrame(
        {
            "feature": df.columns,
            "frequency": frequency,
            "mean_coef": mean_coef,
        }
    )
    return out


def feature_importance_scores(df: pd.DataFrame, labels: Iterable) -> pd.Series:
    """Compute per-feature ANOVA F scores for classification."""

    y = np.asarray(list(labels))
    if len(y) != len(df):
        raise ValueError("labels length must match df rows.")
    X = df.to_numpy(dtype=float)
    scores, _ = f_classif(X, y)
    series = pd.Series(scores, index=df.columns)
    return series.fillna(0.0)


__all__ = ["feature_stability_by_group", "stability_selection", "feature_importance_scores"]
