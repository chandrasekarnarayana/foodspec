"""Embedding feature extraction (PCA, PLS)."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


def pca_embeddings(
    X: np.ndarray,
    *,
    n_components: int = 2,
    prefix: str = "pca",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute PCA embeddings and return scores with metadata."""

    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    X = np.asarray(X, dtype=float)
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(X)
    cols = [f"{prefix}_{i + 1}" for i in range(scores.shape[1])]
    meta = {
        "n_components": n_components,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return pd.DataFrame(scores, columns=cols), meta


def pls_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_components: int = 2,
    prefix: str = "pls",
    mode: Literal["regression", "classification"] = "classification",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute PLS embeddings for regression or classification targets."""

    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    X = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    classes: Optional[list[str]] = None
    if mode == "classification":
        encoder = LabelEncoder()
        y_arr = encoder.fit_transform(y_arr)
        classes = [str(c) for c in encoder.classes_]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y_arr)
    scores = pls.transform(X_scaled)
    if isinstance(scores, tuple):
        scores = scores[0]
    cols = [f"{prefix}_{i + 1}" for i in range(scores.shape[1])]
    meta = {
        "n_components": n_components,
        "mode": mode,
        "classes": classes,
        "x_weights": pls.x_weights_.tolist(),
        "x_loadings": pls.x_loadings_.tolist(),
    }
    return pd.DataFrame(scores, columns=cols), meta


__all__ = ["pca_embeddings", "pls_embeddings"]
