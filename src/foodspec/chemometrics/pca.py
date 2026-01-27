"""Principal Component Analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA

__all__ = ["PCAResult", "run_pca"]


@dataclass
class PCAResult:
    """Container for PCA outputs."""

    scores: np.ndarray
    loadings: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    mean_: np.ndarray


def run_pca(X: np.ndarray, n_components: int = 2) -> Tuple[PCA, PCAResult]:
    """Run PCA on data matrix.

    Parameters
    ----------
    X :
        Array of shape (n_samples, n_features).
    n_components :
        Number of components to compute.

    Returns
    -------
    tuple
        Fitted PCA estimator and PCAResult container.
    """

    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    result = PCAResult(
        scores=scores,
        loadings=pca.components_.T,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        mean_=pca.mean_,
    )
    return pca, result
