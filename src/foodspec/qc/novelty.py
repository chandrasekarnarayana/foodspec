from __future__ import annotations
"""Novelty scoring relative to a training manifold.

Provides a simple distance-to-nearest-neighbor measure and percentile-based
novelty score where higher values indicate more novelty.
"""


from typing import Literal, Tuple

import numpy as np

from foodspec.stats.distances import compute_distances

__all__ = ["novelty_scores", "novelty_score_single"]


def _nearest_neighbor_distances(X: np.ndarray, metric: str) -> np.ndarray:
    D = compute_distances(X, X, metric=metric)
    # Exclude self-distance by setting diagonal to +inf then taking rowwise min
    np.fill_diagonal(D, np.inf)
    return np.min(D, axis=1)


def novelty_scores(
    X_train: np.ndarray,
    X_query: np.ndarray,
    metric: Literal["euclidean", "cosine", "pearson", "sid", "sam"] = "cosine",
    threshold: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute novelty scores and flags for a batch of queries.

    Returns (scores, is_novel) where scores in [0,1] (higher = more novel)
    and is_novel is a boolean mask compared against threshold (default 0.8 quantile).
    """
    train_nn = _nearest_neighbor_distances(X_train, metric)
    ref_dist = float(np.median(train_nn))
    D = compute_distances(X_query, X_train, metric=metric)
    q_nn = np.min(D, axis=1)
    # Scale relative to training median NN distance, then squash to [0,1]
    scaled = q_nn / (ref_dist + 1e-12)
    scores = 1.0 - (1.0 / (1.0 + scaled))
    if threshold is None:
        thr = float(np.quantile(scores, 0.8))
    else:
        thr = float(threshold)
    flags = scores >= thr
    return scores.astype(float), flags.astype(bool)


def novelty_score_single(
    X_train: np.ndarray,
    x_query: np.ndarray,
    metric: Literal["euclidean", "cosine", "pearson", "sid", "sam"] = "cosine",
    threshold: float | None = None,
) -> Tuple[float, bool]:
    scores, flags = novelty_scores(X_train, x_query[None, :], metric=metric, threshold=threshold)
    return float(scores[0]), bool(flags[0])
