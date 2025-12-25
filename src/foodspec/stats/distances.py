"""Spectral distance and similarity metrics.

Implements Euclidean, cosine, Pearson, SID (spectral information divergence),
and SAM (spectral angle mapper), along with utilities for computing
query-to-library distance matrices.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "euclidean_distance",
    "cosine_distance",
    "pearson_distance",
    "sid_distance",
    "sam_angle",
    "compute_distances",
]


def _ensure_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = _ensure_1d(a)
    b = _ensure_1d(b)
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = _ensure_1d(a)
    b = _ensure_1d(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cos_sim = float(np.dot(a, b) / denom)
    return float(1.0 - cos_sim)


def pearson_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = _ensure_1d(a)
    b = _ensure_1d(b)
    if a.size != b.size:
        raise ValueError("Vectors must be the same length for Pearson distance.")
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = (np.linalg.norm(a_c) * np.linalg.norm(b_c)) + 1e-12
    rho = float(np.dot(a_c, b_c) / denom)
    # Convert correlation [-1,1] to distance; use 1 - rho so anti-correlation yields larger distance
    return float(1.0 - rho)


def sid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral information divergence.

    Treats spectra as probability distributions (non-negative, sum to 1).
    Applies a small offset and normalization for stability.
    """
    a = _ensure_1d(a)
    b = _ensure_1d(b)
    a_shift = a - a.min()
    b_shift = b - b.min()
    a_p = a_shift + 1e-12
    b_p = b_shift + 1e-12
    a_p = a_p / a_p.sum()
    b_p = b_p / b_p.sum()
    div = np.sum(a_p * np.log(a_p / b_p)) + np.sum(b_p * np.log(b_p / a_p))
    return float(div)


def sam_angle(a: np.ndarray, b: np.ndarray, degrees: bool = False) -> float:
    """Spectral angle mapper: angle between vectors.

    Returns radians by default; set degrees=True to convert.
    """
    a = _ensure_1d(a)
    b = _ensure_1d(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cos_sim = float(np.dot(a, b) / denom)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    theta = float(np.arccos(cos_sim))
    if degrees:
        theta = float(np.degrees(theta))
    return theta


def compute_distances(
    X_query: np.ndarray,
    X_library: np.ndarray,
    metric: Literal["euclidean", "cosine", "pearson", "sid", "sam"] = "cosine",
    sam_in_degrees: bool = False,
) -> np.ndarray:
    """Compute query-to-library distance matrix.

    Parameters
    ----------
    X_query : array, shape (n_query, n_features)
    X_library : array, shape (n_library, n_features)
    metric : one of {'euclidean','cosine','pearson','sid','sam'}
    sam_in_degrees : if metric='sam', return angles in degrees

    Returns
    -------
    distances : array, shape (n_query, n_library)
    """
    Xq = np.asarray(X_query, dtype=float)
    Xl = np.asarray(X_library, dtype=float)
    if Xq.ndim != 2 or Xl.ndim != 2:
        raise ValueError("X_query and X_library must be 2D arrays.")
    if Xq.shape[1] != Xl.shape[1]:
        raise ValueError("Feature dimension mismatch between query and library.")

    n_q, n_l = Xq.shape[0], Xl.shape[0]
    D = np.zeros((n_q, n_l), dtype=float)
    for i in range(n_q):
        a = Xq[i]
        for j in range(n_l):
            b = Xl[j]
            if metric == "euclidean":
                d = euclidean_distance(a, b)
            elif metric == "cosine":
                d = cosine_distance(a, b)
            elif metric == "pearson":
                d = pearson_distance(a, b)
            elif metric == "sid":
                d = sid_distance(a, b)
            elif metric == "sam":
                d = sam_angle(a, b, degrees=sam_in_degrees)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            D[i, j] = d
    return D
