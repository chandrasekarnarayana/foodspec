"""Hyperspectral unmixing algorithms."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import NMF


def mcr_als(
    X: np.ndarray,
    *,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Multivariate Curve Resolution - Alternating Least Squares (MCR-ALS).

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features).
    n_components : int
        Number of components to extract.
    max_iter : int, default=100
        Maximum ALS iterations.
    tol : float, default=1e-5
        Convergence tolerance (relative error change).
    random_state : int, default=0
        Random seed for initialization.

    Returns
    -------
    C : np.ndarray
        Concentration/abundance matrix (n_samples, n_components).
    S : np.ndarray
        Spectral signatures (n_components, n_features).
    info : dict
        Convergence metrics: final_error, n_iter.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D array (n_samples, n_features).")
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    nmf = NMF(n_components=n_components, init="random", random_state=random_state, max_iter=200)
    C = nmf.fit_transform(np.maximum(X, 0))
    S = nmf.components_

    prev_err = np.inf
    for i in range(max_iter):
        # Update C
        Ct = S @ S.T
        C = X @ S.T @ np.linalg.pinv(Ct)
        C = np.maximum(C, 0)

        # Update S
        St = C.T @ C
        S = np.linalg.pinv(St) @ C.T @ X
        S = np.maximum(S, 0)

        recon = C @ S
        err = float(np.linalg.norm(X - recon) / (np.linalg.norm(X) + 1e-12))
        if abs(prev_err - err) < tol:
            return C, S, {"final_error": err, "n_iter": float(i + 1)}
        prev_err = err

    return C, S, {"final_error": float(prev_err), "n_iter": float(max_iter)}


__all__ = ["mcr_als"]
