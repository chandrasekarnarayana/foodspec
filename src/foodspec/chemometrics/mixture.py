"""Mixture analysis utilities (NNLS and simplified MCR-ALS)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.optimize import nnls as scipy_nnls
except Exception:  # pragma: no cover
    scipy_nnls = None

__all__ = ["nnls_mixture", "mcr_als"]


def nnls_mixture(spectrum: np.ndarray, pure_spectra: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a non-negative least squares mixture.

    Parameters
    ----------
    spectrum:
        Array of shape (n_points,) representing the mixture spectrum.
    pure_spectra:
        Array of shape (n_points, n_components) containing pure component spectra as columns.

    Returns
    -------
    coefficients:
        Non-negative coefficients for each component (shape (n_components,)).
    residual_norm:
        Euclidean norm of the residual.
    """

    spectrum = np.asarray(spectrum, dtype=float).ravel()
    pure_spectra = np.asarray(pure_spectra, dtype=float)
    if pure_spectra.shape[0] != spectrum.shape[0]:
        raise ValueError("pure_spectra rows must match spectrum length.")

    if scipy_nnls is not None:
        coeffs, res = scipy_nnls(pure_spectra, spectrum)
    else:  # simple non-negative least squares fallback
        coeffs, *_ = np.linalg.lstsq(pure_spectra, spectrum, rcond=None)
        coeffs = np.clip(coeffs, 0, None)
        res = np.linalg.norm(spectrum - pure_spectra @ coeffs)
    return coeffs, float(res)


def mcr_als(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a simplified MCR-ALS decomposition with non-negativity clipping.

    Parameters
    ----------
    X:
        Data matrix of shape (n_samples, n_points).
    n_components:
        Number of components to estimate.
    max_iter:
        Maximum number of ALS iterations.
    tol:
        Convergence tolerance on reconstruction error.
    random_state:
        Optional seed for reproducible initialization.

    Returns
    -------
    C:
        Concentration profiles (n_samples, n_components).
    S:
        Spectral profiles (n_points, n_components).
    """

    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n_samples, n_points = X.shape
    S = np.abs(rng.standard_normal(size=(n_points, n_components)))
    C = np.abs(rng.standard_normal(size=(n_samples, n_components)))

    prev_err = np.inf
    for _ in range(max_iter):
        # Update C
        S_pinv = np.linalg.pinv(S)
        C = np.maximum(0, X @ S_pinv)
        # Update S
        C_pinv = np.linalg.pinv(C)
        S = np.maximum(0, (C_pinv @ X).T)

        recon = C @ S.T
        err = np.linalg.norm(X - recon)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return C, S
