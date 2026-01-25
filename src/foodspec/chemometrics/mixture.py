"""Mixture analysis utilities (NNLS and simplified MCR-ALS)."""
from __future__ import annotations


from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.optimize import nnls as scipy_nnls
except Exception:  # pragma: no cover
    scipy_nnls = None

__all__ = ["nnls_mixture", "mcr_als", "run_mixture_analysis_workflow"]


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
        # Update C (solve X ≈ C S^T)
        S_pinv = np.linalg.pinv(S.T)
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


def run_mixture_analysis_workflow(
    mixtures: np.ndarray,
    pure_spectra: Optional[np.ndarray] = None,
    n_components: Optional[int] = None,
    mode: str = "nnls",
) -> dict:
    """Convenience wrapper for mixture analysis.

    Parameters
    ----------
    mixtures : np.ndarray
        Spectra of mixtures, shape (n_samples, n_wavenumbers).
    pure_spectra : Optional[np.ndarray], optional
        Pure/reference spectra, shape (n_components, n_wavenumbers). Required for NNLS.
    n_components : Optional[int], optional
        Number of components for MCR-ALS if ``pure_spectra`` are not provided.
    mode : str, optional
        ``\"nnls\"`` to fit against known pure spectra, or ``\"mcr_als\"`` for unsupervised
        decomposition, by default ``\"nnls\"``.

    Returns
    -------
    dict
        For NNLS: ``{\"mode\": \"nnls\", \"coefficients\": array, \"residual_norms\": array}``.
        For MCR-ALS: ``{\"mode\": \"mcr_als\", \"C\": C, \"S\": S, \"relative_error\": err}``.

    Raises
    ------
    ValueError
        If pure_spectra are missing for NNLS, or n_components is missing for MCR-ALS.

    See also
    --------
    docs/workflows/mixture_analysis.md : Workflow and reporting guidance.
    """
    if mode == "nnls":
        if pure_spectra is None:
            raise ValueError("pure_spectra required for NNLS mode.")
        coeffs = []
        residuals = []
        pure_matrix = pure_spectra.T  # (n_points, n_components)
        for spec in mixtures:
            c, rnorm = nnls_mixture(spec, pure_matrix)
            coeffs.append(c)
            residuals.append(rnorm)
        return {"mode": "nnls", "coefficients": np.vstack(coeffs), "residual_norms": np.array(residuals)}

    if mode == "mcr_als":
        if n_components is None:
            raise ValueError("n_components must be provided for MCR-ALS.")
        C, S = mcr_als(X=mixtures, n_components=n_components)
        recon = C @ S.T
        err = np.linalg.norm(mixtures - recon) / np.linalg.norm(mixtures)
        return {"mode": "mcr_als", "C": C, "S": S, "relative_error": err}

    raise ValueError("mode must be 'nnls' or 'mcr_als'.")
