"""NNLS-based spectral unmixing for mixture analysis."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import nnls as scipy_nnls

__all__ = ["NNLSUnmixer", "unmix_spectrum"]


class NNLSUnmixer:
    """Non-negative least squares spectral unmixer."""

    def __init__(self):
        """Initialize NNLS unmixer."""
        self.library_ = None
        self.n_components_ = None

    def fit(self, library: np.ndarray) -> NNLSUnmixer:
        """
        Fit unmixer with reference library.

        Parameters
        ----------
        library : np.ndarray, shape (n_components, n_wavenumbers)
            Reference spectra (pure components).

        Returns
        -------
        self
        """
        self.library_ = library.astype(np.float64)
        self.n_components_ = library.shape[0]
        return self

    def transform(
        self,
        X: np.ndarray,
        return_residual: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Unmix spectra using NNLS.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_wavenumbers) or (n_wavenumbers,)
            Mixture spectrum/spectra.
        return_residual : bool, default=False
            If True, also return residual sum of squares.

        Returns
        -------
        concentrations : np.ndarray, shape (n_samples, n_components) or (n_components,)
            Estimated component concentrations (non-negative).
        residuals : np.ndarray, optional
            Residual sum of squares for each sample (if return_residual=True).
        """
        if self.library_ is None:
            raise ValueError("Must call fit() first")

        X_input = np.asarray(X)
        input_is_1d = X_input.ndim == 1
        X = np.atleast_2d(X_input)

        concentrations = np.zeros((X.shape[0], self.n_components_))
        residuals = None if not return_residual else np.zeros(X.shape[0])

        for i, spectrum in enumerate(X):
            # Solve: minimize ||library.T @ c - spectrum||^2 subject to c >= 0
            c, rss = scipy_nnls(self.library_.T, spectrum)
            concentrations[i] = c
            if return_residual:
                residuals[i] = rss

        # Return 1D if input was 1D
        if input_is_1d and concentrations.shape[0] == 1:
            concentrations = concentrations[0]

        if return_residual:
            return concentrations, residuals
        return concentrations

    def fit_transform(
        self,
        X: np.ndarray,
        library: Optional[np.ndarray] = None,
        return_residual: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray
            Mixture spectra.
        library : np.ndarray, optional
            Reference library. If None, uses first sample of X as library.
        return_residual : bool, default=False
            If True, return residuals.

        Returns
        -------
        concentrations, residuals
        """
        if library is None:
            raise ValueError("library must be provided")
        self.fit(library)
        return self.transform(X, return_residual=return_residual)

    def predict_mixture(self, concentrations: np.ndarray) -> np.ndarray:
        """
        Reconstruct mixture spectrum from concentrations.

        Parameters
        ----------
        concentrations : np.ndarray, shape (n_components,) or (n_samples, n_components)
            Component concentrations.

        Returns
        -------
        spectra : np.ndarray
            Reconstructed mixture spectra.
        """
        if self.library_ is None:
            raise ValueError("Must call fit() first")

        concentrations = np.atleast_2d(concentrations)
        return concentrations @ self.library_


def unmix_spectrum(
    mixture: np.ndarray,
    library: np.ndarray,
    return_residual: bool = False,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Unmix a single spectrum or batch using NNLS.

    Parameters
    ----------
    mixture : np.ndarray, shape (n_wavenumbers,) or (n_samples, n_wavenumbers)
        Mixture spectrum/spectra.
    library : np.ndarray, shape (n_components, n_wavenumbers)
        Reference library.
    return_residual : bool, default=False
        If True, also return residuals.

    Returns
    -------
    concentrations : np.ndarray
        Estimated concentrations.
    residuals : np.ndarray, optional
        Residual sum of squares.

    Examples
    --------
    >>> library = np.array([[...], [...], [...]])  # 3 pure components
    >>> mixture = np.array([...])  # Unknown blend
    >>> conc = unmix_spectrum(mixture, library)
    >>> print(conc)  # [c1, c2, c3] summing to 1.0
    """
    unmixer = NNLSUnmixer()
    return unmixer.fit_transform(mixture, library=library, return_residual=return_residual)
