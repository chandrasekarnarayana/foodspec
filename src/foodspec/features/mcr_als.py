"""
Multivariate Curve Resolution - Alternating Least Squares (MCR-ALS).

This module provides MCR-ALS for spectral decomposition, allowing
resolution of mixtures into pure component spectra and concentration profiles.

Key Features:
- ALS optimization with multiple constraint types
- Non-negativity, unimodality, closure, and equality constraints
- Rotational ambiguity analysis (band boundaries, feasible regions)
- Convergence diagnostics and lack-of-fit metrics
- sklearn-compatible API

References:
    [1] Jaumot et al. (2015). MCR-ALS GUI 2.0: New features and applications.
        Chemometrics and Intelligent Laboratory Systems, 140, 1-12.
    [2] de Juan & Tauler (2021). Multivariate Curve Resolution: 50 years
        addressing the mixture analysis problem - A review.
        Analytica Chimica Acta, 1145, 59-78.
    [3] Golshan et al. (2016). Resolution and segmentation of hyperspectral
        biomedical images by multivariate curve resolution.
        Analytica Chimica Acta, 909, 14-28.

Example:
    >>> from foodspec.features.mcr_als import MCRALS
    >>> import numpy as np
    >>>
    >>> # Simulated mixture data
    >>> X = np.random.rand(100, 200)  # 100 samples × 200 wavelengths
    >>>
    >>> # MCR-ALS with 3 components
    >>> mcr = MCRALS(n_components=3, max_iter=50, tol=1e-6)
    >>> mcr.fit(X)
    >>>
    >>> # Get pure spectra and concentrations
    >>> C = mcr.transform(X)  # Concentrations
    >>> ST = mcr.components_  # Pure spectra (transposed)
    >>>
    >>> # Reconstruction and lack-of-fit
    >>> X_reconstructed = mcr.inverse_transform(C)
    >>> lof = mcr.score(X)
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class MCRALS(BaseEstimator, TransformerMixin):
    """
    Multivariate Curve Resolution - Alternating Least Squares.

    Decomposes mixture spectra X into concentration profiles C and pure
    component spectra S^T such that X ≈ C @ S^T, subject to constraints.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to resolve.

    max_iter : int, default=50
        Maximum number of ALS iterations.

    tol : float, default=1e-8
        Convergence tolerance on lack-of-fit change.

    c_constraints : list of str, default=['non_neg']
        Constraints on concentration matrix C. Options:
        - 'non_neg': C >= 0
        - 'norm': Each row of C sums to 1 (closure)
        - 'unimodal': Each column of C is unimodal

    st_constraints : list of str, default=['non_neg']
        Constraints on spectra matrix S^T. Options:
        - 'non_neg': S^T >= 0
        - 'norm': Each column of S^T sums to 1
        - 'unimodal': Each row of S^T is unimodal

    initialization : {'pca', 'random'}, default='pca'
        Method for initializing spectra S^T.

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Pure component spectra S^T.

    n_iter_ : int
        Number of ALS iterations performed.

    lack_of_fit_ : list of float
        Lack-of-fit (LOF) at each iteration.

    converged_ : bool
        Whether ALS converged.

    Notes
    -----
    MCR-ALS solves the bilinear decomposition problem:
        X = C @ S^T + E
    where:
    - X: data matrix (n_samples × n_features)
    - C: concentrations (n_samples × n_components)
    - S^T: spectra (n_components × n_features)
    - E: residuals

    The algorithm alternates between:
    1. Fixing S^T, solving for C with constraints
    2. Fixing C, solving for S^T with constraints

    Rotational ambiguity: MCR solutions are not unique. Different (C, S^T)
    pairs can reconstruct X equally well. Use rotational ambiguity analysis
    to assess solution stability.

    References
    ----------
    Jaumot et al. (2015), de Juan & Tauler (2021).
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 50,
        tol: float = 1e-8,
        c_constraints: Optional[list[str]] = None,
        st_constraints: Optional[list[str]] = None,
        initialization: Literal["pca", "random"] = "pca",
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.c_constraints = c_constraints if c_constraints is not None else ["non_neg"]
        self.st_constraints = st_constraints if st_constraints is not None else ["non_neg"]
        self.initialization = initialization
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MCRALS:
        """
        Fit MCR-ALS model to mixture data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Mixture spectra.

        y : ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        self : MCRALS
            Fitted model.
        """
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=float, ensure_2d=True)
        n_samples, n_features = X.shape

        if self.n_components > min(n_samples, n_features):
            raise ValueError(f"n_components={self.n_components} exceeds data dimensions ({n_samples}, {n_features})")

        # Initialize S^T (pure spectra)
        ST = self._initialize_spectra(X)

        # ALS iterations
        self.lack_of_fit_ = []
        self.converged_ = False

        for iteration in range(self.max_iter):
            # Step 1: Fix S^T, solve for C
            C = self._solve_concentrations(X, ST)
            C = self._apply_constraints(C, self.c_constraints, axis=1)

            # Step 2: Fix C, solve for S^T
            ST = self._solve_spectra(X, C)
            ST = self._apply_constraints(ST, self.st_constraints, axis=0)

            # Compute lack-of-fit
            X_reconstructed = C @ ST
            lof = np.linalg.norm(X - X_reconstructed, "fro") / np.linalg.norm(X, "fro")
            self.lack_of_fit_.append(lof)

            # Check convergence
            if iteration > 0:
                delta_lof = abs(self.lack_of_fit_[-2] - self.lack_of_fit_[-1])
                if delta_lof < self.tol:
                    self.converged_ = True
                    break

        self.n_iter_ = iteration + 1
        self.components_ = ST  # Store pure spectra

        if not self.converged_:
            warnings.warn(
                f"MCR-ALS did not converge after {self.max_iter} iterations. Final LOF change: {delta_lof:.2e}",
                UserWarning,
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to concentration space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Mixture spectra.

        Returns
        -------
        C : ndarray of shape (n_samples, n_components)
            Concentration profiles.
        """
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=float, ensure_2d=True)

        # Solve for C given fixed S^T
        C = self._solve_concentrations(X, self.components_)
        C = self._apply_constraints(C, self.c_constraints, axis=1)

        return C

    def inverse_transform(self, C: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from concentration profiles.

        Parameters
        ----------
        C : ndarray of shape (n_samples, n_components)
            Concentration profiles.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed mixture spectra.
        """
        return C @ self.components_

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """
        Compute explained variance (R²) for data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Mixture spectra.

        y : ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        r2 : float
            Explained variance (0 to 1, higher is better).
        """
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=float, ensure_2d=True)
        X = self._validate_data(X, dtype=float, ensure_2d=True, reset=False)

        C = self.transform(X)
        X_reconstructed = self.inverse_transform(C)

        ss_res = np.sum((X - X_reconstructed) ** 2)
        ss_tot = np.sum((X - X.mean(axis=0)) ** 2)

        return 1 - (ss_res / ss_tot)

    def _initialize_spectra(self, X: np.ndarray) -> np.ndarray:
        """Initialize pure spectra S^T using PCA or random selection."""
        rng = np.random.default_rng(self.random_state)

        if self.initialization == "pca":
            # Use first n_components PC loadings
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            pca.fit(X)
            ST = pca.components_

            # Ensure non-negativity if required
            if "non_neg" in self.st_constraints:
                ST = np.abs(ST)

        elif self.initialization == "random":
            # Randomly select spectra from data
            n_samples = X.shape[0]
            indices = rng.choice(n_samples, size=self.n_components, replace=False)
            ST = X[indices, :]

        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")

        return ST

    def _solve_concentrations(self, X: np.ndarray, ST: np.ndarray) -> np.ndarray:
        """Solve for C given X and S^T: X ≈ C @ S^T."""
        # C = X @ S^T.T @ (S^T @ S^T.T)^{-1}
        # Equivalent to: C = X @ pinv(S^T)
        C = X @ np.linalg.pinv(ST)
        return C

    def _solve_spectra(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Solve for S^T given X and C: X ≈ C @ S^T."""
        # S^T = (C.T @ C)^{-1} @ C.T @ X
        # Equivalent to: S^T = pinv(C) @ X
        ST = np.linalg.pinv(C) @ X
        return ST

    def _apply_constraints(self, M: np.ndarray, constraints: list[str], axis: int = 1) -> np.ndarray:
        """
        Apply constraints to matrix M.

        Parameters
        ----------
        M : ndarray
            Matrix to constrain (C or S^T).
        constraints : list of str
            List of constraint names.
        axis : int, default=1
            Axis for normalization (0 for columns, 1 for rows).

        Returns
        -------
        M_constrained : ndarray
            Constrained matrix.
        """
        M_out = M.copy()

        for constraint in constraints:
            if constraint == "non_neg":
                M_out = np.maximum(M_out, 0)

            elif constraint == "norm":
                # Closure constraint (sum to 1)
                sums = M_out.sum(axis=axis, keepdims=True)
                sums = np.where(sums == 0, 1, sums)  # Avoid division by zero
                M_out = M_out / sums

            elif constraint == "unimodal":
                # Enforce unimodality along each profile
                if axis == 1:  # Row-wise (concentration profiles)
                    for i in range(M_out.shape[0]):
                        M_out[i, :] = self._enforce_unimodal(M_out[i, :])
                else:  # Column-wise (spectral profiles)
                    for j in range(M_out.shape[1]):
                        M_out[:, j] = self._enforce_unimodal(M_out[:, j])

            else:
                warnings.warn(f"Unknown constraint: {constraint}", UserWarning)

        return M_out

    def _enforce_unimodal(self, profile: np.ndarray) -> np.ndarray:
        """
        Enforce unimodality on a 1D profile.

        Simple algorithm: Find maximum, then enforce monotonic increase
        before max and monotonic decrease after max.
        """
        n = len(profile)
        max_idx = np.argmax(profile)

        # Monotonic increase up to max
        for i in range(1, max_idx + 1):
            if profile[i] < profile[i - 1]:
                profile[i] = profile[i - 1]

        # Monotonic decrease after max
        for i in range(max_idx + 1, n):
            if profile[i] > profile[i - 1]:
                profile[i] = profile[i - 1]

        return profile


class RotationalAmbiguityAnalysis:
    """
    Analyze rotational ambiguity in MCR solutions.

    MCR solutions are not unique: multiple (C, S^T) pairs can reconstruct X
    equally well. This class computes feasible regions for concentrations
    and spectra using band boundaries method.

    Parameters
    ----------
    mcr_model : MCRALS
        Fitted MCR-ALS model.

    Attributes
    ----------
    c_min_ : ndarray
        Minimum feasible concentrations.

    c_max_ : ndarray
        Maximum feasible concentrations.

    st_min_ : ndarray
        Minimum feasible spectra.

    st_max_ : ndarray
        Maximum feasible spectra.

    ambiguity_index_c_ : float
        Ambiguity index for concentrations (0 = unique, 1 = maximal ambiguity).

    ambiguity_index_st_ : float
        Ambiguity index for spectra.

    References
    ----------
    Abdollahi & Tauler (2011). Calculation and meaning of feasible band
    boundaries in multivariate curve resolution of a two-component system.
    Analytical Chemistry, 83(6), 2461-2468.

    Example
    -------
    >>> from foodspec.features.mcr_als import MCRALS, RotationalAmbiguityAnalysis
    >>>
    >>> mcr = MCRALS(n_components=2)
    >>> mcr.fit(X)
    >>>
    >>> ambiguity = RotationalAmbiguityAnalysis(mcr)
    >>> ambiguity.compute_band_boundaries(X)
    >>>
    >>> print(f"Concentration ambiguity: {ambiguity.ambiguity_index_c_:.3f}")
    >>> print(f"Spectra ambiguity: {ambiguity.ambiguity_index_st_:.3f}")
    """

    def __init__(self, mcr_model: MCRALS):
        if not hasattr(mcr_model, "components_"):
            raise ValueError("MCR model must be fitted before ambiguity analysis")

        self.mcr_model = mcr_model

    def compute_band_boundaries(
        self,
        X: np.ndarray,
        n_rotations: int = 100,
    ) -> RotationalAmbiguityAnalysis:
        """
        Compute feasible band boundaries for concentrations and spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Original mixture data.

        n_rotations : int, default=100
            Number of rotation matrices to sample.

        Returns
        -------
        self : RotationalAmbiguityAnalysis
            Analysis results stored in attributes.

        Notes
        -----
        Method: Generate random rotation matrices T, compute alternative
        solutions (C @ T, T^{-1} @ S^T), apply constraints, track min/max.
        """
        C_orig = self.mcr_model.transform(X)
        ST_orig = self.mcr_model.components_
        n_components = self.mcr_model.n_components

        # Initialize min/max arrays
        C_all = [C_orig]
        ST_all = [ST_orig]

        rng = np.random.default_rng(self.mcr_model.random_state)

        for _ in range(n_rotations):
            # Generate random rotation matrix (orthogonal transformation)
            T = self._random_rotation_matrix(n_components, rng)

            # Apply rotation
            C_rotated = C_orig @ T
            ST_rotated = np.linalg.inv(T) @ ST_orig

            # Apply constraints
            C_rotated = self.mcr_model._apply_constraints(C_rotated, self.mcr_model.c_constraints, axis=1)
            ST_rotated = self.mcr_model._apply_constraints(ST_rotated, self.mcr_model.st_constraints, axis=0)

            # Check if rotation maintains reconstruction quality
            X_reconstructed = C_rotated @ ST_rotated
            rel_error = np.linalg.norm(X - X_reconstructed, "fro") / np.linalg.norm(X, "fro")

            if rel_error < 1.1 * self.mcr_model.lack_of_fit_[-1]:
                # Accept this rotation as feasible
                C_all.append(C_rotated)
                ST_all.append(ST_rotated)

        # Compute band boundaries
        C_stack = np.stack(C_all, axis=0)
        ST_stack = np.stack(ST_all, axis=0)

        self.c_min_ = C_stack.min(axis=0)
        self.c_max_ = C_stack.max(axis=0)
        self.st_min_ = ST_stack.min(axis=0)
        self.st_max_ = ST_stack.max(axis=0)

        # Compute ambiguity indices (average relative range)
        self.ambiguity_index_c_ = np.mean((self.c_max_ - self.c_min_) / (self.c_max_ + 1e-10))
        self.ambiguity_index_st_ = np.mean((self.st_max_ - self.st_min_) / (self.st_max_ + 1e-10))

        return self

    def _random_rotation_matrix(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate random orthogonal rotation matrix via QR decomposition."""
        A = rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)

        # Ensure determinant = +1 (proper rotation)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1

        return Q

    def plot_band_boundaries(
        self,
        X: np.ndarray,
        component_idx: int = 0,
        sample_idx: Optional[int] = None,
    ):
        """
        Plot band boundaries for concentrations or spectra.

        Parameters
        ----------
        X : ndarray
            Original mixture data.
        component_idx : int, default=0
            Component index to plot.
        sample_idx : int, optional
            If provided, plot concentration profile for this sample.
            If None, plot spectral profile.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        fig, ax = plt.subplots(figsize=(10, 6))

        if sample_idx is not None:
            # Plot concentration band for specific sample
            C_orig = self.mcr_model.transform(X)
            x_axis = np.arange(self.mcr_model.n_components)

            ax.fill_between(
                x_axis, self.c_min_[sample_idx, :], self.c_max_[sample_idx, :], alpha=0.3, label="Feasible range"
            )
            ax.plot(x_axis, C_orig[sample_idx, :], "o-", label="Nominal solution")
            ax.set_xlabel("Component")
            ax.set_ylabel("Concentration")
            ax.set_title(f"Concentration Band Boundaries (Sample {sample_idx})")

        else:
            # Plot spectral band for specific component
            ST_orig = self.mcr_model.components_
            x_axis = np.arange(ST_orig.shape[1])

            ax.fill_between(
                x_axis,
                self.st_min_[component_idx, :],
                self.st_max_[component_idx, :],
                alpha=0.3,
                label="Feasible range",
            )
            ax.plot(x_axis, ST_orig[component_idx, :], "-", label="Nominal solution")
            ax.set_xlabel("Wavelength index")
            ax.set_ylabel("Intensity")
            ax.set_title(f"Spectral Band Boundaries (Component {component_idx})")

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig
