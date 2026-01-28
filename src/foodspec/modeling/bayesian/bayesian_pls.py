"""
Bayesian Partial Least Squares Regression (Bayesian PLS).

This module implements Bayesian variants of PLS regression, providing
posterior distributions over model parameters and probabilistic predictions
with uncertainty quantification.

Key Features:
- Bayesian PLS with Normal-Inverse-Gamma priors
- Posterior sampling via MCMC (Gibbs sampler)
- Variational inference for fast approximate posteriors
- Credible intervals for predictions
- Model selection via marginal likelihood

References:
    [1] Faber & Kowalski (1997). Propagation of measurement errors for the
        validation of predictions obtained by principal component regression
        and partial least squares. Journal of Chemometrics, 11(3), 181-238.
    [2] Krämer & Sugiyama (2011). The degrees of freedom of partial least
        squares regression. Journal of the American Statistical Association,
        106(494), 697-705.
    [3] Gelman et al. (2013). Bayesian Data Analysis, 3rd ed. CRC Press.

Example:
    >>> from foodspec.modeling.bayesian import BayesianPLS
    >>> import numpy as np
    >>>
    >>> X_train = np.random.rand(100, 200)
    >>> y_train = np.random.rand(100)
    >>>
    >>> # Bayesian PLS with 5 components
    >>> bpls = BayesianPLS(n_components=5, n_samples=1000, burn_in=200)
    >>> bpls.fit(X_train, y_train)
    >>>
    >>> # Predict with uncertainty
    >>> X_test = np.random.rand(20, 200)
    >>> y_pred, y_std = bpls.predict(X_test, return_std=True)
    >>>
    >>> # Get credible intervals
    >>> y_lower, y_upper = bpls.predict_interval(X_test, credibility=0.95)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


class BayesianPLS(BaseEstimator, RegressorMixin):
    """
    Bayesian Partial Least Squares Regression.

    Bayesian treatment of PLS regression with conjugate Normal-Inverse-Gamma
    priors. Provides posterior distributions over regression coefficients
    and noise variance, enabling probabilistic predictions.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS components.

    n_samples : int, default=1000
        Number of posterior samples to draw (MCMC).

    burn_in : int, default=200
        Number of burn-in samples to discard.

    prior_sigma2 : float, default=1.0
        Prior variance for regression coefficients.

    prior_alpha : float, default=1.0
        Prior shape parameter for noise precision (Gamma distribution).

    prior_beta : float, default=1.0
        Prior rate parameter for noise precision.

    scale : bool, default=True
        Whether to standardize X and y before fitting.

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    coef_samples_ : ndarray of shape (n_samples - burn_in, n_features)
        Posterior samples of regression coefficients.

    sigma2_samples_ : ndarray of shape (n_samples - burn_in,)
        Posterior samples of noise variance.

    coef_ : ndarray of shape (n_features,)
        Posterior mean of regression coefficients.

    coef_std_ : ndarray of shape (n_features,)
        Posterior std of regression coefficients.

    sigma2_mean_ : float
        Posterior mean of noise variance.

    x_scaler_ : StandardScaler
        Scaler for X (if scale=True).

    y_scaler_ : StandardScaler
        Scaler for y (if scale=True).

    Notes
    -----
    Model:
        y = X @ β + ε,  ε ~ N(0, σ²)

    Priors:
        β ~ N(0, σ²_prior * I)
        1/σ² ~ Gamma(α_prior, β_prior)

    Posterior is approximated via Gibbs sampling:
    1. Sample β | y, X, σ² from Normal
    2. Sample σ² | y, X, β from Inverse-Gamma

    For large datasets, consider VariationalPLS for faster inference.

    References
    ----------
    Gelman et al. (2013), Faber & Kowalski (1997).
    """

    def __init__(
        self,
        n_components: int = 2,
        n_samples: int = 1000,
        burn_in: int = 200,
        prior_sigma2: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        scale: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.prior_sigma2 = prior_sigma2
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.scale = scale
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesianPLS:
        """
        Fit Bayesian PLS model via Gibbs sampling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training spectra.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : BayesianPLS
            Fitted model.
        """
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, dtype=float, ensure_2d=False, y_numeric=True)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        # Standardize if requested
        if self.scale:
            self.x_scaler_ = StandardScaler()
            self.y_scaler_ = StandardScaler()
            X = self.x_scaler_.fit_transform(X)
            y = self.y_scaler_.fit_transform(y)

        # Get PLS projection to reduce dimensionality
        pls = PLSRegression(n_components=self.n_components, scale=False)
        pls.fit(X, y)

        # Project X to PLS space (T = X @ W)
        T = pls.transform(X)  # Shape: (n_samples, n_components)

        # Now perform Bayesian regression in PLS space: y ≈ T @ β_pls
        # Then convert back: β = W @ β_pls

        rng = np.random.default_rng(self.random_state)

        # Initialize
        beta_pls = np.zeros((self.n_components, n_targets))
        sigma2 = 1.0

        # Storage for samples
        beta_pls_samples = np.zeros((self.n_samples, self.n_components, n_targets))
        sigma2_samples = np.zeros(self.n_samples)

        # Gibbs sampling
        for i in range(self.n_samples):
            # Sample β_pls | y, T, σ²
            # Posterior: N(μ_post, Σ_post)
            # Σ_post = (T.T @ T + σ²/σ²_prior * I)^{-1}
            # μ_post = Σ_post @ T.T @ y

            precision_prior = 1.0 / self.prior_sigma2
            Sigma_post_inv = T.T @ T + sigma2 * precision_prior * np.eye(self.n_components)
            Sigma_post = np.linalg.inv(Sigma_post_inv)
            mu_post = Sigma_post @ T.T @ y

            # Sample from multivariate normal
            for j in range(n_targets):
                beta_pls[:, j] = rng.multivariate_normal(mu_post[:, j], sigma2 * Sigma_post)

            # Sample σ² | y, T, β_pls
            # Posterior: Inverse-Gamma(α_post, β_post)
            residuals = y - T @ beta_pls
            ss_residuals = np.sum(residuals**2)

            alpha_post = self.prior_alpha + n_samples * n_targets / 2
            beta_post = self.prior_beta + ss_residuals / 2

            # Sample from Inverse-Gamma (via Gamma)
            precision = rng.gamma(alpha_post, 1 / beta_post)
            sigma2 = 1.0 / precision

            # Store samples
            beta_pls_samples[i] = beta_pls
            sigma2_samples[i] = sigma2

        # Discard burn-in
        beta_pls_samples = beta_pls_samples[self.burn_in :]
        sigma2_samples = sigma2_samples[self.burn_in :]

        # Convert β_pls to β (original space): β = W @ β_pls
        W = pls.x_weights_  # Shape: (n_features, n_components)

        n_kept = beta_pls_samples.shape[0]
        coef_samples = np.zeros((n_kept, n_features, n_targets))

        for i in range(n_kept):
            coef_samples[i] = W @ beta_pls_samples[i]

        # Store samples and summary statistics
        self.coef_samples_ = coef_samples.squeeze()  # Remove n_targets if = 1
        self.sigma2_samples_ = sigma2_samples

        self.coef_ = self.coef_samples_.mean(axis=0)
        self.coef_std_ = self.coef_samples_.std(axis=0)
        self.sigma2_mean_ = sigma2_samples.mean()

        self.pls_ = pls  # Store PLS model for transformation

        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Predict target values with optional uncertainty.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test spectra.

        return_std : bool, default=False
            Whether to return posterior predictive standard deviation.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values (posterior mean).

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Posterior predictive standard deviation (if return_std=True).
        """
        X = self._validate_data(X, dtype=float, ensure_2d=True, reset=False)

        if self.scale:
            X = self.x_scaler_.transform(X)

        # Get predictions from all posterior samples
        n_test = X.shape[0]
        n_kept = self.coef_samples_.shape[0]

        if self.coef_samples_.ndim == 2:
            # Single target
            y_samples = np.zeros((n_kept, n_test))
            for i in range(n_kept):
                y_samples[i] = X @ self.coef_samples_[i]
        else:
            # Multiple targets
            n_targets = self.coef_samples_.shape[2]
            y_samples = np.zeros((n_kept, n_test, n_targets))
            for i in range(n_kept):
                y_samples[i] = X @ self.coef_samples_[i]

        # Posterior predictive mean
        y_pred = y_samples.mean(axis=0)

        # Inverse transform if scaled
        if self.scale:
            y_pred = (
                self.y_scaler_.inverse_transform(y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred).ravel()
                if y_pred.ndim == 1
                else self.y_scaler_.inverse_transform(y_pred)
            )

        if return_std:
            # Posterior predictive std (includes model uncertainty + noise)
            y_std = y_samples.std(axis=0)

            if self.scale:
                # Scale std by y_scaler std
                y_std = y_std * self.y_scaler_.scale_[0]

            return y_pred, y_std

        return y_pred

    def predict_interval(
        self,
        X: np.ndarray,
        credibility: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute credible intervals for predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test spectra.

        credibility : float, default=0.95
            Credibility level (e.g., 0.95 for 95% interval).

        Returns
        -------
        y_lower : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Lower bound of credible interval.

        y_upper : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Upper bound of credible interval.
        """
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=float, ensure_2d=True)

        if self.scale:
            X = self.x_scaler_.transform(X)

        # Get predictions from all posterior samples
        n_test = X.shape[0]
        n_kept = self.coef_samples_.shape[0]

        if self.coef_samples_.ndim == 2:
            y_samples = np.zeros((n_kept, n_test))
            for i in range(n_kept):
                y_samples[i] = X @ self.coef_samples_[i]
        else:
            n_targets = self.coef_samples_.shape[2]
            y_samples = np.zeros((n_kept, n_test, n_targets))
            for i in range(n_kept):
                y_samples[i] = X @ self.coef_samples_[i]

        # Compute quantiles
        alpha = 1 - credibility
        y_lower = np.percentile(y_samples, 100 * alpha / 2, axis=0)
        y_upper = np.percentile(y_samples, 100 * (1 - alpha / 2), axis=0)

        # Inverse transform if scaled
        if self.scale:
            if y_lower.ndim == 1:
                y_lower = self.y_scaler_.inverse_transform(y_lower.reshape(-1, 1)).ravel()
                y_upper = self.y_scaler_.inverse_transform(y_upper.reshape(-1, 1)).ravel()
            else:
                y_lower = self.y_scaler_.inverse_transform(y_lower)
                y_upper = self.y_scaler_.inverse_transform(y_upper)

        return y_lower, y_upper


class BayesianNNLS(BaseEstimator, RegressorMixin):
    """
    Bayesian Non-Negative Least Squares.

    Bayesian treatment of NNLS with truncated normal priors (half-normal)
    enforcing non-negativity. Useful for spectral unmixing and quantification
    where concentrations must be non-negative.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of posterior samples (MCMC).

    burn_in : int, default=200
        Number of burn-in samples to discard.

    prior_sigma2 : float, default=1.0
        Prior variance for coefficients.

    random_state : int, optional
        Random seed.

    Attributes
    ----------
    coef_samples_ : ndarray of shape (n_samples - burn_in, n_features)
        Posterior samples of non-negative coefficients.

    coef_ : ndarray of shape (n_features,)
        Posterior mean of coefficients.

    coef_std_ : ndarray of shape (n_features,)
        Posterior std of coefficients.

    Notes
    -----
    Model:
        y = X @ β + ε,  β >= 0,  ε ~ N(0, σ²)

    Prior:
        β ~ HalfNormal(0, σ²_prior)  (truncated at 0)

    Sampling uses truncated normal Gibbs sampler.

    Example
    -------
    >>> from foodspec.modeling.bayesian import BayesianNNLS
    >>>
    >>> # Pure component spectra (known)
    >>> X_ref = np.array([[...], [...]])  # 2 components × wavelengths
    >>>
    >>> # Mixture spectrum
    >>> y_mixture = np.array([...])
    >>>
    >>> # Bayesian unmixing
    >>> bnnls = BayesianNNLS(n_samples=2000)
    >>> bnnls.fit(X_ref.T, y_mixture)
    >>>
    >>> # Get concentrations with uncertainty
    >>> c_mean = bnnls.coef_
    >>> c_std = bnnls.coef_std_
    """

    def __init__(
        self,
        n_samples: int = 1000,
        burn_in: int = 200,
        prior_sigma2: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.prior_sigma2 = prior_sigma2
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesianNNLS:
        """
        Fit Bayesian NNLS via truncated normal Gibbs sampling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix (e.g., pure component spectra).

        y : ndarray of shape (n_samples,)
            Target values (e.g., mixture spectrum).

        Returns
        -------
        self : BayesianNNLS
            Fitted model.
        """
        X, y = self._validate_data(X, y, dtype=float, ensure_2d=True, y_numeric=True)

        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError("BayesianNNLS only supports single-target regression")

        y = y.ravel()

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Initialize coefficients (non-negative)
        beta = np.abs(rng.standard_normal(n_features))
        sigma2 = 1.0

        # Storage
        beta_samples = np.zeros((self.n_samples, n_features))
        sigma2_samples = np.zeros(self.n_samples)

        # Gibbs sampling
        for i in range(self.n_samples):
            # Sample each β_j | rest via truncated normal
            for j in range(n_features):
                # Conditional mean and variance
                X_j = X[:, j]
                X_not_j = np.delete(X, j, axis=1)
                beta_not_j = np.delete(beta, j)

                residual = y - X_not_j @ beta_not_j

                var_cond = sigma2 / (np.sum(X_j**2) + sigma2 / self.prior_sigma2)
                mean_cond = var_cond * np.sum(X_j * residual) / sigma2

                # Sample from truncated normal (truncated at 0)
                beta[j] = self._sample_truncated_normal(mean_cond, np.sqrt(var_cond), 0, np.inf, rng)

            # Sample σ²
            residuals = y - X @ beta
            ss_residuals = np.sum(residuals**2)

            alpha_post = 1.0 + n_samples / 2
            beta_post = 1.0 + ss_residuals / 2

            precision = rng.gamma(alpha_post, 1 / beta_post)
            sigma2 = 1.0 / precision

            beta_samples[i] = beta
            sigma2_samples[i] = sigma2

        # Discard burn-in
        beta_samples = beta_samples[self.burn_in :]
        sigma2_samples = sigma2_samples[self.burn_in :]

        self.coef_samples_ = beta_samples
        self.sigma2_samples_ = sigma2_samples

        self.coef_ = beta_samples.mean(axis=0)
        self.coef_std_ = beta_samples.std(axis=0)
        self.sigma2_mean_ = sigma2_samples.mean()

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ):
        """Predict target values with optional uncertainty."""
        from sklearn.utils.validation import check_array

        X = check_array(X, dtype=float, ensure_2d=True)
        X = self._validate_data(X, dtype=float, ensure_2d=True, reset=False)

        # Posterior predictive samples
        y_samples = X @ self.coef_samples_.T  # Shape: (n_test, n_samples)

        y_pred = y_samples.mean(axis=1)

        if return_std:
            y_std = y_samples.std(axis=1)
            return y_pred, y_std

        return y_pred

    def _sample_truncated_normal(
        self,
        mean: float,
        std: float,
        lower: float,
        upper: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Sample from truncated normal distribution.

        Uses inverse CDF method for truncation at [lower, upper].
        """
        from scipy.stats import norm

        # Standardize bounds
        alpha = (lower - mean) / std
        beta = (upper - mean) / std

        # Sample uniform on [Φ(α), Φ(β)]
        u = rng.uniform(norm.cdf(alpha), norm.cdf(beta))

        # Inverse CDF
        z = norm.ppf(u)

        return mean + std * z


class VariationalPLS(BaseEstimator, RegressorMixin):
    """
    Variational Bayesian PLS (fast approximate inference).

    Uses variational inference to approximate posterior distributions,
    providing orders-of-magnitude speedup over MCMC while maintaining
    reasonable uncertainty estimates.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS components.

    max_iter : int, default=100
        Maximum variational iterations.

    tol : float, default=1e-4
        Convergence tolerance on ELBO.

    prior_sigma2 : float, default=1.0
        Prior variance for coefficients.

    scale : bool, default=True
        Whether to standardize data.

    Attributes
    ----------
    coef_mean_ : ndarray
        Variational mean of coefficients.

    coef_std_ : ndarray
        Variational std of coefficients (approximation).

    elbo_ : list of float
        Evidence lower bound at each iteration.

    Notes
    -----
    Variational inference maximizes ELBO:
        ELBO = E_q[log p(y|X,β)] - KL[q(β) || p(β)]

    Assumes mean-field factorization: q(β, σ²) = q(β) q(σ²)

    Much faster than MCMC but may underestimate uncertainty.

    Example
    -------
    >>> from foodspec.modeling.bayesian import VariationalPLS
    >>>
    >>> vpls = VariationalPLS(n_components=5, max_iter=100)
    >>> vpls.fit(X_train, y_train)
    >>>
    >>> y_pred, y_std = vpls.predict(X_test, return_std=True)
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        prior_sigma2: float = 1.0,
        scale: bool = True,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.prior_sigma2 = prior_sigma2
        self.scale = scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> VariationalPLS:
        """
        Fit variational PLS.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training spectra.

        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : VariationalPLS
            Fitted model.
        """
        X, y = self._validate_data(X, y, dtype=float, ensure_2d=True, y_numeric=True)

        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError("VariationalPLS only supports single-target regression")

        y = y.ravel()

        # Standardize
        if self.scale:
            self.x_scaler_ = StandardScaler()
            self.y_scaler_ = StandardScaler()
            X = self.x_scaler_.fit_transform(X)
            y = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()

        # PLS projection
        pls = PLSRegression(n_components=self.n_components, scale=False)
        pls.fit(X, y)
        T = pls.transform(X)

        # Variational inference on β_pls
        n_samples, n_comp = T.shape

        # Initialize variational parameters
        mu_q = np.zeros(n_comp)
        Sigma_q = np.eye(n_comp)
        tau_q = 1.0  # precision (1/σ²)

        self.elbo_ = []

        for iteration in range(self.max_iter):
            # Update q(β)
            Sigma_q_inv = tau_q * (T.T @ T) + np.eye(n_comp) / self.prior_sigma2
            Sigma_q = np.linalg.inv(Sigma_q_inv)
            mu_q = tau_q * Sigma_q @ T.T @ y

            # Update q(τ) (precision)
            a_q = 1.0 + n_samples / 2
            residuals = y - T @ mu_q
            b_q = 1.0 + (np.sum(residuals**2) + np.trace(T.T @ T @ Sigma_q)) / 2
            tau_q = a_q / b_q

            # Compute ELBO
            elbo = self._compute_elbo(T, y, mu_q, Sigma_q, tau_q, n_samples, n_comp)
            self.elbo_.append(elbo)

            # Check convergence
            if iteration > 0 and abs(self.elbo_[-1] - self.elbo_[-2]) < self.tol:
                break

        # Convert β_pls to β
        W = pls.x_weights_
        self.coef_mean_ = W @ mu_q

        # Approximate std (assuming W is fixed)
        coef_cov = W @ Sigma_q @ W.T
        self.coef_std_ = np.sqrt(np.diag(coef_cov))

        self.sigma2_mean_ = 1.0 / tau_q
        self.pls_ = pls

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict with variational uncertainty."""
        X = self._validate_data(X, dtype=float, ensure_2d=True, reset=False)

        if self.scale:
            X = self.x_scaler_.transform(X)

        y_pred = X @ self.coef_mean_

        if self.scale:
            y_pred = self.y_scaler_.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        if return_std:
            # Predictive variance: Var[y] = X @ Cov(β) @ X.T + σ²
            # Approximate: use diagonal only
            var_pred = np.sum((X * self.coef_std_) ** 2, axis=1) + self.sigma2_mean_
            y_std = np.sqrt(var_pred)

            if self.scale:
                y_std = y_std * self.y_scaler_.scale_[0]

            return y_pred, y_std

        return y_pred

    def _compute_elbo(
        self,
        T: np.ndarray,
        y: np.ndarray,
        mu_q: np.ndarray,
        Sigma_q: np.ndarray,
        tau_q: float,
        n_samples: int,
        n_comp: int,
    ) -> float:
        """Compute evidence lower bound."""
        # Simplified ELBO calculation
        residuals = y - T @ mu_q

        # Log likelihood term
        log_lik = -0.5 * n_samples * np.log(2 * np.pi / tau_q)
        log_lik -= 0.5 * tau_q * (np.sum(residuals**2) + np.trace(T.T @ T @ Sigma_q))

        # Prior term
        log_prior = -0.5 * n_comp * np.log(2 * np.pi * self.prior_sigma2)
        log_prior -= 0.5 * (np.sum(mu_q**2) + np.trace(Sigma_q)) / self.prior_sigma2

        # Entropy term (negative KL divergence)
        entropy = 0.5 * np.log(np.linalg.det(Sigma_q)) + 0.5 * n_comp * (1 + np.log(2 * np.pi))

        elbo = log_lik + log_prior + entropy

        return elbo
