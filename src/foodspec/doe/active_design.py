"""Active Design of Experiments via Bayesian Optimization.

Implements intelligent sample selection strategies for spectroscopy experiments.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C


class AcquisitionFunction:
    """Acquisition functions for Bayesian optimization."""

    @staticmethod
    def expected_improvement(
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        y_best: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """Expected Improvement (EI) acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        with np.errstate(divide="warn"):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.ravel()

    @staticmethod
    def upper_confidence_bound(
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        kappa: float = 2.0,
    ) -> np.ndarray:
        """Upper Confidence Bound (UCB) acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        return mu + kappa * sigma

    @staticmethod
    def probability_of_improvement(
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        y_best: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """Probability of Improvement (PI) acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        with np.errstate(divide="warn"):
            Z = (mu - y_best - xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0

        return pi.ravel()


class BayesianOptimizer:
    """Bayesian Optimization for intelligent sample selection.

    Parameters
    ----------
    acquisition : {'ei', 'ucb', 'pi'}, default='ei'
        Acquisition function type.
    kernel : optional
        GP kernel (default: RBF).
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        acquisition: Literal["ei", "ucb", "pi"] = "ei",
        kernel=None,
        random_state: Optional[int] = None,
    ):
        self.acquisition = acquisition
        self.random_state = random_state

        if kernel is None:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=random_state,
            normalize_y=True,
        )

        self.X_observed = []
        self.y_observed = []

    def suggest(
        self,
        X_candidates: np.ndarray,
        n_suggestions: int = 1,
    ) -> np.ndarray:
        """
        Suggest next samples to measure.

        Parameters
        ----------
        X_candidates : ndarray of shape (n_candidates, n_features)
            Pool of candidate samples.
        n_suggestions : int, default=1
            Number of samples to suggest.

        Returns
        -------
        X_suggested : ndarray of shape (n_suggestions, n_features)
            Suggested samples.
        """
        if len(self.X_observed) == 0:
            # Random initialization
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(X_candidates), size=n_suggestions, replace=False)
            return X_candidates[indices]

        # Fit GP on observed data
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed).reshape(-1, 1)
        self.gp.fit(X_obs, y_obs)

        # Compute acquisition function
        if self.acquisition == "ei":
            acq_values = AcquisitionFunction.expected_improvement(X_candidates, self.gp, y_obs.max())
        elif self.acquisition == "ucb":
            acq_values = AcquisitionFunction.upper_confidence_bound(X_candidates, self.gp)
        elif self.acquisition == "pi":
            acq_values = AcquisitionFunction.probability_of_improvement(X_candidates, self.gp, y_obs.max())

        # Select top suggestions
        top_indices = np.argsort(acq_values)[-n_suggestions:][::-1]

        return X_candidates[top_indices]

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Update with new observations."""
        self.X_observed.extend(X_new.tolist())
        self.y_observed.extend(y_new.tolist())


class ActiveDesign:
    """Active learning design for spectral experiments.

    Example
    -------
    >>> from foodspec.doe import ActiveDesign
    >>>
    >>> design = ActiveDesign(acquisition='ei')
    >>> X_next = design.suggest(X_candidates, n_suggestions=5)
    >>> # Measure X_next...
    >>> design.update(X_next, y_measured)
    """

    def __init__(
        self,
        acquisition: str = "ei",
        random_state: Optional[int] = None,
    ):
        self.optimizer = BayesianOptimizer(
            acquisition=acquisition,
            random_state=random_state,
        )

    def suggest(self, X_candidates: np.ndarray, n_suggestions: int = 1) -> np.ndarray:
        """Suggest next samples."""
        return self.optimizer.suggest(X_candidates, n_suggestions)

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Update with observations."""
        self.optimizer.update(X_new, y_new)
