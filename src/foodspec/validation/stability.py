"""Bootstrap stability and confidence interval analysis for models."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

__all__ = ["BootstrapStability", "StabilityIndex"]


class BootstrapStability:
    """
    Assess model stability via bootstrap resampling.

    Quantifies how parameter estimates vary under resampling.
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
        random_state: Optional[int] = None,
    ):
        """
        Initialize BootstrapStability.

        Parameters
        ----------
        n_bootstrap : int, default=100
            Number of bootstrap samples.
        confidence : float, default=0.95
            Confidence level for intervals (0 to 1).
        random_state : int, optional
            Random seed.
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.random_state = random_state

    def assess_parameter_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_func: Callable,
        param_func: Callable,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assess parameter stability via bootstrap.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target variable.
        fit_func : callable
            Function to fit model: fit_func(X, y) -> model.
        param_func : callable
            Extract parameters: param_func(model) -> params.

        Returns
        -------
        param_mean : np.ndarray
            Mean parameter values.
        param_std : np.ndarray
            Standard deviation of parameters.
        param_ci : np.ndarray, shape (n_params, 2)
            Confidence intervals [lower, upper].

        Examples
        --------
        >>> from sklearn.linear_model import Ridge
        >>> bs = BootstrapStability(n_bootstrap=100)
        >>> X, y = np.random.randn(100, 10), np.random.randn(100)
        >>> mean, std, ci = bs.assess_parameter_stability(
        ...     X, y,
        ...     fit_func=lambda x, yy: Ridge().fit(x, yy),
        ...     param_func=lambda m: m.coef_
        ... )
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        params_bootstrap = []

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            try:
                model = fit_func(X_boot, y_boot)
                params = param_func(model)
                params_bootstrap.append(params)
            except Exception:
                continue

        if not params_bootstrap:
            raise ValueError("Bootstrap sampling failed")

        params_bootstrap = np.array(params_bootstrap)

        param_mean = np.mean(params_bootstrap, axis=0)
        param_std = np.std(params_bootstrap, axis=0, ddof=1)

        alpha = 1 - self.confidence
        q_lower = alpha / 2
        q_upper = 1 - alpha / 2

        param_ci = np.column_stack([
            np.percentile(params_bootstrap, q_lower * 100, axis=0),
            np.percentile(params_bootstrap, q_upper * 100, axis=0),
        ])

        self.bootstrap_samples_ = params_bootstrap

        return param_mean, param_std, param_ci

    def assess_prediction_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_func: Callable,
        X_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assess prediction stability via bootstrap.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training targets.
        fit_func : callable
            Function to fit model: fit_func(X, y) -> model.
        X_test : np.ndarray, optional
            Test features. If None, uses X.

        Returns
        -------
        pred_mean : np.ndarray
            Mean predictions.
        pred_std : np.ndarray
            Standard deviation of predictions.
        pred_ci : np.ndarray, shape (n_samples, 2)
            Confidence intervals.
        """
        if X_test is None:
            X_test = X

        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        n_test = X_test.shape[0]

        preds_bootstrap = np.zeros((self.n_bootstrap, n_test))

        for i in range(self.n_bootstrap):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            try:
                model = fit_func(X_boot, y_boot)
                preds = model.predict(X_test)
                preds_bootstrap[i] = preds
            except Exception:
                continue

        pred_mean = np.mean(preds_bootstrap, axis=0)
        pred_std = np.std(preds_bootstrap, axis=0, ddof=1)

        alpha = 1 - self.confidence
        q_lower = alpha / 2
        q_upper = 1 - alpha / 2

        pred_ci = np.column_stack([
            np.percentile(preds_bootstrap, q_lower * 100, axis=0),
            np.percentile(preds_bootstrap, q_upper * 100, axis=0),
        ])

        self.bootstrap_predictions_ = preds_bootstrap

        return pred_mean, pred_std, pred_ci


class StabilityIndex:
    """Calculate various stability indices for models."""

    @staticmethod
    def jackknife_resampling(
        X: np.ndarray,
        y: np.ndarray,
        fit_func: Callable,
        param_func: Callable,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jackknife resampling (leave-one-out).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray
            Target variable.
        fit_func : callable
            Function to fit model.
        param_func : callable
            Extract parameters.

        Returns
        -------
        param_mean : np.ndarray
            Mean parameter values.
        param_std : np.ndarray
            Standard deviation (jackknife estimate).
        """
        n_samples = X.shape[0]
        params_jack = []

        for i in range(n_samples):
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False
            X_jack = X[mask]
            y_jack = y[mask]

            try:
                model = fit_func(X_jack, y_jack)
                params = param_func(model)
                params_jack.append(params)
            except Exception:
                continue

        if not params_jack:
            raise ValueError("Jackknife resampling failed")

        params_jack = np.array(params_jack)
        param_mean = np.mean(params_jack, axis=0)

        # Jackknife variance estimate
        param_var = ((n_samples - 1) / n_samples) * np.sum(
            (params_jack - param_mean) ** 2, axis=0
        )
        param_std = np.sqrt(param_var)

        return param_mean, param_std

    @staticmethod
    def parameter_stability_ratio(
        bootstrap_std: np.ndarray,
        parameter_estimates: np.ndarray,
    ) -> np.ndarray:
        """
        Compute parameter stability ratio (coefficient of variation).

        Parameters
        ----------
        bootstrap_std : np.ndarray
            Standard deviation from bootstrap.
        parameter_estimates : np.ndarray
            Estimated parameters.

        Returns
        -------
        stability_ratio : np.ndarray
            Stability ratio (std / |estimate|).
        """
        return bootstrap_std / (np.abs(parameter_estimates) + 1e-8)

    @staticmethod
    def model_reproducibility_index(
        predictions_1: np.ndarray,
        predictions_2: np.ndarray,
    ) -> float:
        """
        Calculate reproducibility index between two prediction sets.

        Parameters
        ----------
        predictions_1 : np.ndarray
            Predictions from model 1.
        predictions_2 : np.ndarray
            Predictions from model 2.

        Returns
        -------
        r_rep : float
            Reproducibility index (0 to 1).

        Notes
        -----
        r_rep = 1 - sqrt(sum((y1 - y2)^2) / sum((y1 - mean(y1))^2))
        """
        ss_diff = np.sum((predictions_1 - predictions_2) ** 2)
        ss_total = np.sum((predictions_1 - np.mean(predictions_1)) ** 2)

        return max(0, 1 - np.sqrt(ss_diff / (ss_total + 1e-8)))

    @staticmethod
    def sensitivity_index(
        X: np.ndarray,
        y: np.ndarray,
        fit_func: Callable,
    ) -> np.ndarray:
        """
        Calculate feature sensitivity indices.

        Measures how much model performance changes with feature perturbation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray
            Target variable.
        fit_func : callable
            Function to fit and evaluate model (returns MSE or similar).

        Returns
        -------
        sensitivity : np.ndarray, shape (n_features,)
            Sensitivity index for each feature.
        """
        n_features = X.shape[1]
        sensitivity = np.zeros(n_features)

        baseline_error = fit_func(X, y)

        for j in range(n_features):
            X_perturbed = X.copy()
            X_perturbed[:, j] += 0.1 * X[:, j].std()

            perturbed_error = fit_func(X_perturbed, y)
            sensitivity[j] = (perturbed_error - baseline_error) / (baseline_error + 1e-8)

        return np.abs(sensitivity)

    @staticmethod
    def plot_stability(
        bootstrap_samples: np.ndarray,
        feature_names: Optional[list] = None,
        ax=None,
    ):
        """
        Plot bootstrap distribution of parameters.

        Parameters
        ----------
        bootstrap_samples : np.ndarray, shape (n_bootstrap, n_features)
            Bootstrap samples.
        feature_names : list, optional
            Feature names.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        n_features = bootstrap_samples.shape[1]
        if feature_names is None:
            feature_names = [f"Param {i}" for i in range(n_features)]

        ax.boxplot([bootstrap_samples[:, i] for i in range(n_features)])
        ax.set_xticklabels(feature_names)
        ax.set_ylabel("Parameter Value")
        ax.set_title("Bootstrap Distribution of Parameters")
        ax.grid(True, alpha=0.3)

        return ax
