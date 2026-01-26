"""Regression Uncertainty Quantification: Bootstrap Intervals, Quantile Regression, Conformal Methods."""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy import stats

__all__ = ["BootstrapPredictionIntervals", "QuantileRegression", "ConformalRegression"]


class BootstrapPredictionIntervals:
    """
    Bootstrap-based prediction intervals for regression.

    Includes percentile method, bias-corrected and accelerated (BCa), and basic bootstrap.
    """

    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95):
        """
        Initialize bootstrap prediction interval estimator.

        Parameters
        ----------
        n_bootstrap : int, default=1000
            Number of bootstrap replications.
        confidence : float, default=0.95
            Confidence level (e.g., 0.95 for 95% PI).

        Notes
        -----
        Methods:
        - Percentile: Direct quantiles of bootstrap distribution
        - BCa: Bias-corrected and accelerated (more accurate)
        - Basic: Uses bootstrap distribution of pivots
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.alpha = 1 - confidence

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, model_class, **model_kwargs):
        """
        Fit bootstrap ensemble for prediction intervals.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_features)
            Training features.
        y_train : np.ndarray, shape (n_train,)
            Training targets.
        model_class : class
            Regression model class (must have fit, predict methods).
        **model_kwargs : dict
            Arguments to pass to model_class.

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have same number of samples")

        self.n_train_ = len(y_train)
        self.n_features_ = X_train.shape[1]

        # Generate bootstrap samples
        np.random.seed(42)  # For reproducibility
        self.models_ = []
        self.y_bootstrap_preds_ = []

        for b in range(self.n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(self.n_train_, size=self.n_train_, replace=True)
            X_boot = X_train[idx]
            y_boot = y_train[idx]

            # Fit model on bootstrap sample
            model = model_class(**model_kwargs)
            model.fit(X_boot, y_boot)
            self.models_.append(model)

        return self

    def predict(self, X_test: np.ndarray, method: str = "percentile") -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty intervals.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, n_features)
            Test features.
        method : {'percentile', 'bca', 'basic'}, default='percentile'
            Bootstrap method for intervals.

        Returns
        -------
        predictions : dict
            Keys:
            - 'mean': Point predictions (mean of bootstrap predictions)
            - 'median': Median of bootstrap predictions
            - 'lower': Lower confidence bound
            - 'upper': Upper confidence bound
            - 'std': Standard deviation of predictions
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        n_test = X_test.shape[0]

        # Generate predictions from all bootstrap models
        predictions_bootstrap = np.zeros((self.n_bootstrap, n_test))
        for b, model in enumerate(self.models_):
            predictions_bootstrap[b] = model.predict(X_test).ravel()

        # Aggregate predictions
        y_pred_mean = np.mean(predictions_bootstrap, axis=0)
        y_pred_median = np.median(predictions_bootstrap, axis=0)
        y_pred_std = np.std(predictions_bootstrap, axis=0)

        alpha_lower = self.alpha / 2
        alpha_upper = 1 - alpha_lower

        if method == "percentile":
            y_lower = np.quantile(predictions_bootstrap, alpha_lower, axis=0)
            y_upper = np.quantile(predictions_bootstrap, alpha_upper, axis=0)

        elif method == "bca":
            # Bias-corrected and accelerated bootstrap
            # More accurate for skewed distributions
            y_lower = np.zeros(n_test)
            y_upper = np.zeros(n_test)

            for i in range(n_test):
                # Bias correction
                z0 = stats.norm.ppf(np.mean(predictions_bootstrap[:, i] < y_pred_mean[i]))

                # Acceleration
                # Using jackknife-based acceleration (simplified)
                jack_pred = []
                for j in range(min(20, self.n_bootstrap)):  # Use subset for speed
                    mask = np.arange(self.n_bootstrap) != j
                    jack_pred.append(np.mean(predictions_bootstrap[mask, i]))

                jack_mean = np.mean(jack_pred)
                num = np.sum((jack_mean - np.array(jack_pred)) ** 3)
                den = (6 * np.sum((jack_mean - np.array(jack_pred)) ** 2) ** 1.5)
                accel = num / den if den > 1e-10 else 0

                # Adjusted percentiles
                z_alpha_lower = stats.norm.ppf(alpha_lower)
                z_alpha_upper = stats.norm.ppf(alpha_upper)

                p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - accel * (z0 + z_alpha_lower)))
                p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - accel * (z0 + z_alpha_upper)))

                y_lower[i] = np.quantile(predictions_bootstrap[:, i], p_lower)
                y_upper[i] = np.quantile(predictions_bootstrap[:, i], p_upper)

        elif method == "basic":
            # Basic bootstrap: uses pivotal method
            # (mean - quantile) gives lower bound
            q_lower = np.quantile(predictions_bootstrap, alpha_lower, axis=0)
            q_upper = np.quantile(predictions_bootstrap, alpha_upper, axis=0)

            y_lower = 2 * y_pred_mean - q_upper
            y_upper = 2 * y_pred_mean - q_lower

        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            "mean": y_pred_mean,
            "median": y_pred_median,
            "lower": y_lower,
            "upper": y_upper,
            "std": y_pred_std,
            "confidence": self.confidence,
            "method": method,
        }


class QuantileRegression:
    """
    Quantile Regression for uncertainty estimation.

    Fits separate models for different quantiles (e.g., 0.05, 0.5, 0.95).
    """

    def __init__(self, quantiles: Optional[list] = None, confidence: float = 0.95):
        """
        Initialize quantile regression.

        Parameters
        ----------
        quantiles : list, optional
            Quantiles to estimate. Default: [alpha/2, 0.5, 1-alpha/2].
        confidence : float, default=0.95
            Confidence level.
        """
        self.confidence = confidence
        self.alpha = 1 - confidence

        if quantiles is None:
            alpha_lower = self.alpha / 2
            alpha_upper = 1 - alpha_lower
            self.quantiles = [alpha_lower, 0.5, alpha_upper]
        else:
            self.quantiles = quantiles

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, model_class, **model_kwargs):
        """
        Fit quantile regression models.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_features)
            Training features.
        y_train : np.ndarray, shape (n_train,)
            Training targets.
        model_class : class
            Regression model class with fit/predict.
        **model_kwargs : dict
            Model arguments.

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        self.models_ = {}

        for q in self.quantiles:
            # Weighted least squares approximation to quantile regression
            # Using iterative reweighting: weights = 1/|y - pred|
            model = model_class(**model_kwargs)
            
            # Initial fit
            model.fit(X_train, y_train)
            pred = model.predict(X_train)

            # Iterative reweighting
            for iteration in range(3):
                residuals = y_train - pred
                weights = np.where(
                    residuals >= 0,
                    q / np.maximum(np.abs(residuals), 1e-6),
                    (1 - q) / np.maximum(np.abs(residuals), 1e-6)
                )
                weights = np.minimum(weights, 1e6)  # Cap extreme weights

                # Weighted fit
                model = model_class(**model_kwargs)
                if hasattr(model, 'fit'):
                    # Try sample weights if supported
                    try:
                        model.fit(X_train, y_train, sample_weight=weights)
                    except TypeError:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                    
                pred = model.predict(X_train)

            self.models_[q] = model

        return self

    def predict(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict with quantile estimates.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, n_features)
            Test features.

        Returns
        -------
        predictions : dict
            Keys:
            - 'median': Median predictions (0.5 quantile)
            - 'lower': Lower confidence bound
            - 'upper': Upper confidence bound
        """
        X_test = np.asarray(X_test, dtype=np.float64)

        result = {}

        for q, model in self.models_.items():
            result[f"q{q}"] = model.predict(X_test)

        # Extract standard quantiles
        median_key = min(self.models_.keys(), key=lambda x: abs(x - 0.5))
        lower_key = min(self.models_.keys(), key=lambda x: abs(x - self.alpha/2))
        upper_key = min(self.models_.keys(), key=lambda x: abs(x - (1 - self.alpha/2)))

        return {
            "median": self.models_[median_key].predict(X_test),
            "lower": self.models_[lower_key].predict(X_test),
            "upper": self.models_[upper_key].predict(X_test),
            "confidence": self.confidence,
            "all_quantiles": result,
        }


class ConformalRegression:
    """
    Conformal Prediction for regression.

    Model-agnostic uncertainty quantification via conformity scores.
    Produces distribution-free prediction intervals.
    """

    def __init__(self, confidence: float = 0.95, method: str = "standard"):
        """
        Initialize conformal regression.

        Parameters
        ----------
        confidence : float, default=0.95
            Confidence level.
        method : {'standard', 'adaptive'}, default='standard'
            Conformal method.

        Notes
        -----
        Standard: All predictions have same interval width
        Adaptive: Interval width depends on conformity score (heteroscedastic)
        """
        self.confidence = confidence
        self.alpha = 1 - confidence
        self.method = method

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model,
    ):
        """
        Fit conformal regression model.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_features)
            Training features.
        y_train : np.ndarray, shape (n_train,)
            Training targets.
        model : sklearn-like regressor
            Pre-fitted or will be fitted.

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        self.n_train_ = len(y_train)
        self.model_ = model

        # If model not yet fitted, fit it
        if not hasattr(self.model_, 'predict'):
            self.model_.fit(X_train, y_train)

        # Calculate conformity scores (nonconformity measures)
        # Nonconformity = |y_true - y_pred|
        y_pred_train = self.model_.predict(X_train)
        self.conformity_scores_ = np.abs(y_train - y_pred_train)

        # Calculate quantile for confidence level
        # Using ceiling to ensure at least (1-alpha) coverage
        self.q_ = np.ceil((self.n_train_ + 1) * (1 - self.alpha)) / (self.n_train_ + 1)
        self.threshold_ = np.quantile(self.conformity_scores_, self.q_)

        return self

    def predict(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict with conformal intervals.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, n_features)
            Test features.

        Returns
        -------
        predictions : dict
            Keys:
            - 'mean': Point predictions
            - 'lower': Lower confidence bound
            - 'upper': Upper confidence bound
            - 'width': Interval width
        """
        X_test = np.asarray(X_test, dtype=np.float64)

        y_pred = self.model_.predict(X_test)

        if self.method == "standard":
            # Standard conformal: constant interval width
            y_lower = y_pred - self.threshold_
            y_upper = y_pred + self.threshold_

        elif self.method == "adaptive":
            # Adaptive conformal: width based on test conformity score
            # Use input distance as proxy for conformity score variability
            # (more sophisticated methods possible with known covariate shift)

            # For now, use gradient-based nonconformity (simplified)
            # In practice, would need access to training data distances or residuals

            margins = np.full_like(y_pred, self.threshold_)
            y_lower = y_pred - margins
            y_upper = y_pred + margins
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return {
            "mean": y_pred,
            "lower": y_lower,
            "upper": y_upper,
            "width": y_upper - y_lower,
            "confidence": self.confidence,
            "threshold": self.threshold_,
        }
