"""Partial Least Squares Regression (PLSR) with VIP scores."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ["PLSRegression", "VIPCalculator"]


class PLSRegression(BaseEstimator, RegressorMixin):
    """
    Partial Least Squares Regression (PLSR).

    Uses nipals algorithm for component extraction.
    """

    def __init__(
        self,
        n_components: int = 3,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
    ):
        """
        Initialize PLSR.

        Parameters
        ----------
        n_components : int, default=3
            Number of latent components.
        scale : bool, default=True
            If True, center and scale X and y.
        max_iter : int, default=500
            Maximum NIPALS iterations.
        tol : float, default=1e-6
            Convergence tolerance.
        """
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> PLSRegression:
        """
        Fit PLSR model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target variable(s).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        # Standardize
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0, ddof=1)
        if self.scale:
            X = (X - self.x_mean_) / (self.x_std_ + 1e-8)

        self.y_mean_ = y.mean(axis=0)
        self.y_std_ = y.std(axis=0, ddof=1)
        if self.scale:
            y = (y - self.y_mean_) / (self.y_std_ + 1e-8)

        # NIPALS
        self.x_loadings_ = np.zeros((n_features, self.n_components))
        self.y_loadings_ = np.zeros((n_targets, self.n_components))
        self.x_scores_ = np.zeros((n_samples, self.n_components))
        self.y_scores_ = np.zeros((n_samples, self.n_components))
        self.weights_ = np.zeros((n_features, self.n_components))

        X_fit = X.copy()
        y_fit = y.copy()

        for h in range(self.n_components):
            # Initialize w with column of X'y with largest absolute value
            w = X_fit.T @ y_fit[:, 0]
            w /= np.linalg.norm(w) + 1e-8

            for _ in range(self.max_iter):
                # Score vector
                t = X_fit @ w
                t /= np.linalg.norm(t) + 1e-8

                # Loading vector
                c = y_fit.T @ t
                c /= np.linalg.norm(c) + 1e-8

                # Weight update
                w_new = X_fit.T @ (y_fit @ c)
                w_new /= np.linalg.norm(w_new) + 1e-8

                if np.linalg.norm(w_new - w) < self.tol:
                    break
                w = w_new

            # Score and loading
            t = X_fit @ w
            p = (X_fit.T @ t) / (t.T @ t + 1e-8)
            c = (y_fit.T @ t) / (t.T @ t + 1e-8)

            # Store
            self.weights_[:, h] = w
            self.x_scores_[:, h] = t
            self.y_scores_[:, h] = (y_fit @ c).ravel()
            self.x_loadings_[:, h] = p
            self.y_loadings_[:, h] = c

            # Deflate
            X_fit = X_fit - np.outer(t, p)
            y_fit = y_fit - np.outer(t, c)

        # Regression coefficients
        self.coef_ = (
            self.weights_ @ np.linalg.lstsq(self.weights_.T @ self.x_loadings_, self.y_loadings_.T, rcond=None)[0]
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : np.ndarray
            Predicted values.
        """
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.x_mean_) / (self.x_std_ + 1e-8) if self.scale else X

        y_pred = X_scaled @ self.coef_

        if self.scale:
            y_pred = y_pred * self.y_std_ + self.y_mean_

        return y_pred.ravel() if y_pred.shape[1] == 1 else y_pred

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform to PLS scores.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        T : np.ndarray
            PLS scores.
        """
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.x_mean_) / (self.x_std_ + 1e-8) if self.scale else X
        return X_scaled @ self.weights_


class VIPCalculator:
    """Calculate Variable Importance in Projection (VIP) scores."""

    @staticmethod
    def calculate_vip(
        X: np.ndarray,
        y: np.ndarray,
        n_components: Optional[int] = None,
    ) -> np.ndarray:
        """
        Calculate VIP scores from PLSR.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target variable.
        n_components : int, optional
            Number of PLS components. If None, uses X.shape[1]//2.

        Returns
        -------
        vip : np.ndarray, shape (n_features,)
            VIP scores.

        Notes
        -----
        VIP = sqrt(p * sum((w_h * t_h'y)^2) / (y'y))
        where p = n_features, w_h = X weights, t_h = X scores.

        References
        ----------
        Wold, S., Sjöström, M., & Eriksson, L. (2001).
        PLS-regression: a basic tool of chemometrics.
        Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130.
        """
        if n_components is None:
            n_components = X.shape[1] // 2

        pls = PLSRegression(n_components=n_components, scale=True)
        pls.fit(X, y)

        n_features = X.shape[1]
        ss_y = np.sum((pls.y_scores_) ** 2, axis=0)
        ss_y_total = np.sum(ss_y)

        vip = np.zeros(n_features)
        for i in range(n_features):
            weight_contrib = np.sum(((pls.weights_[i, :] ** 2) * ss_y) / (ss_y_total + 1e-8))
            vip[i] = np.sqrt(n_features * weight_contrib)

        return vip

    @staticmethod
    def plot_vip(
        vip: np.ndarray,
        feature_names: Optional[list] = None,
        threshold: float = 1.0,
        ax=None,
    ):
        """
        Plot VIP scores.

        Parameters
        ----------
        vip : np.ndarray
            VIP scores.
        feature_names : list, optional
            Feature names.
        threshold : float, default=1.0
            VIP threshold (typically 1.0).
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
            ax = plt.gca()

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(len(vip))]

        sorted_idx = np.argsort(vip)[::-1]
        sorted_vip = vip[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]

        colors = ["red" if v > threshold else "blue" for v in sorted_vip]
        ax.barh(sorted_names, sorted_vip, color=colors, alpha=0.7)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
        ax.set_xlabel("VIP Score")
        ax.set_title("Variable Importance in Projection (VIP)")
        ax.legend()

        return ax
