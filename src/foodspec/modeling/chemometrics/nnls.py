"""Non-Negative Least Squares (NNLS) regression for constrained prediction."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ["NNLSRegression", "ConstrainedLasso"]


class NNLSRegression(BaseEstimator, RegressorMixin):
    """
    Non-Negative Least Squares Regression.

    Solves: minimize ||Xb - y||_2 subject to b >= 0.
    """

    def __init__(self, scale: bool = True):
        """
        Initialize NNLS regression.

        Parameters
        ----------
        scale : bool, default=True
            If True, standardize features.
        """
        self.scale = scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> NNLSRegression:
        """
        Fit NNLS regression.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target variable.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0, ddof=1)

        if self.scale:
            X = (X - self.x_mean_) / (self.x_std_ + 1e-8)

        self.y_mean_ = y.mean()
        self.y_std_ = y.std(ddof=1)

        if self.scale:
            y = (y - self.y_mean_) / (self.y_std_ + 1e-8)

        # Solve NNLS for each feature
        self.coef_ = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            self.coef_[j], _ = nnls(X[:, [j]], y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using NNLS.

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

        return y_pred

    def get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute residuals.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True target values.

        Returns
        -------
        residuals : np.ndarray
            Residuals (y - y_pred).
        """
        return y - self.predict(X)


class ConstrainedLasso(BaseEstimator, RegressorMixin):
    """
    Non-negative LASSO with grouped constraints.

    Solves: minimize 0.5 * ||Xb - y||_2^2 + alpha * ||b||_1
    subject to b >= 0 and sum(b in groups) = 1 (optional).
    """

    def __init__(
        self,
        alpha: float = 0.01,
        scale: bool = True,
        sum_to_one: bool = False,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        """
        Initialize Constrained LASSO.

        Parameters
        ----------
        alpha : float, default=0.01
            Regularization strength.
        scale : bool, default=True
            If True, standardize features.
        sum_to_one : bool, default=False
            If True, coefficients sum to 1 (for composition data).
        max_iter : int, default=1000
            Maximum iterations.
        tol : float, default=1e-4
            Convergence tolerance.
        """
        self.alpha = alpha
        self.scale = scale
        self.sum_to_one = sum_to_one
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> ConstrainedLasso:
        """
        Fit Constrained LASSO using coordinate descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target variable.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0, ddof=1)

        if self.scale:
            X = (X - self.x_mean_) / (self.x_std_ + 1e-8)

        self.y_mean_ = y.mean()
        self.y_std_ = y.std(ddof=1)

        if self.scale:
            y = (y - self.y_mean_) / (self.y_std_ + 1e-8)

        n_samples, n_features = X.shape

        # Coordinate descent
        coef = np.zeros(n_features)

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            for j in range(n_features):
                # Compute gradient
                residual = y - X @ coef + X[:, j] * coef[j]
                grad = -X[:, j] @ residual / n_samples

                # Soft-thresholding with non-negativity
                coef[j] = max(0, grad - self.alpha) if grad > self.alpha else 0

            # Project to sum-to-one if needed
            if self.sum_to_one:
                coef = np.maximum(coef, 0)
                coef_sum = coef.sum()
                if coef_sum > 1e-8:
                    coef /= coef_sum

            # Check convergence
            if np.linalg.norm(coef - coef_old) < self.tol:
                break

        self.coef_ = coef

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using Constrained LASSO.

        Parameters
        ----------
        X : np.ndarray
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

        return y_pred

    def get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute residuals.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True target values.

        Returns
        -------
        residuals : np.ndarray
            Residuals (y - y_pred).
        """
        return y - self.predict(X)

    def sparsity(self) -> float:
        """
        Calculate sparsity (fraction of zero coefficients).

        Returns
        -------
        sparsity : float
            Fraction of zero coefficients (0 to 1).
        """
        return np.sum(self.coef_ == 0) / len(self.coef_)
