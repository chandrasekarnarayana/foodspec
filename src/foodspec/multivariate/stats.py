"""Statistical multivariate tests: Hotelling's T-squared and MANOVA."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import scipy.stats as stats
from numpy.linalg import inv

from foodspec.multivariate.base import MultivariateComponent, MultivariateResult


class HotellingT2Component(MultivariateComponent):
    method = "hotelling_t2"
    requires_y = True

    def __init__(self, alpha: float = 0.05, **kwargs: Any):
        super().__init__(alpha=alpha, **kwargs)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "HotellingT2Component":
        if y is None:
            raise ValueError("Hotelling T2 requires two groups in y.")
        unique = np.unique(y)
        if unique.size != 2:
            raise ValueError("Hotelling T2 requires exactly two groups.")
        g1, g2 = unique
        x1, x2 = X[y == g1], X[y == g2]
        n1, n2 = x1.shape[0], x2.shape[0]
        mean1, mean2 = np.mean(x1, axis=0), np.mean(x2, axis=0)
        cov1, cov2 = np.cov(x1, rowvar=False), np.cov(x2, rowvar=False)
        sp = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
        diff = mean1 - mean2
        t2 = (n1 * n2) / (n1 + n2) * diff.T @ inv(sp) @ diff
        p = X.shape[1]
        f_stat = (n1 + n2 - p - 1) * t2 / ((n1 + n2 - 2) * p)
        p_value = 1 - stats.f.cdf(f_stat, p, n1 + n2 - p - 1)
        self.model = {"t2": t2, "f_stat": f_stat, "p_value": p_value}
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        if self.model is None:
            raise RuntimeError("HotellingT2Component not fitted.")
        return np.array([self.model["t2"]])

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultivariateResult:
        self.fit(X, y)
        return MultivariateResult(
            method=self.method,
            scores=np.array([[self.model["t2"]]]),
            params=self.params,
            model=self.model,
            metadata={"f_stat": self.model["f_stat"], "p_value": self.model["p_value"]},
        )


class MANOVAComponent(MultivariateComponent):
    method = "manova"
    requires_y = True

    def __init__(self, alpha: float = 0.05, **kwargs: Any):
        super().__init__(alpha=alpha, **kwargs)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MANOVAComponent":
        if y is None:
            raise ValueError("MANOVA requires group labels y.")
        unique = np.unique(y)
        if unique.size < 2:
            raise ValueError("MANOVA requires at least two groups.")
        overall_mean = np.mean(X, axis=0)
        p = X.shape[1]
        g = unique.size
        b_matrix = np.zeros((p, p))
        w_matrix = np.zeros((p, p))
        for grp in unique:
            xg = X[y == grp]
            n_g = xg.shape[0]
            mean_g = np.mean(xg, axis=0)
            mean_diff = (mean_g - overall_mean).reshape(-1, 1)
            b_matrix += n_g * mean_diff @ mean_diff.T
            x_centered = xg - mean_g
            w_matrix += x_centered.T @ x_centered
        wilks_lambda = np.linalg.det(w_matrix) / np.linalg.det(w_matrix + b_matrix)
        df1 = p * (g - 1)
        df2 = np.sum([p + g - 1 for _ in unique])  # approximate, simple form
        chi_stat = -((X.shape[0] - 1) - (p + g) / 2) * np.log(wilks_lambda)
        p_value = 1 - stats.chi2.cdf(chi_stat, df1)
        self.model = {
            "wilks_lambda": wilks_lambda,
            "chi_stat": chi_stat,
            "p_value": p_value,
            "df1": df1,
            "df2": df2,
        }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        if self.model is None:
            raise RuntimeError("MANOVAComponent not fitted.")
        return np.array([self.model["wilks_lambda"]])

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultivariateResult:
        self.fit(X, y)
        return MultivariateResult(
            method=self.method,
            scores=np.array([[self.model["wilks_lambda"]]]),
            params=self.params,
            model=self.model,
            metadata={
                "wilks_lambda": self.model["wilks_lambda"],
                "chi_stat": self.model["chi_stat"],
                "p_value": self.model["p_value"],
                "df1": self.model["df1"],
                "df2": self.model["df2"],
            },
        )


__all__ = ["HotellingT2Component", "MANOVAComponent"]
