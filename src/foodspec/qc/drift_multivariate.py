"""Multivariate Drift Detection: MMD, Wasserstein, and Advanced Methods."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

__all__ = ["MMDDriftDetector", "WassersteinDriftDetector", "MultivariateDriftMonitor"]


class MMDDriftDetector:
    """
    Maximum Mean Discrepancy (MMD) for multivariate drift detection.

    Non-parametric test for whether two distributions differ.
    More sensitive than univariate tests for high-dimensional data.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        kernel_width: Optional[float] = None,
        alpha: float = 0.05,
    ):
        """
        Initialize MMD drift detector.

        Parameters
        ----------
        kernel : {'rbf', 'linear'}, default='rbf'
            Kernel function to use.
        kernel_width : float, optional
            Kernel width/bandwidth. If None, uses median heuristic.
        alpha : float, default=0.05
            Significance level for drift detection.

        Notes
        -----
        MMD^2 = mean distance between kernel feature means of two distributions.

        References
        ----------
        Gretton et al. (2012). A kernel two-sample test.
        Journal of Machine Learning Research, 13, 723-773.
        """
        self.kernel = kernel
        self.kernel_width = kernel_width
        self.alpha = alpha
        self.reference_data_ = None
        self.mmd_threshold_ = None

    def initialize(self, X_ref: np.ndarray):
        """
        Initialize detector with reference distribution.

        Parameters
        ----------
        X_ref : np.ndarray, shape (n_ref, n_features)
            Reference data.

        Returns
        -------
        self
        """
        X_ref = np.asarray(X_ref, dtype=np.float64)

        if X_ref.ndim != 2:
            raise ValueError("X_ref must be 2-dimensional")

        self.reference_data_ = X_ref
        self.n_ref_ = X_ref.shape[0]
        self.n_features_ = X_ref.shape[1]

        # Estimate kernel width using median heuristic
        if self.kernel_width is None:
            if self.kernel == "rbf":
                # Median pairwise distance
                pairwise_dists = cdist(X_ref, X_ref)
                self.kernel_width = np.median(pairwise_dists[pairwise_dists > 0])
                if self.kernel_width == 0:
                    self.kernel_width = 1.0
            else:
                self.kernel_width = 1.0

        # Estimate threshold (not robust; use permutation test in practice)
        # Simple approximation: threshold from known asymptotic distribution
        self.mmd_threshold_ = np.sqrt(2 / min(self.n_ref_, self.n_ref_)) * stats.norm.ppf(1 - self.alpha)

        return self

    def _compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X and Y."""
        if self.kernel == "rbf":
            # RBF kernel: exp(-||x-y||^2 / (2*width^2))
            sq_dists = cdist(X, Y, metric="sqeuclidean")
            K = np.exp(-sq_dists / (2 * self.kernel_width ** 2))
        elif self.kernel == "linear":
            K = np.dot(X, Y.T)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        return K

    def compute_mmd(self, X_test: np.ndarray) -> float:
        """
        Compute MMD^2 between reference and test distributions.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, n_features)
            Test data.

        Returns
        -------
        mmd2 : float
            MMD^2 statistic (squared Maximum Mean Discrepancy).
        """
        X_test = np.asarray(X_test, dtype=np.float64)

        if X_test.ndim != 2:
            X_test = X_test.reshape(-1, self.n_features_)

        if X_test.shape[1] != self.n_features_:
            raise ValueError(f"X_test has {X_test.shape[1]} features, expected {self.n_features_}")

        # Kernel matrix: reference vs reference
        K_ref_ref = self._compute_kernel_matrix(self.reference_data_, self.reference_data_)

        # Kernel matrix: test vs test
        K_test_test = self._compute_kernel_matrix(X_test, X_test)

        # Kernel matrix: reference vs test
        K_ref_test = self._compute_kernel_matrix(self.reference_data_, X_test)

        # MMD^2 = mean K(X,X) + mean K(Y,Y) - 2*mean K(X,Y)
        mmd2 = (
            np.mean(K_ref_ref)
            + np.mean(K_test_test)
            - 2 * np.mean(K_ref_test)
        )

        return float(mmd2)

    def detect(self, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in test data.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, n_features)
            Test data to check for drift.

        Returns
        -------
        result : dict
            Drift detection results.
        """
        mmd2 = self.compute_mmd(X_test)
        is_drift = mmd2 > self.mmd_threshold_

        return {
            "mmd2": float(mmd2),
            "threshold": float(self.mmd_threshold_),
            "is_drift": bool(is_drift),
            "p_value_approx": float(1 - stats.norm.cdf(np.sqrt(mmd2 / self.mmd_threshold_))),
            "n_test": X_test.shape[0],
            "kernel": self.kernel,
            "kernel_width": float(self.kernel_width),
        }


class WassersteinDriftDetector:
    """
    Wasserstein distance for drift detection.

    Uses optimal transport distance between distributions.
    More computationally expensive but better theoretical properties.
    """

    def __init__(self, approximation: str = "sliced", n_projections: int = 50, alpha: float = 0.05):
        """
        Initialize Wasserstein drift detector.

        Parameters
        ----------
        approximation : {'sliced', 'euclidean'}, default='sliced'
            Wasserstein approximation method.
            - sliced: Fast sliced Wasserstein (1D projections)
            - euclidean: Exact in 2D, approximation in higher dimensions
        n_projections : int, default=50
            Number of random projections for sliced Wasserstein.
        alpha : float, default=0.05
            Significance level.

        Notes
        -----
        Sliced Wasserstein is much faster: O(n log n) vs O(n^3) for exact.
        """
        self.approximation = approximation
        self.n_projections = n_projections
        self.alpha = alpha
        self.reference_data_ = None

    def initialize(self, X_ref: np.ndarray):
        """Initialize with reference distribution."""
        X_ref = np.asarray(X_ref, dtype=np.float64)

        if X_ref.ndim != 2:
            raise ValueError("X_ref must be 2-dimensional")

        self.reference_data_ = X_ref
        self.n_ref_ = X_ref.shape[0]
        self.n_features_ = X_ref.shape[1]

        return self

    def _sliced_wasserstein(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute sliced Wasserstein distance."""
        n_features = X.shape[1]

        distances = []
        for _ in range(self.n_projections):
            # Random unit direction
            theta = np.random.randn(n_features)
            theta /= np.linalg.norm(theta)

            # Project data onto this direction
            X_proj = np.dot(X, theta)
            Y_proj = np.dot(Y, theta)

            # Sort projections
            X_proj_sorted = np.sort(X_proj)
            Y_proj_sorted = np.sort(Y_proj)

            # Handle different sizes via interpolation
            if len(X_proj_sorted) < len(Y_proj_sorted):
                X_proj_sorted = np.interp(
                    np.linspace(0, 1, len(Y_proj_sorted)),
                    np.linspace(0, 1, len(X_proj_sorted)),
                    X_proj_sorted
                )
            elif len(Y_proj_sorted) < len(X_proj_sorted):
                Y_proj_sorted = np.interp(
                    np.linspace(0, 1, len(X_proj_sorted)),
                    np.linspace(0, 1, len(Y_proj_sorted)),
                    Y_proj_sorted
                )

            # 1D Wasserstein distance
            distances.append(np.mean(np.abs(X_proj_sorted - Y_proj_sorted)))

        return np.mean(distances)

    def compute_distance(self, X_test: np.ndarray) -> float:
        """Compute Wasserstein distance to reference."""
        X_test = np.asarray(X_test, dtype=np.float64)

        if X_test.ndim != 2:
            X_test = X_test.reshape(-1, self.n_features_)

        if self.approximation == "sliced":
            distance = self._sliced_wasserstein(self.reference_data_, X_test)
        else:
            # Euclidean: simple L2 distance between means
            distance = np.linalg.norm(np.mean(self.reference_data_, axis=0) - np.mean(X_test, axis=0))

        return float(distance)

    def detect(self, X_test: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect drift using Wasserstein distance.

        Parameters
        ----------
        X_test : np.ndarray
            Test data.
        threshold : float, optional
            Distance threshold for drift. If None, uses reference percentile.

        Returns
        -------
        result : dict
            Drift detection results.
        """
        distance = self.compute_distance(X_test)

        if threshold is None:
            # Estimate threshold as 95th percentile of reference self-distances
            np.random.seed(42)
            n_splits = min(10, len(self.reference_data_) // 2)
            ref_distances = []

            for _ in range(n_splits):
                idx1 = np.random.choice(len(self.reference_data_), len(self.reference_data_) // 2, replace=False)
                idx2 = np.setdiff1d(np.arange(len(self.reference_data_)), idx1)

                d = self._sliced_wasserstein(
                    self.reference_data_[idx1],
                    self.reference_data_[idx2]
                ) if self.approximation == "sliced" else np.linalg.norm(
                    np.mean(self.reference_data_[idx1], axis=0) - np.mean(self.reference_data_[idx2], axis=0)
                )
                ref_distances.append(d)

            threshold = np.percentile(ref_distances, 95)

        is_drift = distance > threshold

        return {
            "wasserstein_distance": float(distance),
            "threshold": float(threshold),
            "is_drift": bool(is_drift),
            "n_test": X_test.shape[0],
            "approximation": self.approximation,
        }


class MultivariateDriftMonitor:
    """Real-time multivariate drift monitoring combining multiple methods."""

    def __init__(self, alpha: float = 0.05):
        """Initialize drift monitor."""
        self.alpha = alpha
        self.mmd_detector = None
        self.wasserstein_detector = None
        self.history_ = []

    def initialize(self, X_ref: np.ndarray):
        """Initialize with reference data."""
        self.mmd_detector = MMDDriftDetector(alpha=self.alpha)
        self.mmd_detector.initialize(X_ref)

        self.wasserstein_detector = WassersteinDriftDetector(alpha=self.alpha)
        self.wasserstein_detector.initialize(X_ref)

        return self

    def detect(self, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift using multiple methods.

        Parameters
        ----------
        X_test : np.ndarray
            Test data.

        Returns
        -------
        result : dict
            Unified drift detection result.
        """
        mmd_result = self.mmd_detector.detect(X_test)
        wasserstein_result = self.wasserstein_detector.detect(X_test)

        # Combine results
        n_methods = 2
        n_drift_votes = int(mmd_result["is_drift"]) + int(wasserstein_result["is_drift"])
        consensus_drift = n_drift_votes >= (n_methods / 2)

        result = {
            "is_drift": bool(consensus_drift),
            "drift_votes": n_drift_votes,
            "n_methods": n_methods,
            "mmd": mmd_result,
            "wasserstein": wasserstein_result,
            "timestamp": len(self.history_),
        }

        self.history_.append(result)

        return result

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift monitoring history."""
        if not self.history_:
            return {"n_samples_monitored": 0}

        drifts = [h["is_drift"] for h in self.history_]
        mmd_scores = [h["mmd"]["mmd2"] for h in self.history_]

        return {
            "n_samples_monitored": len(self.history_),
            "n_drift_alerts": int(np.sum(drifts)),
            "drift_rate": float(np.mean(drifts)),
            "avg_mmd2": float(np.mean(mmd_scores)),
            "max_mmd2": float(np.max(mmd_scores)),
        }
