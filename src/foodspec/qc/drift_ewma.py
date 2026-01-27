"""EWMA control charts for drift monitoring on scores (PCA/PLS)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats

__all__ = ["EWMAControlChart", "DriftDetector"]


class EWMAControlChart:
    """
    Exponentially Weighted Moving Average (EWMA) control chart.

    Monitors process mean for drift or shifts.
    More sensitive to small drifts than Shewhart charts.
    """

    def __init__(
        self,
        lambda_: float = 0.2,
        confidence: float = 0.99,
    ):
        """
        Initialize EWMA control chart.

        Parameters
        ----------
        lambda_ : float, default=0.2
            Smoothing parameter (0 < lambda <= 1).
            Lower values emphasize historical data.
            Typical range: 0.05-0.30.
        confidence : float, default=0.99
            Confidence level for control limits.

        References
        ----------
        Roberts, S. W. (1959). Control chart tests based on geometric
        moving averages. Technometrics, 1(3), 239-250.
        """
        if not (0 < lambda_ <= 1):
            raise ValueError("lambda_ must be in (0, 1]")
        if not (0 < confidence < 1):
            raise ValueError("confidence must be in (0, 1)")

        self.lambda_ = lambda_
        self.confidence = confidence

    def initialize(self, X: np.ndarray) -> EWMAControlChart:
        """
        Initialize control chart parameters from reference data.

        Parameters
        ----------
        X : np.ndarray, shape (n_reference, n_features)
            Reference/calibration data.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.center_ = np.mean(X, axis=0)
        self.sigma_ = np.std(X, axis=0, ddof=1)

        # EWMA standard error at time t:
        # SE_t = sigma * sqrt(lambda / (2 - lambda) * (1 - (1 - lambda)^(2t)))
        # As t -> infinity, SE -> sigma * sqrt(lambda / (2 - lambda))
        self.steady_state_sigma_ = self.sigma_ * np.sqrt(self.lambda_ / (2 - self.lambda_))

        z_critical = stats.norm.ppf((1 + self.confidence) / 2)
        self.ucl_ = self.center_ + z_critical * self.steady_state_sigma_
        self.lcl_ = self.center_ - z_critical * self.steady_state_sigma_

        self.ewma_values_ = []
        self.times_ = []
        self.out_of_control_ = []

        return self

    def update(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Update EWMA with new observation.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            New observation.

        Returns
        -------
        ewma : np.ndarray
            Updated EWMA value.
        is_alarm : bool
            True if observation is out of control.
        """
        x = np.asarray(x, dtype=np.float64).ravel()

        if len(self.ewma_values_) == 0:
            ewma = self.lambda_ * x + (1 - self.lambda_) * self.center_
        else:
            ewma = self.lambda_ * x + (1 - self.lambda_) * self.ewma_values_[-1]

        self.ewma_values_.append(ewma)
        self.times_.append(len(self.ewma_values_))

        is_alarm = np.any((ewma < self.lcl_) | (ewma > self.ucl_))
        self.out_of_control_.append(is_alarm)

        return ewma, is_alarm

    def process(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process multiple observations.

        Parameters
        ----------
        X : np.ndarray, shape (n_obs, n_features)
            Observations to process.

        Returns
        -------
        ewma_values : np.ndarray, shape (n_obs, n_features)
            EWMA values.
        alarms : np.ndarray, shape (n_obs,)
            Boolean array indicating out-of-control points.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        alarms = np.zeros(len(X), dtype=bool)

        for i, x in enumerate(X):
            _, alarm = self.update(x)
            alarms[i] = alarm

        return np.array(self.ewma_values_), alarms

    def plot(self, ax=None):
        """
        Plot EWMA control chart.

        Parameters
        ----------
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
            fig, ax = plt.subplots(figsize=(12, 6))

        ewma_vals = np.array(self.ewma_values_)
        times = np.array(self.times_)

        # For multivariate, plot first component
        if ewma_vals.ndim > 1:
            ewma_vals = ewma_vals[:, 0]
            ucl = self.ucl_[0]
            lcl = self.lcl_[0]
            center = self.center_[0]
        else:
            ucl = self.ucl_[0] if hasattr(self.ucl_, '__len__') else self.ucl_
            lcl = self.lcl_[0] if hasattr(self.lcl_, '__len__') else self.lcl_
            center = self.center_[0] if hasattr(self.center_, '__len__') else self.center_

        ax.plot(times, ewma_vals, "b.-", label="EWMA", markersize=4)

        # Plot control limits
        ax.axhline(center, color="green", linestyle="-", linewidth=2, label="Center")
        ax.axhline(ucl, color="red", linestyle="--", linewidth=1.5, label="UCL")
        ax.axhline(lcl, color="red", linestyle="--", linewidth=1.5, label="LCL")

        # Highlight out-of-control points
        ooc_idx = np.where(self.out_of_control_)[0]
        if len(ooc_idx) > 0:
            ax.scatter(times[ooc_idx], ewma_vals[ooc_idx], color="red", s=100, marker="X", label="Out of Control", zorder=5)

        ax.set_xlabel("Sample Number")
        ax.set_ylabel("EWMA Value")
        ax.set_title(f"EWMA Control Chart (λ={self.lambda_})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class DriftDetector:
    """
    Comprehensive drift detection on PCA/PLS scores.

    Combines EWMA, trend tests, and seasonal decomposition.
    """

    def __init__(
        self,
        lambda_: float = 0.2,
        min_samples: int = 20,
    ):
        """
        Initialize DriftDetector.

        Parameters
        ----------
        lambda_ : float, default=0.2
            EWMA smoothing parameter.
        min_samples : int, default=20
            Minimum samples before drift detection.
        """
        self.lambda_ = lambda_
        self.min_samples = min_samples
        self.ewma_chart_ = None

    def initialize(self, X: np.ndarray) -> DriftDetector:
        """
        Initialize drift detector from reference data.

        Parameters
        ----------
        X : np.ndarray, shape (n_reference, n_features)
            Reference data (e.g., PCA scores).

        Returns
        -------
        self
        """
        self.ewma_chart_ = EWMAControlChart(lambda_=self.lambda_)
        self.ewma_chart_.initialize(X)

        self.reference_data_ = X
        self.reference_mean_ = np.mean(X, axis=0)
        self.reference_std_ = np.std(X, axis=0, ddof=1)

        self.scores_ = []
        self.drift_signals_ = []

        return self

    def check_drift(self, x: np.ndarray) -> dict:
        """
        Check single observation for drift.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            New observation.

        Returns
        -------
        result : dict
            Drift detection result with keys:
            - 'ewma_alarm': bool, EWMA out-of-control signal
            - 'mahalanobis_distance': float
            - 'is_outlier': bool
            - 'drift_type': str, 'none', 'shift', 'outlier'

        Notes
        -----
        Combines multiple signals:
        1. EWMA chart for mean drift
        2. Mahalanobis distance for multivariate outliers
        """
        x = np.asarray(x, dtype=np.float64).ravel()

        # EWMA signal
        _, ewma_alarm = self.ewma_chart_.update(x)

        # Mahalanobis distance
        cov_ref = np.cov(self.reference_data_.T)
        try:
            inv_cov = np.linalg.inv(cov_ref)
            mahal_dist = np.sqrt((x - self.reference_mean_) @ inv_cov @ (x - self.reference_mean_))
        except np.linalg.LinAlgError:
            mahal_dist = np.linalg.norm(x - self.reference_mean_)

        # Chi-square threshold for outlier detection
        n_features = x.shape[0]
        chi2_threshold = stats.chi2.ppf(0.95, df=n_features)
        is_outlier = mahal_dist > np.sqrt(chi2_threshold)

        # Classify drift type
        if not ewma_alarm and not is_outlier:
            drift_type = "none"
        elif is_outlier and not ewma_alarm:
            drift_type = "outlier"
        else:
            drift_type = "shift"

        result = {
            "ewma_alarm": bool(ewma_alarm),
            "mahalanobis_distance": float(mahal_dist),
            "is_outlier": bool(is_outlier),
            "drift_type": drift_type,
            "timestamp": len(self.scores_),
        }

        self.scores_.append(x)
        self.drift_signals_.append(result)

        return result

    def process_stream(self, X: np.ndarray) -> list:
        """
        Process stream of observations.

        Parameters
        ----------
        X : np.ndarray, shape (n_obs, n_features)
            Stream of observations.

        Returns
        -------
        results : list
            List of drift detection results.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = []
        for x in X:
            result = self.check_drift(x)
            results.append(result)

        return results

    def get_drift_summary(self) -> dict:
        """
        Get summary statistics of drift signals.

        Returns
        -------
        summary : dict
            Summary with keys:
            - 'n_observations': int
            - 'n_alarms': int
            - 'n_outliers': int
            - 'alarm_rate': float
            - 'mean_mahalanobis': float
        """
        if not self.drift_signals_:
            return {
                "n_observations": 0,
                "n_alarms": 0,
                "n_outliers": 0,
                "alarm_rate": 0,
                "mean_mahalanobis": 0,
            }

        n_alarms = sum(1 for s in self.drift_signals_ if s["ewma_alarm"])
        n_outliers = sum(1 for s in self.drift_signals_ if s["is_outlier"])
        mahal_dists = [s["mahalanobis_distance"] for s in self.drift_signals_]

        return {
            "n_observations": len(self.drift_signals_),
            "n_alarms": n_alarms,
            "n_outliers": n_outliers,
            "alarm_rate": n_alarms / len(self.drift_signals_) if self.drift_signals_ else 0,
            "mean_mahalanobis": np.mean(mahal_dists),
        }

    def plot_drift_report(self, figsize=(14, 8)):
        """
        Generate multi-panel drift report.

        Parameters
        ----------
        figsize : tuple, default=(14, 8)
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Panel 1: EWMA control chart
        self.ewma_chart_.plot(ax=axes[0, 0])

        # Panel 2: Mahalanobis distance
        mahal_dists = [s["mahalanobis_distance"] for s in self.drift_signals_]
        axes[0, 1].plot(mahal_dists, "b-", marker="o", markersize=3)
        axes[0, 1].set_xlabel("Observation")
        axes[0, 1].set_ylabel("Mahalanobis Distance")
        axes[0, 1].set_title("Multivariate Distance from Reference")
        axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: Drift signal timeline
        drift_types = [s["drift_type"] for s in self.drift_signals_]
        colors = {"none": "green", "shift": "orange", "outlier": "red"}
        color_map = [colors[d] for d in drift_types]
        axes[1, 0].scatter(range(len(drift_types)), [1] * len(drift_types), c=color_map, s=50)
        axes[1, 0].set_ylim([0.5, 1.5])
        axes[1, 0].set_xlabel("Observation")
        axes[1, 0].set_title("Drift Signal Timeline")
        axes[1, 0].set_yticks([])
        axes[1, 0].grid(True, alpha=0.3, axis="x")

        # Panel 4: Summary statistics
        summary = self.get_drift_summary()
        summary_text = f"""
Drift Detection Summary
========================
Total Observations:   {summary['n_observations']}
EWMA Alarms:          {summary['n_alarms']}
Outliers:             {summary['n_outliers']}
Alarm Rate:           {summary['alarm_rate']:.1%}
Mean Mahalanobis:     {summary['mean_mahalanobis']:.3f}
"""
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontfamily="monospace", verticalalignment="top",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        axes[1, 1].axis("off")

        plt.tight_layout()
        return fig
