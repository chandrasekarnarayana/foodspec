"""Advanced QC Control Charts: CUSUM and Process Capability Indices."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats

__all__ = ["CUSUMChart", "CapabilityIndices", "ProcessCapability"]


class CUSUMChart:
    """
    Cumulative Sum Control Chart (CUSUM).

    More sensitive to small drifts than Shewhart charts.
    Accounts for mean shift over time with memory.
    """

    def __init__(
        self,
        target: float = 0,
        k: float = 0.5,
        h: float = 4.77,
        confidence: float = 0.99,
    ):
        """
        Initialize CUSUM chart.

        Parameters
        ----------
        target : float, default=0
            Target/mean value for the process.
        k : float, default=0.5
            Reference value (sensitivity parameter). Typically 0.5 × standard deviation.
        h : float, default=4.77
            Decision limit height. Higher → fewer false alarms. Default gives ~ARL=500.
        confidence : float, default=0.99
            Confidence level for reference calculations.

        Notes
        -----
        CUSUM is optimal for detecting small shifts (0.5-2 sigma).
        - h=4.77 gives ARL₀ ≈ 500 (false alarm every ~500 points)
        - k=0.5 detects 1-sigma shift efficiently
        
        References
        ----------
        Page, E. S. (1954). Continuous inspection schemes.
        Biometrika, 41(1/2), 100-115.
        """
        self.target = target
        self.k = k
        self.h = h
        self.confidence = confidence

    def initialize(self, X: np.ndarray) -> CUSUMChart:
        """
        Initialize CUSUM chart from reference data.

        Parameters
        ----------
        X : np.ndarray, shape (n_reference,)
            Reference/calibration data.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64).ravel()

        if len(X) < 3:
            raise ValueError("Need at least 3 reference points")

        self.mean_ = np.mean(X)
        self.std_ = np.std(X, ddof=1)

        # Use target if provided, otherwise use reference mean
        if self.target is None:
            self.target_ = self.mean_
        else:
            self.target_ = self.target

        # Reference value k is typically 0.5*sigma if not specified
        if self.k is None:
            self.k_ = 0.5 * self.std_
        else:
            self.k_ = self.k

        self.cusum_pos_ = []  # Upper CUSUM
        self.cusum_neg_ = []  # Lower CUSUM
        self.observations_ = []
        self.alarms_pos_ = []  # Upper control limit violations
        self.alarms_neg_ = []  # Lower control limit violations

        self.C_pos = 0  # Upper cumulative sum
        self.C_neg = 0  # Lower cumulative sum

        return self

    def update(self, x: float) -> Tuple[float, float, bool]:
        """
        Update CUSUM with new observation.

        Parameters
        ----------
        x : float
            New observation.

        Returns
        -------
        C_pos : float
            Upper CUSUM value.
        C_neg : float
            Lower CUSUM value.
        is_alarm : bool
            True if CUSUM exceeds decision limit (h).

        Notes
        -----
        C_pos = max(0, C_pos + (x - target - k))
        C_neg = min(0, C_neg + (x - target + k))
        """
        # Standardize observation
        z = (x - self.target_) / self.std_ if self.std_ > 0 else 0

        # Update cumulative sums
        self.C_pos = max(0, self.C_pos + (z - self.k_))
        self.C_neg = min(0, self.C_neg + (z + self.k_))

        self.cusum_pos_.append(self.C_pos)
        self.cusum_neg_.append(self.C_neg)
        self.observations_.append(x)

        # Check for alarms
        is_alarm = (abs(self.C_pos) > self.h) or (abs(self.C_neg) > self.h)
        self.alarms_pos_.append(self.C_pos > self.h)
        self.alarms_neg_.append(self.C_neg < -self.h)

        return self.C_pos, self.C_neg, is_alarm

    def process(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process stream of observations.

        Parameters
        ----------
        X : np.ndarray, shape (n_obs,)
            Observations to process.

        Returns
        -------
        C_pos : np.ndarray
            Upper CUSUM values.
        C_neg : np.ndarray
            Lower CUSUM values.
        alarms : np.ndarray
            Boolean array indicating alarms.
        """
        X = np.asarray(X, dtype=np.float64).ravel()

        for x in X:
            self.update(x)

        C_pos = np.array(self.cusum_pos_)
        C_neg = np.array(self.cusum_neg_)
        alarms = np.logical_or(self.alarms_pos_, self.alarms_neg_)

        return C_pos, C_neg, alarms

    def plot(self, ax=None):
        """
        Plot CUSUM chart.

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

        times = np.arange(len(self.cusum_pos_))

        # Plot CUSUM curves
        ax.plot(times, self.cusum_pos_, "b-", linewidth=1.5, label="C+")
        ax.plot(times, self.cusum_neg_, "r-", linewidth=1.5, label="C-")

        # Plot decision limits
        ax.axhline(self.h, color="blue", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(-self.h, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

        # Highlight alarms
        alarms = np.logical_or(self.alarms_pos_, self.alarms_neg_)
        alarm_idx = np.where(alarms)[0]
        if len(alarm_idx) > 0:
            cusum_at_alarms_pos = [self.cusum_pos_[i] if self.alarms_pos_[i] else None for i in alarm_idx]
            cusum_at_alarms_neg = [self.cusum_neg_[i] if self.alarms_neg_[i] else None for i in alarm_idx]
            ax.scatter(alarm_idx, cusum_at_alarms_pos, color="blue", s=100, marker="X", zorder=5)
            ax.scatter(alarm_idx, cusum_at_alarms_neg, color="red", s=100, marker="X", zorder=5)

        ax.set_xlabel("Sample Number")
        ax.set_ylabel("CUSUM Value")
        ax.set_title(f"CUSUM Control Chart (h={self.h}, k={self.k_})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def get_run_length(self) -> Dict[str, Any]:
        """
        Calculate run length statistics.

        Returns
        -------
        stats : dict
            Run length statistics (time since last alarm).
        """
        alarms = np.logical_or(self.alarms_pos_, self.alarms_neg_)
        alarm_indices = np.where(alarms)[0]

        if len(alarm_indices) == 0:
            return {
                "n_observations": len(self.observations_),
                "n_alarms": 0,
                "run_length": len(self.observations_),
                "avg_run_length": None,
            }

        run_lengths = np.diff(alarm_indices)
        return {
            "n_observations": len(self.observations_),
            "n_alarms": len(alarm_indices),
            "run_length": len(self.observations_) - alarm_indices[-1],
            "avg_run_length": float(np.mean(run_lengths)) if len(run_lengths) > 0 else None,
            "min_run_length": int(np.min(run_lengths)) if len(run_lengths) > 0 else None,
            "max_run_length": int(np.max(run_lengths)) if len(run_lengths) > 0 else None,
        }


class CapabilityIndices:
    """Calculate process capability indices (Cp, Cpk, Pp, Ppk)."""

    @staticmethod
    def calculate(
        X: np.ndarray,
        lower_spec: float,
        upper_spec: float,
        sample_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Calculate process capability indices.

        Parameters
        ----------
        X : np.ndarray
            Process measurements.
        lower_spec : float
            Lower specification limit (LSL).
        upper_spec : float
            Upper specification limit (USL).
        sample_size : int, optional
            Subgroup size for Cp/Cpk calculation. If None, uses Pp/Ppk.

        Returns
        -------
        indices : dict
            Dictionary with keys:
            - Pp: Preliminary process capability
            - Ppk: Preliminary process capability (one-sided)
            - Cp: Process capability (assuming subgroups)
            - Cpk: Process capability (one-sided)
            - Cpm: Taguchi capability index

        Notes
        -----
        Cp/Cpk: Used for rational subgroups (subgroup strategy known)
        Pp/Ppk: Used for individual measurements
        
        Cpk = min((USL - mean) / (3*sigma), (mean - LSL) / (3*sigma))

        Cpm = (USL - LSL) / (6 * sqrt(sigma^2 + (mean - target)^2))
        """
        X = np.asarray(X, dtype=np.float64).ravel()

        if len(X) < 2:
            raise ValueError("Need at least 2 measurements")

        mean = np.mean(X)
        std_overall = np.std(X, ddof=1)

        # Specification limits
        tolerance = upper_spec - lower_spec
        center = (upper_spec + lower_spec) / 2

        # Preliminary capability (individuals)
        Pp = tolerance / (6 * std_overall) if std_overall > 0 else np.inf
        Ppk_upper = (upper_spec - mean) / (3 * std_overall) if std_overall > 0 else np.inf
        Ppk_lower = (mean - lower_spec) / (3 * std_overall) if std_overall > 0 else np.inf
        Ppk = min(Ppk_upper, Ppk_lower)

        # Process capability (subgrouped)
        if sample_size is not None and sample_size > 1:
            # Estimate within-group variance
            n_subgroups = len(X) // sample_size
            if n_subgroups > 1:
                subgroup_ranges = []
                for i in range(n_subgroups):
                    subgroup = X[i * sample_size : (i + 1) * sample_size]
                    subgroup_ranges.append(np.max(subgroup) - np.min(subgroup))

                avg_range = np.mean(subgroup_ranges)
                # d2 constant for Shewhart charts (approximation)
                d2_dict = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326}
                d2 = d2_dict.get(sample_size, 1.128)
                std_within = avg_range / d2

                Cp = tolerance / (6 * std_within) if std_within > 0 else np.inf
                Cpk_upper = (upper_spec - mean) / (3 * std_within) if std_within > 0 else np.inf
                Cpk_lower = (mean - lower_spec) / (3 * std_within) if std_within > 0 else np.inf
                Cpk = min(Cpk_upper, Cpk_lower)
            else:
                Cp = Pp
                Cpk = Ppk
        else:
            Cp = Pp
            Cpk = Ppk

        # Taguchi capability index
        Cpm = tolerance / (6 * np.sqrt(std_overall ** 2 + (mean - center) ** 2)) if std_overall > 0 or mean != center else np.inf

        # Process yield (defect rate)
        Z_lower = (mean - lower_spec) / std_overall if std_overall > 0 else np.inf
        Z_upper = (upper_spec - mean) / std_overall if std_overall > 0 else np.inf
        ppm_lower = stats.norm.cdf(-Z_lower) * 1e6
        ppm_upper = (1 - stats.norm.cdf(Z_upper)) * 1e6
        ppm_total = ppm_lower + ppm_upper

        return {
            "Pp": float(Pp),
            "Ppk": float(Ppk),
            "Cp": float(Cp),
            "Cpk": float(Cpk),
            "Cpm": float(Cpm),
            "ppm_lower": float(ppm_lower),
            "ppm_upper": float(ppm_upper),
            "ppm_total": float(ppm_total),
            "process_yield": 100 * (1 - ppm_total / 1e6),
        }

    @staticmethod
    def classify_capability(Cpk: float) -> str:
        """
        Classify process capability based on Cpk.

        Parameters
        ----------
        Cpk : float
            Capability index.

        Returns
        -------
        classification : str
            Capability classification.
        """
        if Cpk >= 1.67:
            return "Excellent"
        elif Cpk >= 1.33:
            return "Very Good"
        elif Cpk >= 1.0:
            return "Good"
        elif Cpk >= 0.67:
            return "Acceptable"
        else:
            return "Unacceptable"


class ProcessCapability:
    """
    Comprehensive process capability analysis.

    Combines multiple capability indices with visual reporting.
    """

    def __init__(self, lower_spec: float, upper_spec: float):
        """
        Initialize process capability analyzer.

        Parameters
        ----------
        lower_spec : float
            Lower specification limit.
        upper_spec : float
            Upper specification limit.
        """
        self.lower_spec = lower_spec
        self.upper_spec = upper_spec
        self.results_ = None

    def analyze(self, X: np.ndarray, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform comprehensive capability analysis.

        Parameters
        ----------
        X : np.ndarray
            Process measurements.
        sample_size : int, optional
            Subgroup size.

        Returns
        -------
        results : dict
            Comprehensive analysis results.
        """
        X = np.asarray(X, dtype=np.float64).ravel()

        indices = CapabilityIndices.calculate(X, self.lower_spec, self.upper_spec, sample_size)

        self.results_ = {
            "n_observations": len(X),
            "mean": float(np.mean(X)),
            "std": float(np.std(X, ddof=1)),
            "min": float(np.min(X)),
            "max": float(np.max(X)),
            "specification_limits": {
                "lower": self.lower_spec,
                "upper": self.upper_spec,
                "center": (self.lower_spec + self.upper_spec) / 2,
            },
            "indices": indices,
            "capability_classification": CapabilityIndices.classify_capability(indices["Cpk"]),
            "within_spec": np.sum((X >= self.lower_spec) & (X <= self.upper_spec)) / len(X),
        }

        return self.results_

    def report(self) -> str:
        """
        Generate text report of capability analysis.

        Returns
        -------
        report : str
            Formatted report.
        """
        if self.results_ is None:
            return "No analysis performed. Call analyze() first."

        r = self.results_
        idx = r["indices"]

        report = f"""
Process Capability Analysis
============================
Sample Size:                {r['n_observations']}
Mean:                       {r['mean']:.4f}
Std Dev:                    {r['std']:.4f}
Min:                        {r['min']:.4f}
Max:                        {r['max']:.4f}

Specification Limits:
  Lower (LSL):              {r['specification_limits']['lower']:.4f}
  Upper (USL):              {r['specification_limits']['upper']:.4f}
  Center (Target):          {r['specification_limits']['center']:.4f}

Process Capability Indices:
  Pp (Preliminary):         {idx['Pp']:.4f}
  Ppk (Preliminary):        {idx['Ppk']:.4f}
  Cp (Process):             {idx['Cp']:.4f}
  Cpk (Process):            {idx['Cpk']:.4f}
  Cpm (Taguchi):            {idx['Cpm']:.4f}

Process Quality:
  Within Specification:     {r['within_spec']:.1%}
  Defects (PPM):            {idx['ppm_total']:.0f}
  Lower Defects:            {idx['ppm_lower']:.0f}
  Upper Defects:            {idx['ppm_upper']:.0f}

Classification:             {r['capability_classification']}
"""
        return report
