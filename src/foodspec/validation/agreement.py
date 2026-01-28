"""Agreement analysis between methods: Bland-Altman, Deming regression."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats

__all__ = ["BlandAltmanAnalysis", "DemingRegression"]


class BlandAltmanAnalysis:
    """
    Bland-Altman agreement analysis between two methods.

    Analyzes systematic and random agreement between measurement methods.
    """

    def __init__(self, confidence: float = 0.95):
        """
        Initialize Bland-Altman analysis.

        Parameters
        ----------
        confidence : float, default=0.95
            Confidence level for limits of agreement.
        """
        self.confidence = confidence

    def calculate(
        self,
        method1: np.ndarray,
        method2: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate Bland-Altman agreement indices.

        Parameters
        ----------
        method1 : np.ndarray, shape (n_samples,)
            Measurements from method 1.
        method2 : np.ndarray, shape (n_samples,)
            Measurements from method 2.

        Returns
        -------
        mean_diff : float
            Mean difference (bias).
        std_diff : float
            Standard deviation of differences.
        lower_loa : float
            Lower limit of agreement (mean - 1.96*std).
        upper_loa : float
            Upper limit of agreement (mean + 1.96*std).
        correlation : float
            Correlation coefficient.

        Notes
        -----
        Bland & Altman (1986). Statistical methods for assessing agreement
        between two methods of clinical measurement.
        Lancet, 1(8476), 307-310.

        Examples
        --------
        >>> m1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> m2 = np.array([1.1, 2.05, 2.95, 3.9, 5.1])
        >>> ba = BlandAltmanAnalysis()
        >>> mean_diff, std_diff, ll, ul, corr = ba.calculate(m1, m2)
        """
        method1 = np.asarray(method1, dtype=np.float64).ravel()
        method2 = np.asarray(method2, dtype=np.float64).ravel()

        if len(method1) != len(method2):
            raise ValueError("method1 and method2 must have same length")

        # Calculate differences
        differences = method1 - method2
        means = (method1 + method2) / 2

        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        # Limits of agreement
        z_critical = stats.norm.ppf((1 + self.confidence) / 2)
        lower_loa = mean_diff - z_critical * std_diff
        upper_loa = mean_diff + z_critical * std_diff

        # Correlation
        correlation = np.corrcoef(method1, method2)[0, 1]

        self.method1_ = method1
        self.method2_ = method2
        self.differences_ = differences
        self.means_ = means
        self.mean_diff_ = mean_diff
        self.std_diff_ = std_diff
        self.lower_loa_ = lower_loa
        self.upper_loa_ = upper_loa

        return mean_diff, std_diff, lower_loa, upper_loa, correlation

    def plot(self, ax=None, title: Optional[str] = None):
        """
        Plot Bland-Altman diagram.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        title : str, optional
            Plot title.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot differences vs means
        ax.scatter(self.means_, self.differences_, alpha=0.6)

        # Plot mean difference
        ax.axhline(self.mean_diff_, color="red", linestyle="-", linewidth=2, label=f"Mean = {self.mean_diff_:.3f}")

        # Plot limits of agreement
        ax.axhline(self.upper_loa_, color="red", linestyle="--", linewidth=1.5, label="±1.96 SD")
        ax.axhline(self.lower_loa_, color="red", linestyle="--", linewidth=1.5)

        ax.set_xlabel("Mean of Methods")
        ax.set_ylabel("Difference (Method 1 - Method 2)")
        if title is None:
            title = "Bland-Altman Plot"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def get_report(self) -> str:
        """
        Generate text report of agreement analysis.

        Returns
        -------
        report : str
            Formatted report.
        """
        report = f"""
Bland-Altman Agreement Analysis
================================
Mean Difference (Bias):     {self.mean_diff_:.4f}
Std Dev of Differences:     {self.std_diff_:.4f}

Limits of Agreement ({self.confidence * 100:.0f}% CI):
  Lower Limit:              {self.lower_loa_:.4f}
  Upper Limit:              {self.upper_loa_:.4f}

Correlation Coefficient:    {np.corrcoef(self.method1_, self.method2_)[0, 1]:.4f}
"""
        return report


class DemingRegression:
    """
    Deming regression for method comparison.

    Accounts for measurement error in both x and y.
    Assumes errors are normally distributed with known variance ratio.
    """

    def __init__(self, variance_ratio: float = 1.0):
        """
        Initialize Deming regression.

        Parameters
        ----------
        variance_ratio : float, default=1.0
            Ratio of error variances (var_y / var_x).
        """
        self.variance_ratio = variance_ratio

    def fit(self, X: np.ndarray, y: np.ndarray) -> DemingRegression:
        """
        Fit Deming regression line.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples,)
            Reference method measurements.
        y : np.ndarray, shape (n_samples,)
            Test method measurements.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        X_centered = X - x_mean
        y_centered = y - y_mean

        s_xx = np.sum(X_centered**2)
        s_yy = np.sum(y_centered**2)
        s_xy = np.sum(X_centered * y_centered)

        # Deming regression slope
        lambda_param = self.variance_ratio
        discriminant = (s_yy - lambda_param * s_xx) ** 2 + 4 * lambda_param * (s_xy**2)
        slope = ((s_yy - lambda_param * s_xx) + np.sqrt(discriminant)) / (2 * s_xy)

        intercept = y_mean - slope * x_mean

        self.slope_ = slope
        self.intercept_ = intercept
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using Deming regression line.

        Parameters
        ----------
        X : np.ndarray
            Reference measurements.

        Returns
        -------
        y_pred : np.ndarray
            Predicted test measurements.
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        return self.slope_ * X + self.intercept_

    def get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get residuals perpendicular to regression line.

        Parameters
        ----------
        X : np.ndarray
            Reference measurements.
        y : np.ndarray
            Test measurements.

        Returns
        -------
        residuals : np.ndarray
            Perpendicular residuals.
        """
        y_pred = self.predict(X)
        # Perpendicular distance to line: |y - y_pred| / sqrt(1 + slope^2)
        return (y - y_pred) / np.sqrt(1 + self.slope_**2)

    def plot(self, ax=None, title: Optional[str] = None):
        """
        Plot Deming regression with data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        title : str, optional
            Plot title.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot data
        ax.scatter(self.X_, self.y_, alpha=0.6, label="Data")

        # Plot regression line
        x_range = np.array([self.X_.min(), self.X_.max()])
        y_range = self.predict(x_range)
        ax.plot(x_range, y_range, "r-", linewidth=2, label=f"y = {self.slope_:.3f}x + {self.intercept_:.3f}")

        # Plot identity line (perfect agreement)
        lim = [min(self.X_.min(), self.y_.min()), max(self.X_.max(), self.y_.max())]
        ax.plot(lim, lim, "k--", alpha=0.5, linewidth=1, label="Perfect agreement")

        ax.set_xlabel("Reference Method")
        ax.set_ylabel("Test Method")
        if title is None:
            title = f"Deming Regression (λ={self.variance_ratio})"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        return ax

    def get_concordance_correlation(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Calculate Lin's concordance correlation coefficient.

        Parameters
        ----------
        X : np.ndarray
            Reference measurements.
        y : np.ndarray
            Test measurements.

        Returns
        -------
        ccc : float
            Concordance correlation coefficient (-1 to 1).

        Notes
        -----
        Lin, L. I. (1989). A concordance correlation coefficient to evaluate
        reproducibility. Biometrics, 45(1), 255-268.
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        mean_x = np.mean(X)
        mean_y = np.mean(y)
        var_x = np.var(X, ddof=1)
        var_y = np.var(y, ddof=1)
        cov_xy = np.cov(X, y, ddof=1)[0, 1]

        numerator = 2 * cov_xy
        denominator = var_x + var_y + (mean_x - mean_y) ** 2

        return numerator / (denominator + 1e-8)

    def get_report(self) -> str:
        """
        Generate text report of Deming regression.

        Returns
        -------
        report : str
            Formatted report.
        """
        ccc = self.get_concordance_correlation(self.X_, self.y_)
        residuals = self.get_residuals(self.X_, self.y_)

        report = f"""
Deming Regression Analysis
==========================
Slope:                      {self.slope_:.4f}
Intercept:                  {self.intercept_:.4f}
Variance Ratio (λ):         {self.variance_ratio:.4f}

Concordance Correlation:    {ccc:.4f}
Std Dev of Residuals:       {np.std(residuals, ddof=1):.4f}

Mean Perpendicular Error:   {np.mean(np.abs(residuals)):.4f}
Max Perpendicular Error:    {np.max(np.abs(residuals)):.4f}
"""
        return report
