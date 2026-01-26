"""QC gates for workflow enforcement.

Provides 3 QC gates with real metrics and configurable thresholds:
1. DataIntegrityGate - checks input data quality
2. SpectralQualityGate - checks spectral data quality  
3. ModelReliabilityGate - checks model performance

Each gate produces a GateResult with pass/fail/warn/skip status and
saves metrics to artifacts/.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class GateResult:
    """Result of a single QC gate evaluation."""

    name: str
    """Gate name (e.g. 'data_integrity')."""

    status: str  # "pass" | "fail" | "warn" | "skip"
    """Outcome: pass, fail, warn, or skip."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    """Computed metrics as dict."""

    thresholds: Dict[str, Any] = field(default_factory=dict)
    """Applied thresholds."""

    message: str = ""
    """Human-readable summary."""

    remediation: List[str] = field(default_factory=list)
    """Suggestions for fixing failures."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    def summary_str(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"[{self.status.upper()}] {self.name}",
            f"Message: {self.message}",
            f"Metrics: {len(self.metrics)} computed",
        ]
        if self.remediation:
            lines.append(f"Remediation:")
            for hint in self.remediation:
                lines.append(f"  - {hint}")
        return "\n".join(lines)


class DataIntegrityGate:
    """QC Gate 1: Check input data quality."""

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        """Initialize gate with thresholds.
        
        Parameters
        ----------
        thresholds : Optional[Dict[str, Any]]
            Override defaults with custom thresholds. Keys:
            - max_missingness_per_col: float (default 0.05)
            - max_duplicate_fraction: float (default 0.02)
            - min_rows: int (default 20)
            - min_samples_per_class: int (default 5)
        """
        self.thresholds = thresholds or {}
        self.max_missingness = self.thresholds.get("max_missingness_per_col", 0.05)
        self.max_duplicates = self.thresholds.get("max_duplicate_fraction", 0.02)
        self.min_rows = self.thresholds.get("min_rows", 20)
        self.min_per_class = self.thresholds.get("min_samples_per_class", 5)

    def run(self, df: pd.DataFrame, label_col: Optional[str] = None) -> GateResult:
        """Evaluate data integrity.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        label_col : Optional[str]
            Label column name for class balance check.
        
        Returns
        -------
        GateResult
        """
        metrics = {}
        issues = []

        # Row count
        metrics["row_count"] = len(df)
        if len(df) < self.min_rows:
            issues.append(f"Too few rows: {len(df)} < {self.min_rows}")

        # Column missingness
        missing_per_col = {}
        for col in df.columns:
            missing_frac = df[col].isna().sum() / len(df)
            missing_per_col[col] = round(missing_frac, 3)
            if missing_frac > self.max_missingness:
                issues.append(f"Column '{col}' missing: {100*missing_frac:.1f}% > {100*self.max_missingness:.1f}%")

        metrics["missing_per_column"] = missing_per_col
        metrics["max_missingness_observed"] = round(max(missing_per_col.values()), 3) if missing_per_col else 0.0

        # Duplicate rows
        n_duplicates = df.duplicated().sum()
        dup_frac = n_duplicates / len(df) if len(df) > 0 else 0.0
        metrics["duplicate_rows"] = int(n_duplicates)
        metrics["duplicate_fraction"] = round(dup_frac, 3)
        if dup_frac > self.max_duplicates:
            issues.append(f"Duplicate rows: {100*dup_frac:.1f}% > {100*self.max_duplicates:.1f}%")

        # Label distribution
        if label_col and label_col in df.columns:
            label_counts = df[label_col].value_counts()
            metrics["label_distribution"] = label_counts.to_dict()
            metrics["n_classes"] = len(label_counts)

            # Check class imbalance
            min_class_count = label_counts.min()
            if min_class_count < self.min_per_class:
                issues.append(f"Class imbalance: min class has {min_class_count} samples < {self.min_per_class}")

            # Entropy as diversity metric
            probs = label_counts / label_counts.sum()
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            metrics["label_entropy"] = round(entropy, 3)

        # Decide status
        if issues:
            status = "fail"
            message = f"{len(issues)} issue(s) detected"
            remediation = issues
        else:
            status = "pass"
            message = "Data integrity checks passed"
            remediation = []

        return GateResult(
            name="data_integrity",
            status=status,
            metrics=metrics,
            thresholds={
                "max_missingness_per_col": self.max_missingness,
                "max_duplicate_fraction": self.max_duplicates,
                "min_rows": self.min_rows,
                "min_samples_per_class": self.min_per_class,
            },
            message=message,
            remediation=remediation,
        )


class SpectralQualityGate:
    """QC Gate 2: Check spectral data quality."""

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        """Initialize gate with thresholds.
        
        Parameters
        ----------
        thresholds : Optional[Dict[str, Any]]
            Override defaults. Keys:
            - max_outlier_fraction: float (default 0.05)
            - min_snr: float (default 5.0)
        """
        self.thresholds = thresholds or {}
        self.max_outlier_frac = self.thresholds.get("max_outlier_fraction", 0.05)
        self.min_snr = self.thresholds.get("min_snr", 5.0)

    def run(
        self,
        df: pd.DataFrame,
        spectral_cols: Optional[List[str]] = None,
    ) -> GateResult:
        """Evaluate spectral quality.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data with spectral features.
        spectral_cols : Optional[List[str]]
            Column names of spectral features. If None, tries to auto-detect.
        
        Returns
        -------
        GateResult
        """
        metrics = {}
        issues = []

        # Auto-detect spectral columns (numeric, many per sample)
        if spectral_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            spectral_cols = numeric_cols

        if not spectral_cols:
            return GateResult(
                name="spectral_quality",
                status="skip",
                message="No spectral columns detected; skipping spectral QC.",
                metrics={},
            )

        # Extract spectral data
        try:
            X = df[spectral_cols].to_numpy(dtype=float)
        except Exception as e:
            return GateResult(
                name="spectral_quality",
                status="fail",
                message=f"Failed to extract spectral data: {e}",
                metrics={},
                remediation=["Check spectral column data types and values."],
            )

        # Outlier detection (z-score on total intensity)
        intensities = X.sum(axis=1)
        z_scores = np.abs((intensities - np.mean(intensities)) / (np.std(intensities) + 1e-8))
        outlier_mask = z_scores > 2.5  # Use 2.5 instead of 3.0 for better detection
        outlier_frac = outlier_mask.sum() / len(X)
        metrics["outlier_fraction"] = round(outlier_frac, 3)
        metrics["n_outliers"] = int(outlier_mask.sum())

        if outlier_frac > self.max_outlier_frac:
            issues.append(f"Outlier fraction {100*outlier_frac:.1f}% > {100*self.max_outlier_frac:.1f}%")

        # SNR proxy (signal-to-noise ratio as robust metric)
        # Use: SNR ~ (p95 - p5) / std(gradient)
        p95 = np.percentile(X, 95, axis=1)
        p5 = np.percentile(X, 5, axis=1)
        signal_est = p95 - p5
        diffs = np.diff(X, axis=1)
        noise_est = np.std(diffs, axis=1) + 1e-8
        snr = signal_est / noise_est
        metrics["snr_median"] = round(float(np.median(snr)), 3)
        metrics["snr_p10"] = round(float(np.percentile(snr, 10)), 3)

        if metrics["snr_p10"] < self.min_snr:
            issues.append(f"SNR (p10) {metrics['snr_p10']} < {self.min_snr}")

        # Saturation (high intensity fraction)
        max_per_spectrum = X.max(axis=1)
        p99_overall = np.percentile(X, 99)
        saturation_frac = (max_per_spectrum > 0.95 * np.max(X)).sum() / len(X)
        metrics["saturation_fraction"] = round(saturation_frac, 3)

        # Decide status
        if issues:
            status = "fail"
            message = f"{len(issues)} spectral quality issue(s)"
            remediation = issues
        else:
            status = "pass"
            message = "Spectral quality checks passed"
            remediation = []

        return GateResult(
            name="spectral_quality",
            status=status,
            metrics=metrics,
            thresholds={
                "max_outlier_fraction": self.max_outlier_frac,
                "min_snr": self.min_snr,
            },
            message=message,
            remediation=remediation,
        )


class ModelReliabilityGate:
    """QC Gate 3: Check model reliability."""

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        """Initialize gate with thresholds.
        
        Parameters
        ----------
        thresholds : Optional[Dict[str, Any]]
            Override defaults. Keys:
            - min_accuracy: float (default 0.7)
            - max_calibration_error: float (default 0.1)
        """
        self.thresholds = thresholds or {}
        self.min_accuracy = self.thresholds.get("min_accuracy", 0.7)
        self.max_calib_error = self.thresholds.get("max_calibration_error", 0.1)

    def run(self, modeling_result: Optional[Dict[str, Any]] = None) -> GateResult:
        """Evaluate model reliability.
        
        Parameters
        ----------
        modeling_result : Optional[Dict[str, Any]]
            Result dict from fit_predict or similar, with 'metrics' key.
        
        Returns
        -------
        GateResult
        """
        if modeling_result is None or not modeling_result.get("metrics"):
            return GateResult(
                name="model_reliability",
                status="skip",
                message="No modeling results; skipping model reliability check.",
                metrics={},
            )

        metrics_dict = modeling_result.get("metrics", {})
        metrics = dict(metrics_dict)  # Copy
        issues = []

        # Check accuracy if present
        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            if accuracy < self.min_accuracy:
                issues.append(f"Accuracy {accuracy:.3f} < {self.min_accuracy}")
        else:
            # Try to use f1 or other metric as fallback
            f1 = metrics.get("f1")
            if f1 is not None and f1 < self.min_accuracy:
                issues.append(f"F1 score {f1:.3f} < {self.min_accuracy}")

        # Check calibration error if present (ECE or similar)
        ece = metrics.get("ece")
        if ece is not None:
            if ece > self.max_calib_error:
                issues.append(f"Calibration error {ece:.3f} > {self.max_calib_error}")

        # Decide status
        if issues:
            status = "fail"
            message = f"{len(issues)} model reliability issue(s)"
            remediation = issues
        else:
            status = "pass"
            message = "Model reliability checks passed"
            remediation = []

        return GateResult(
            name="model_reliability",
            status=status,
            metrics=metrics,
            thresholds={
                "min_accuracy": self.min_accuracy,
                "max_calibration_error": self.max_calib_error,
            },
            message=message,
            remediation=remediation,
        )
