"""QC policy definitions and evaluators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class QCPolicy:
    """Centralized QC thresholds and evaluation helpers."""

    required: bool = False
    thresholds: Dict[str, float] = field(default_factory=dict)
    actions: Dict[str, str] = field(default_factory=dict)
    min_health_score: float = 0.7
    max_spike_fraction: float = 0.02
    max_saturation_fraction: float = 0.05
    max_baseline_lowfreq: float = 5.0
    max_outlier_fraction: float = 0.1
    max_imbalance_ratio: float = 10.0
    min_samples_per_class: int = 20

    def __post_init__(self) -> None:
        for key, value in self.thresholds.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def evaluate_spectrum(self, health, outliers=None) -> Dict[str, Any]:
        """Evaluate spectral QC results against policy thresholds."""
        flags: List[str] = []
        table = health.table
        metrics = {
            "health_mean": float(health.aggregates.get("health_mean", table["health_score"].mean())),
            "spike_fraction_mean": float(table["spike_fraction"].mean()),
            "saturation_fraction_mean": float(table["saturation_fraction"].mean()),
            "baseline_lowfreq_mean": float(table["baseline_lowfreq"].mean()),
        }
        if metrics["health_mean"] < self.min_health_score:
            flags.append("low_health_score")
        if metrics["spike_fraction_mean"] > self.max_spike_fraction:
            flags.append("excess_spike_fraction")
        if metrics["saturation_fraction_mean"] > self.max_saturation_fraction:
            flags.append("excess_saturation")
        if metrics["baseline_lowfreq_mean"] > self.max_baseline_lowfreq:
            flags.append("baseline_drift")

        if outliers is not None:
            outlier_rate = float(np.mean(outliers.labels)) if len(outliers.labels) else 0.0
            metrics["outlier_fraction"] = outlier_rate
            if outlier_rate > self.max_outlier_fraction:
                flags.append("excess_outliers")

        return {
            "status": "pass" if not flags else "fail",
            "flags": flags,
            "scores": metrics,
            "recommended_actions": [self.actions.get(f, f) for f in flags],
        }

    def evaluate_dataset(self, balance: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate dataset balance diagnostics against policy thresholds."""
        flags: List[str] = []
        metrics = {
            "imbalance_ratio": float(balance.get("imbalance_ratio", 0.0)),
            "undersized_classes": balance.get("undersized_classes", []),
        }
        if metrics["imbalance_ratio"] > self.max_imbalance_ratio:
            flags.append("severe_imbalance")
        if metrics["undersized_classes"]:
            flags.append("undersized_classes")
        return {
            "status": "pass" if not flags else "fail",
            "flags": flags,
            "scores": metrics,
            "recommended_actions": [self.actions.get(f, f) for f in flags],
        }

    def evaluate_spectral(self, health, outliers=None) -> Dict[str, Any]:
        """Backwards-compatible alias for evaluate_spectrum."""
        return self.evaluate_spectrum(health, outliers)

    def evaluate_balance(self, balance: Dict[str, Any]) -> Dict[str, Any]:
        """Backwards-compatible alias for evaluate_dataset."""
        return self.evaluate_dataset(balance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "thresholds": dict(self.thresholds),
            "actions": dict(self.actions),
            "min_health_score": self.min_health_score,
            "max_spike_fraction": self.max_spike_fraction,
            "max_saturation_fraction": self.max_saturation_fraction,
            "max_baseline_lowfreq": self.max_baseline_lowfreq,
            "max_outlier_fraction": self.max_outlier_fraction,
            "max_imbalance_ratio": self.max_imbalance_ratio,
            "min_samples_per_class": self.min_samples_per_class,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QCPolicy":
        if payload is None:
            payload = {}
        thresholds = payload.get("thresholds", {})
        actions = payload.get("actions", {})
        return cls(
            required=bool(payload.get("required", False)),
            thresholds=thresholds,
            actions=actions,
            min_health_score=payload.get("min_health_score", cls.min_health_score),
            max_spike_fraction=payload.get("max_spike_fraction", cls.max_spike_fraction),
            max_saturation_fraction=payload.get("max_saturation_fraction", cls.max_saturation_fraction),
            max_baseline_lowfreq=payload.get("max_baseline_lowfreq", cls.max_baseline_lowfreq),
            max_outlier_fraction=payload.get("max_outlier_fraction", cls.max_outlier_fraction),
            max_imbalance_ratio=payload.get("max_imbalance_ratio", cls.max_imbalance_ratio),
            min_samples_per_class=payload.get("min_samples_per_class", cls.min_samples_per_class),
        )


__all__ = ["QCPolicy"]
