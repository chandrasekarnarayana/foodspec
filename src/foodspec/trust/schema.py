from __future__ import annotations

"""Schema objects for trust artifacts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CalibrationArtifact:
    method: str
    model_path: str
    n_calibration: int
    n_test: int
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "model_path": self.model_path,
            "n_calibration": int(self.n_calibration),
            "n_test": int(self.n_test),
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "warnings": self.warnings,
        }


@dataclass
class ConformalArtifact:
    alpha: float
    condition_key: Optional[str]
    coverage: Optional[float]
    mean_set_size: float
    coverage_ci: Optional[tuple[float, float]] = None
    efficiency: Optional[float] = None
    efficiency_ci: Optional[tuple[float, float]] = None
    coverage_curve: List[Dict[str, float]] = field(default_factory=list)
    per_group: Optional[List[Dict[str, Any]]] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "condition_key": self.condition_key,
            "coverage": self.coverage,
            "mean_set_size": float(self.mean_set_size),
            "coverage_ci": self.coverage_ci,
            "efficiency": self.efficiency,
            "efficiency_ci": self.efficiency_ci,
            "coverage_curve": self.coverage_curve,
            "per_group": self.per_group,
            "warnings": self.warnings,
        }


@dataclass
class AbstentionArtifact:
    tau: float
    max_set_size: Optional[int]
    density_threshold: Optional[float]
    abstain_rate: float
    accuracy_on_answered: Optional[float]
    risk_coverage: Dict[str, List[float]]
    reasons: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau": float(self.tau),
            "max_set_size": self.max_set_size,
            "density_threshold": self.density_threshold,
            "abstain_rate": float(self.abstain_rate),
            "accuracy_on_answered": self.accuracy_on_answered,
            "risk_coverage": self.risk_coverage,
            "reasons": self.reasons,
        }


@dataclass
class ReadinessArtifact:
    score: float
    components: Dict[str, float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "components": self.components,
            "notes": self.notes,
        }


__all__ = [
    "CalibrationArtifact",
    "ConformalArtifact",
    "AbstentionArtifact",
    "ReadinessArtifact",
]
