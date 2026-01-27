from __future__ import annotations

"""Regulatory readiness scoring."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional


@dataclass
class RegulatoryReadiness:
    score: float
    components: Dict[str, float]
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {"score": float(self.score), "components": self.components, "notes": self.notes}


def compute_readiness_score(
    checklist: Mapping[str, float | bool],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> RegulatoryReadiness:
    """Compute readiness score from a checklist."""
    components: Dict[str, float] = {}
    notes: List[str] = []
    for key, value in checklist.items():
        if isinstance(value, bool):
            components[key] = 1.0 if value else 0.0
        else:
            components[key] = float(value)
        if components[key] < 0.5:
            notes.append(f"{key} needs improvement")
    weights = weights or {k: 1.0 for k in components}
    total_weight = sum(weights.get(k, 1.0) for k in components)
    score = 0.0
    for key, value in components.items():
        score += value * weights.get(key, 1.0)
    score = 100.0 * score / total_weight if total_weight else 0.0
    return RegulatoryReadiness(score=score, components=components, notes=notes)


def evaluate_run_readiness(run_dir: Path, trust_payload: Mapping[str, object]) -> RegulatoryReadiness:
    """Evaluate readiness from run artifacts and trust outputs."""
    run_dir = Path(run_dir)
    notes: List[str] = []
    manifest_ok = (run_dir / "manifest.json").exists()
    summary_ok = (run_dir / "run_summary.json").exists()
    logs_ok = (run_dir / "logs" / "run.log").exists()
    reproducibility = float(manifest_ok and summary_ok and logs_ok)
    if not reproducibility:
        notes.append("Reproducibility artifacts incomplete")

    metrics_present = (run_dir / "tables" / "metrics.csv").exists()
    validation_rigor = 1.0 if metrics_present else 0.0
    if not metrics_present:
        notes.append("Validation metrics missing")

    drift_present = (run_dir / "qc_report.json").exists()
    drift_monitoring = 1.0 if drift_present else 0.0

    calib = trust_payload.get("calibration", {}) if isinstance(trust_payload, Mapping) else {}
    ece = None
    if isinstance(calib, Mapping):
        ece = calib.get("metrics_after", {}).get("ece")
    calibration_quality = 0.0
    if ece is None:
        notes.append("Calibration metrics missing")
    else:
        calibration_quality = 1.0 if float(ece) <= 0.1 else 0.5

    conformal = trust_payload.get("conformal", {}) if isinstance(trust_payload, Mapping) else {}
    coverage = None
    alpha = None
    if isinstance(conformal, Mapping):
        coverage = conformal.get("coverage")
        alpha = conformal.get("alpha")
    uncertainty_guarantees = 0.0
    if coverage is not None and alpha is not None:
        target = 1.0 - float(alpha)
        uncertainty_guarantees = 1.0 if float(coverage) >= target - 0.02 else 0.5
    else:
        notes.append("Conformal coverage missing")

    model_card = (run_dir / "model_card.md").exists() or (run_dir / "cards" / "model_card.md").exists()
    dataset_card = (run_dir / "dataset_card.md").exists() or (run_dir / "cards" / "dataset_card.md").exists()
    documentation = 1.0 if (model_card and dataset_card) else (0.5 if (model_card or dataset_card) else 0.0)
    if documentation == 0.0:
        notes.append("Model/dataset cards missing")

    checklist = {
        "validation_rigor": validation_rigor,
        "drift_monitoring": drift_monitoring,
        "calibration_quality": calibration_quality,
        "uncertainty_guarantees": uncertainty_guarantees,
        "documentation_completeness": documentation,
        "reproducibility_artifacts": reproducibility,
    }
    readiness = compute_readiness_score(checklist)
    readiness.notes.extend(notes)
    return readiness


def load_trust_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


__all__ = ["RegulatoryReadiness", "compute_readiness_score", "evaluate_run_readiness", "load_trust_payload"]
