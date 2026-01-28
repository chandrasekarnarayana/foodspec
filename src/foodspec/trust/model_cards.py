"""Model card helpers for trust and regulatory readiness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _format_list(items: List[str]) -> str:
    if not items:
        return "None recorded."
    return "\n".join([f"- {item}" for item in items])


def _format_dict(data: Dict[str, Any]) -> str:
    if not data:
        return "None recorded."
    lines = []
    for key, value in data.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


@dataclass
class ModelCard:
    name: str
    version: str
    overview: str
    intended_use: str
    non_goals: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    training_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    conformal: Dict[str, Any] = field(default_factory=dict)
    qc: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    failure_modes: List[str] = field(default_factory=list)
    ethics: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "overview": self.overview,
            "intended_use": self.intended_use,
            "non_goals": list(self.non_goals),
            "limitations": list(self.limitations),
            "training_data": dict(self.training_data),
            "metrics": dict(self.metrics),
            "calibration": dict(self.calibration),
            "conformal": dict(self.conformal),
            "qc": dict(self.qc),
            "reproducibility": dict(self.reproducibility),
            "failure_modes": list(self.failure_modes),
            "ethics": self.ethics,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Model Card",
            "",
            "## Overview",
            self.overview,
            "",
            "## Intended Use",
            self.intended_use,
            "",
            "## Non-goals",
            _format_list(self.non_goals),
            "",
            "## Limitations",
            _format_list(self.limitations),
            "",
            "## Training Data Provenance",
            _format_dict(self.training_data),
            "",
            "## Metrics Summary",
            _format_dict(self.metrics),
            "",
            "## Calibration Summary",
            _format_dict(self.calibration),
            "",
            "## Conformal Summary",
            _format_dict(self.conformal),
            "",
            "## QC Constraints",
            _format_dict(self.qc),
            "",
            "## Reproducibility",
            _format_dict(self.reproducibility),
            "",
            "## Failure Modes",
            _format_list(self.failure_modes),
        ]
        if self.ethics:
            lines.extend(["", "## Ethics", self.ethics])
        return "\n".join(lines).strip() + "\n"


def write_model_card(run_dir: str | Path, card: ModelCard | Dict[str, Any], format: str = "md") -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = card.to_dict() if isinstance(card, ModelCard) else dict(card)
    if format == "json":
        path = run_dir / "model_card.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    if format != "md":
        raise ValueError("format must be 'md' or 'json'")
    path = run_dir / "model_card.md"
    content = ModelCard(**payload).to_markdown() if not isinstance(card, ModelCard) else card.to_markdown()
    path.write_text(content, encoding="utf-8")
    return path


__all__ = ["ModelCard", "write_model_card"]
