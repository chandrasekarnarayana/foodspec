"""Dataset card helpers for trust and regulatory readiness."""
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
class DatasetCard:
    name: str
    description: str
    collection: str
    features: List[str] = field(default_factory=list)
    size: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)
    splits: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    qc_summary: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    licensing: Optional[str] = None
    privacy: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "collection": self.collection,
            "features": list(self.features),
            "size": dict(self.size),
            "labels": dict(self.labels),
            "splits": dict(self.splits),
            "provenance": dict(self.provenance),
            "qc_summary": dict(self.qc_summary),
            "limitations": list(self.limitations),
            "licensing": self.licensing,
            "privacy": self.privacy,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Dataset Card",
            "",
            "## Description",
            self.description,
            "",
            "## Collection",
            self.collection,
            "",
            "## Features",
            _format_list(self.features),
            "",
            "## Size",
            _format_dict(self.size),
            "",
            "## Labels",
            _format_dict(self.labels),
            "",
            "## Splits",
            _format_dict(self.splits),
            "",
            "## Provenance",
            _format_dict(self.provenance),
            "",
            "## QC Summary",
            _format_dict(self.qc_summary),
            "",
            "## Limitations",
            _format_list(self.limitations),
        ]
        if self.licensing:
            lines.extend(["", "## Licensing", self.licensing])
        if self.privacy:
            lines.extend(["", "## Privacy", self.privacy])
        if self.notes:
            lines.extend(["", "## Notes", self.notes])
        return "\n".join(lines).strip() + "\n"


def write_dataset_card(run_dir: str | Path, card: DatasetCard | Dict[str, Any], format: str = "md") -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = card.to_dict() if isinstance(card, DatasetCard) else dict(card)
    if format == "json":
        path = run_dir / "dataset_card.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    if format != "md":
        raise ValueError("format must be 'md' or 'json'")
    path = run_dir / "dataset_card.md"
    content = DatasetCard(**payload).to_markdown() if not isinstance(card, DatasetCard) else card.to_markdown()
    path.write_text(content, encoding="utf-8")
    return path


__all__ = ["DatasetCard", "write_dataset_card"]
