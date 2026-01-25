"""Model card helpers for trust and regulatory readiness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelCard:
    name: str
    version: str
    intended_use: str
    limitations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    data_summary: Dict[str, str] = field(default_factory=dict)
    ethics: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "intended_use": self.intended_use,
            "limitations": list(self.limitations),
            "metrics": dict(self.metrics),
            "data_summary": dict(self.data_summary),
            "ethics": self.ethics,
        }


__all__ = ["ModelCard"]

