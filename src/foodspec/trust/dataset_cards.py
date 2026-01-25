"""Dataset card helpers for trust and regulatory readiness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetCard:
    name: str
    description: str
    collection: str
    features: List[str] = field(default_factory=list)
    licensing: Optional[str] = None
    privacy: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "collection": self.collection,
            "features": list(self.features),
            "licensing": self.licensing,
            "privacy": self.privacy,
            "notes": self.notes,
        }


__all__ = ["DatasetCard"]

