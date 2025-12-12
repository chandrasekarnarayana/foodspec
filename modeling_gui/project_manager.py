"""Simple project manager for FoodSpec GUI.

Allows grouping multiple datasets with roles/instrument/batch labels and saving
to a JSON file for reuse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List


@dataclass
class ProjectEntry:
    path: str
    instrument: str = ""
    batch: str = ""
    matrix: str = ""
    role: str = ""


@dataclass
class Project:
    name: str
    entries: List[ProjectEntry] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "entries": [asdict(e) for e in self.entries]}, indent=2)

    @staticmethod
    def from_json(text: str) -> "Project":
        payload = json.loads(text)
        entries = [ProjectEntry(**e) for e in payload.get("entries", [])]
        return Project(name=payload.get("name", "project"), entries=entries)

    def save(self, path: Path):
        path.write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Project":
        return Project.from_json(path.read_text())

