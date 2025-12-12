"""
Feature & model registry for provenance and audit trails.
Stores entries in a JSON index (could be swapped for SQLite later).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hash_dataset(df) -> str:
    try:
        return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
    except Exception:
        return "unknown"


@dataclass
class RegistryEntry:
    dataset_hash: str
    protocol_name: str
    protocol_version: str
    preprocessing: Dict[str, Any]
    features: List[Dict[str, Any]]
    model_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    validation_strategy: Optional[str] = None
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    inputs: List[str] = field(default_factory=list)


class FeatureModelRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: List[RegistryEntry] = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self.entries = [RegistryEntry(**e) for e in raw]
            except Exception:
                self.entries = []

    def add_entry(self, entry: RegistryEntry):
        self.entries.append(entry)
        self._save()

    def register_run(self, run_id: str, metadata: Dict[str, Any]):
        """
        Lightweight run registration (without model).
        """
        entry = RegistryEntry(
            dataset_hash=metadata.get("dataset_hash", "unknown"),
            protocol_name=metadata.get("protocol", "unknown"),
            protocol_version=metadata.get("protocol_version", "unknown"),
            preprocessing=metadata.get("preprocessing", {}),
            features=metadata.get("features", []),
            metrics=metadata.get("metrics", {}),
            provenance={
                "timestamp": metadata.get("timestamp"),
                "user": metadata.get("user"),
                "tool_version": metadata.get("tool_version"),
            },
            run_id=run_id,
            validation_strategy=metadata.get("validation_strategy"),
            inputs=metadata.get("inputs", []),
        )
        self.add_entry(entry)

    def register_model(self, run_id: str, model_path: str, model_metadata: Dict[str, Any]):
        """
        Log a model produced in a run.
        """
        entry = RegistryEntry(
            dataset_hash=model_metadata.get("dataset_hash", "unknown"),
            protocol_name=model_metadata.get("protocol_name", "unknown"),
            protocol_version=model_metadata.get("protocol_version", "unknown"),
            preprocessing=model_metadata.get("preprocessing", {}),
            features=model_metadata.get("features", []),
            model_id=model_metadata.get("model_id", Path(model_path).stem),
            model_path=model_path,
            model_type=model_metadata.get("model_type"),
            metrics=model_metadata.get("metrics", {}),
            provenance={
                "timestamp": model_metadata.get("timestamp"),
                "user": model_metadata.get("user"),
                "tool_version": model_metadata.get("tool_version"),
            },
            run_id=run_id,
            validation_strategy=model_metadata.get("validation_strategy"),
            inputs=model_metadata.get("inputs", []),
        )
        self.add_entry(entry)

    def _save(self):
        self.path.write_text(json.dumps([asdict(e) for e in self.entries], indent=2), encoding="utf-8")

    def query_by_feature(self, feature_name: str) -> List[RegistryEntry]:
        return [e for e in self.entries if any(f.get("name") == feature_name for f in e.features)]

    def query_by_protocol(self, name: str, version: Optional[str] = None) -> List[RegistryEntry]:
        return [
            e
            for e in self.entries
            if e.protocol_name == name and (version is None or e.protocol_version == version)
        ]

    def list_models(self) -> List[str]:
        return [e.model_id for e in self.entries if e.model_id]
