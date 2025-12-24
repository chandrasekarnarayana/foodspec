"""YAML-driven experiment engine for reproducible FoodSpec runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from foodspec.core.run_record import RunRecord, _capture_environment, _hash_path


def _require_keys(section: Dict[str, Any], required: List[str], context: str) -> None:
    missing = [k for k in required if k not in section]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {', '.join(missing)}")


def _hash_dict(payload: Dict[str, Any]) -> str:
    """Stable 8-char hash for dictionaries used to track step configs."""

    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


@dataclass
class DatasetSpec:
    """Dataset location and schema specification."""

    path: Path
    modality: str = "raman"
    schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "modality": self.modality,
            "schema": self.schema,
        }


@dataclass
class ExperimentConfig:
    """Complete experiment configuration loaded from exp.yml."""

    dataset: DatasetSpec
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    qc: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    modeling: Dict[str, Any] = field(default_factory=dict)
    reporting: Dict[str, Any] = field(default_factory=dict)
    seeds: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    moats: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        """Load and validate an experiment YAML file."""

        yaml_path = Path(path)
        config = yaml.safe_load(yaml_path.read_text())
        if not isinstance(config, dict):
            raise ValueError("exp.yml must be a mapping")

        _require_keys(config, ["dataset", "preprocessing", "qc", "features", "modeling", "reporting"], "exp.yml")

        dataset_cfg = config["dataset"]
        _require_keys(dataset_cfg, ["path"], "dataset")
        dataset_path = Path(dataset_cfg["path"]).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        dataset = DatasetSpec(
            path=dataset_path,
            modality=dataset_cfg.get("modality", "raman"),
            schema=dataset_cfg.get("schema", {}),
        )

        return cls(
            dataset=dataset,
            preprocessing=config.get("preprocessing", {}),
            qc=config.get("qc", {}),
            features=config.get("features", {}),
            modeling=config.get("modeling", {}),
            reporting=config.get("reporting", {}),
            seeds=config.get("seeds", {}),
            outputs=config.get("outputs", {}),
            moats=config.get("moats", {}),
        )

    @property
    def config_hash(self) -> str:
        return _hash_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "preprocessing": self.preprocessing,
            "qc": self.qc,
            "features": self.features,
            "modeling": self.modeling,
            "reporting": self.reporting,
            "seeds": self.seeds,
            "outputs": self.outputs,
            "moats": self.moats,
        }

    def _step_payloads(self) -> List[tuple[str, Dict[str, Any]]]:
        return [
            ("dataset", self.dataset.to_dict()),
            ("preprocessing", self.preprocessing),
            ("qc", self.qc),
            ("features", self.features),
            ("modeling", self.modeling),
            ("reporting", self.reporting),
        ]

    def build_run_record(self) -> RunRecord:
        """Create a RunRecord capturing hashes, seeds, and output targets."""

        dataset_hash = _hash_path(self.dataset.path)
        record = RunRecord(
            workflow_name="experiment",
            config=self.to_dict(),
            dataset_hash=dataset_hash,
            environment=_capture_environment(),
            random_seeds=self.seeds or {},
        )

        for name, payload in self._step_payloads():
            record.add_step(name, _hash_dict(payload), metadata=payload)

        base_dir = self.outputs.get("base_dir")
        if base_dir:
            record.add_output_path(base_dir)

        return record

    def summary(self) -> str:
        lines = [
            "Experiment Configuration",
            f"  Dataset: {self.dataset.path} ({self.dataset.modality})",
            f"  Preprocessing: {list(self.preprocessing.keys()) or 'unspecified'}",
            f"  QC thresholds: {self.qc.get('thresholds', 'unspecified')}",
            f"  Features: {self.features.get('preset', 'custom')}",
            f"  Modeling suite: {self.modeling.get('suite', [])}",
            f"  Reporting targets: {self.reporting.get('targets', [])}",
            f"  Seeds: {self.seeds if self.seeds else 'none provided'}",
            f"  Moats: {list(self.moats.keys()) or 'none'}",
        ]
        return "\n".join(lines)


class ExperimentEngine:
    """Convenience wrapper around ExperimentConfig for loading and summarizing exp.yml."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentEngine":
        return cls(ExperimentConfig.from_yaml(path))

    def build_run_record(self) -> RunRecord:
        return self.config.build_run_record()

    def summary(self) -> str:
        return self.config.summary()

    @property
    def config_hash(self) -> str:
        return self.config.config_hash


__all__ = ["DatasetSpec", "ExperimentConfig", "ExperimentEngine"]
