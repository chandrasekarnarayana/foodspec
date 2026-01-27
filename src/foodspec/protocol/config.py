"""
Protocol configuration dataclasses.

Provides ProtocolConfig for defining protocol specifications including:
- Protocol metadata (name, version, description)
- Step definitions
- Expected columns and metadata
- Validation strategy

Part of the protocol execution framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass
class ProtocolConfig:
    """Configuration for a protocol execution."""

    name: str
    description: str = ""
    when_to_use: str = ""
    version: str = "0.1.0"
    min_foodspec_version: Optional[str] = None
    seed: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_columns: Dict[str, str] = field(default_factory=dict)
    report_templates: Dict[str, str] = field(default_factory=dict)
    required_metadata: List[str] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    validation_strategy: str = "standard"  # standard | batch_aware | group_stratified
    qc: Dict[str, Any] = field(default_factory=dict)
    # Outcome typing for modeling
    outcome_type: str = "classification"  # classification | regression | count | survival
    target_column: Optional[str] = None
    event_column: Optional[str] = None      # survival
    time_column: Optional[str] = None       # survival
    exposure_columns: List[str] = field(default_factory=list)  # for 2SLS / offsets
    instrument_columns: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProtocolConfig":
        """Create ProtocolConfig from dictionary."""
        return ProtocolConfig(
            name=d.get("name", "Unnamed_Protocol"),
            description=d.get("description", ""),
            when_to_use=d.get("when_to_use", ""),
            version=d.get("version", d.get("protocol_version", "0.1.0")),
            min_foodspec_version=d.get("min_foodspec_version"),
            seed=d.get("seed", 0),
            steps=d.get("steps", []),
            expected_columns=d.get("expected_columns", {}),
            report_templates=d.get("report_templates", {}),
            required_metadata=d.get("required_metadata", []),
            inputs=d.get("inputs", []),
            validation_strategy=d.get("validation_strategy", "standard"),
            qc=d.get("qc", {}),
            outcome_type=d.get("outcome_type", d.get("task", {}).get("outcome_type", "classification")),
            target_column=d.get("target_column", d.get("task", {}).get("target_column")),
            event_column=d.get("event_column", d.get("task", {}).get("event_column")),
            time_column=d.get("time_column", d.get("task", {}).get("time_column")),
            exposure_columns=d.get("exposure_columns", d.get("task", {}).get("exposure_columns", [])),
            instrument_columns=d.get("instrument_columns", d.get("task", {}).get("instrument_columns", [])),
        )

    @staticmethod
    def from_file(path: Union[str, Path]) -> "ProtocolConfig":
        """Load protocol configuration from YAML or JSON file."""
        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise ImportError("PyYAML not installed.")
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        return ProtocolConfig.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Convert protocol config to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "version": self.version,
            "min_foodspec_version": self.min_foodspec_version,
            "seed": self.seed,
            "steps": self.steps,
            "expected_columns": self.expected_columns,
            "report_templates": self.report_templates,
            "required_metadata": self.required_metadata,
            "inputs": self.inputs,
            "validation_strategy": self.validation_strategy,
            "qc": self.qc,
            "outcome_type": self.outcome_type,
            "target_column": self.target_column,
            "event_column": self.event_column,
            "time_column": self.time_column,
            "exposure_columns": self.exposure_columns,
            "instrument_columns": self.instrument_columns,
        }


@dataclass
class ProtocolRunResult:
    """Result from running a protocol."""

    run_dir: Optional[Path]
    logs: List[str]
    metadata: Dict[str, Any]
    tables: Dict[str, Any]  # Dict[str, pd.DataFrame]
    figures: Dict[str, Any]
    report: str = ""
    summary: str = ""
    qc_artifacts: Dict[str, Any] = field(default_factory=dict)
