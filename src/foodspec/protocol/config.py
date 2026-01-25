from __future__ import annotations
"""
Protocol configuration dataclasses.

Provides ProtocolConfig for defining protocol specifications including:
- Protocol metadata (name, version, description)
- Step definitions
- Expected columns and metadata
- Validation strategy

Part of the protocol execution framework.
"""


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


@dataclass
class ProtocolRunResult:
    """Result from running a protocol."""

    run_dir: Optional[Path]
    logs: List[str]
    metadata: Dict[str, Any]
    tables: Dict[str, Any]  # Dict[str, pd.DataFrame]
    figures: Dict[str, Any]
    report: str
    summary: str
