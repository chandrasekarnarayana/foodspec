from __future__ import annotations

"""Experiment runner (scaffold).

Defines `ExperimentSpec` and `ExperimentRunner` for YAML/JSON-based experiment
execution that wires protocols, inputs, and overrides into repeatable runs.

This is a scaffold only: parsing, validation, and execution are stubs.
"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ExperimentSpec:
    """Declarative experiment specification (scaffold).

    Parameters
    ----------
    name : str
        Experiment name.
    version : str
        Experiment version.
    protocol : Union[str, Path]
        Path to protocol file or named protocol id.
    inputs : List[Union[str, Path]]
        Input dataset paths.
    overrides : Dict[str, Any]
        Protocol parameter overrides (e.g., normalization, baseline).
    seed : int
        Random seed for reproducibility.

    Methods
    -------
    validate()
        Basic schema validation.
    to_dict()
        JSON-friendly representation.
    __hash__()
        Hash for reproducibility tracking.
    """

    name: str
    version: str = "0.1.0"
    protocol: Union[str, Path] = ""
    inputs: List[Union[str, Path]] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def validate(self) -> Dict[str, Any]:
        issues: List[str] = []
        if not self.name:
            issues.append("name required")
        if not self.protocol:
            issues.append("protocol required")
        if not isinstance(self.inputs, list) or not self.inputs:
            issues.append("inputs must be a non-empty list")
        return {"ok": len(issues) == 0, "issues": issues}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "protocol": str(self.protocol),
            "inputs": [str(p) for p in self.inputs],
            "overrides": dict(self.overrides),
            "seed": int(self.seed),
        }

    def __hash__(self) -> int:
        key = (
            self.name,
            self.version,
            str(self.protocol),
            tuple(str(p) for p in self.inputs),
            tuple(sorted(self.overrides.items())),
            self.seed,
        )
        return hash(key)


class ExperimentRunner:
    """Run experiments from an `ExperimentSpec` (scaffold).

    Methods
    -------
    from_file(path)
        Parse YAML/JSON into `ExperimentSpec`.
    run(spec)
        Execute the experiment and return a summary dict.
    summary()
        Return human-readable summary string of last run.
    """

    def __init__(self):
        self.last_result: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_file(path: Union[str, Path]) -> ExperimentSpec:
        """Load experiment spec from YAML/JSON file (placeholder)."""
        # TODO: Implement YAML/JSON parsing and schema validation
        p = Path(path)
        _ = p.read_text()  # placeholder read
        return ExperimentSpec(name=p.stem, protocol="examples/protocols/Example.yml", inputs=["data/example.csv"])  # type: ignore[arg-type]

    def run(self, spec: ExperimentSpec) -> Dict[str, Any]:
        """Execute experiment (placeholder).

        Returns a dict summary with keys like 'name', 'protocol', 'inputs', 'status'.
        """
        # TODO: Wire into foodspec.protocol.ProtocolRunner and execute
        self.last_result = {
            "name": spec.name,
            "protocol": str(spec.protocol),
            "inputs": [str(p) for p in spec.inputs],
            "status": "not_implemented",
        }
        return self.last_result

    def summary(self) -> str:
        """Human-readable summary of the last run (placeholder)."""
        if not self.last_result:
            return "No experiment run yet."
        return f"Experiment {self.last_result['name']} -> status: {self.last_result['status']}"


__all__ = ["ExperimentSpec", "ExperimentRunner"]
