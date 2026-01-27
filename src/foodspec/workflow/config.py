"""Workflow configuration and dataclasses for orchestrator.

Defines WorkflowConfig, which captures all CLI inputs and protocol overrides
for a single workflow execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class WorkflowConfig:
    """Configuration for a workflow run.

    Captures protocol path, input files, output directory, mode, and CLI overrides.

    Examples
    --------
    Create a workflow config and execute it::

        cfg = WorkflowConfig(
            protocol="examples/protocols/oils.yaml",
            inputs=[Path("data.csv")],
            output_dir=Path("runs/exp1"),
            mode="research",
            seed=42,
        )
    """

    # Required
    protocol: Path | str
    """Path to protocol YAML/JSON file."""

    inputs: List[Path | str]
    """Input CSV file paths."""

    # Optional with defaults
    output_dir: Optional[Path | str] = None
    """Output directory for run. Auto-generated if not provided."""

    mode: str = "research"
    """Workflow mode: 'research' or 'regulatory'."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    # Optional CLI overrides (only if protocol allows_override)
    scheme: Optional[str] = None
    """Validation scheme: 'random', 'lobo', 'loso', 'nested'."""

    model: Optional[str] = None
    """Model name override."""

    feature_type: Optional[str] = None
    """Feature type override."""

    label_col: Optional[str] = None
    """Label column name override."""

    group_col: Optional[str] = None
    """Group column name override."""

    # Pipeline control
    enable_preprocessing: bool = True
    enable_features: bool = True
    enable_modeling: bool = True
    enable_viz: bool = True

    # Phase 2: QC and regulatory enforcement
    enforce_qc: bool = False
    """Enforce QC gates (fail on gate failure). Default: advisory."""

    enable_trust: bool = False
    """Enable trust stack (calibration + conformal + abstention)."""

    enable_reporting: bool = True
    """Enable structured reporting."""

    allow_placeholder_trust: bool = False
    """Allow placeholder trust in strict regulatory mode (for development). Default: False (reject placeholder)."""

    # Advanced
    generate_pdf: bool = False
    """Generate PDF report in addition to HTML."""

    verbose: bool = False
    """Enable verbose logging."""

    dry_run: bool = False
    """Validate config but don't execute."""

    # Capture of what was overridden (for logging)
    cli_overrides: Dict[str, Any] = field(default_factory=dict)
    """Dict of {field_name: original_value} for fields overridden via CLI."""

    def __post_init__(self):
        """Normalize paths."""
        self.protocol = Path(self.protocol)
        self.inputs = [Path(p) for p in self.inputs]
        if self.output_dir:
            self.output_dir = Path(self.output_dir)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate workflow config.

        Returns
        -------
        (is_valid, errors)
            Tuple of boolean validity and list of error messages.
        """
        errors = []

        if not self.protocol.exists():
            errors.append(f"Protocol file not found: {self.protocol}")

        if not self.inputs:
            errors.append("No input files provided")

        for inp in self.inputs:
            if not inp.exists():
                errors.append(f"Input file not found: {inp}")

        if self.mode not in ("research", "regulatory"):
            errors.append(f"Mode must be 'research' or 'regulatory', got: {self.mode}")

        if self.seed is not None and not isinstance(self.seed, int):
            errors.append(f"Seed must be int, got: {type(self.seed)}")

        return len(errors) == 0, errors

    def summary(self) -> str:
        """Return human-readable summary of config."""
        parts = [
            f"Protocol: {self.protocol}",
            f"Inputs: {len(self.inputs)} file(s)",
            f"Mode: {self.mode}",
            f"Seed: {self.seed}",
        ]
        if self.cli_overrides:
            parts.append(f"CLI Overrides: {', '.join(self.cli_overrides.keys())}")
        return " | ".join(parts)
