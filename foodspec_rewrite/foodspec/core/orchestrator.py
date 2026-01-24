"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

ExecutionEngine orchestrates FoodSpec workflows end-to-end.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.manifest import RunManifest
from foodspec.core.protocol import ProtocolV2

try:  # Optional numpy seeding for determinism
    import numpy as np
except ImportError:  # pragma: no cover - optional
    np = None


@dataclass
class RunResult:
    """Result bundle returned by the execution engine."""

    output_dir: Path
    manifest: RunManifest
    logs: List[str] = field(default_factory=list)


class ExecutionEngine:
    """Execute a ProtocolV2 workflow with deterministic orchestration.

    Only minimal functionality is implemented; each stage is optional and
    raises NotImplementedError if explicitly requested.
    """

    def __init__(self) -> None:
        self.logs: List[str] = []

    def _log(self, msg: str) -> None:
        self.logs.append(msg)

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if np is not None:
            np.random.seed(seed)
        self._log(f"Seeds set to {seed}")

    def run(
        self,
        protocol_or_path: Union[ProtocolV2, str, Path],
        outdir: Union[str, Path],
        seed: int = 0,
    ) -> RunResult:
        """Run the workflow, returning a RunResult.

        Minimal implementation: validates and records manifest; other stages
        raise NotImplementedError only if requested by protocol contents.
        """

        self.logs.clear()
        self._seed(seed)

        protocol = self._load_protocol(protocol_or_path)
        self._log("Protocol loaded and defaults applied")

        output_dir = Path(outdir)
        artifacts = ArtifactRegistry(output_dir)
        artifacts.ensure_layout()
        self._log(f"Artifact layout ensured at {output_dir}")

        # Minimal stage checks: if the user requests more than the minimal
        # pipeline, we surface explicit NotImplementedError to keep behavior
        # deterministic and clear.
        self._check_stage_requests(protocol)

        # Build manifest (uses data fingerprint if file exists)
        data_path = Path(protocol.data.input)
        data_file = data_path if data_path.exists() else None
        manifest = RunManifest.build(
            protocol_snapshot=protocol.model_dump(mode="python"),
            data_path=data_file,
            seed=seed,
            artifacts={
                "metrics": str(artifacts.metrics_path),
                "qc": str(artifacts.qc_path),
                "predictions": str(artifacts.predictions_path),
                "plots": str(artifacts.plots_dir),
                "report_html": str(artifacts.report_html_path),
                "report_pdf": str(artifacts.report_pdf_path),
                "bundle": str(artifacts.bundle_dir),
                "manifest": str(artifacts.manifest_path),
                "logs": str(artifacts.logs_path),
            },
            warnings=[] if data_file else ["Data file not found; fingerprint omitted."],
        )
        artifacts.write_json(artifacts.manifest_path, json.loads(json.dumps(manifest.__dict__)))
        self._log("Manifest written")

        # Write logs
        artifacts.logs_path.write_text("\n".join(self.logs))

        return RunResult(output_dir=output_dir, manifest=manifest, logs=list(self.logs))

    def _load_protocol(self, protocol_or_path: Union[ProtocolV2, str, Path]) -> ProtocolV2:
        if isinstance(protocol_or_path, ProtocolV2):
            return protocol_or_path.apply_defaults()
        return ProtocolV2.load(protocol_or_path)

    def _check_stage_requests(self, protocol: ProtocolV2) -> None:
        """Surface NotImplementedError only when a stage is requested."""

        if protocol.preprocess.recipe or protocol.preprocess.steps:
            raise NotImplementedError("Preprocess stage not implemented yet.")
        if protocol.qc.thresholds or protocol.qc.metrics:
            raise NotImplementedError("QC stage not implemented yet.")
        if protocol.features.modules or protocol.features.strategy not in {"auto", ""}:
            raise NotImplementedError("Features stage not implemented yet.")
        if protocol.model.estimator not in {"logreg", ""}:  # baseline accepted but not run
            raise NotImplementedError("Model stage not implemented yet.")
        if protocol.validation.scheme not in {"train_test_split", ""}:
            raise NotImplementedError("Validation stage not implemented yet.")
        if protocol.uncertainty.conformal:
            raise NotImplementedError("Uncertainty stage not implemented yet.")
        if protocol.interpretability.methods or protocol.interpretability.marker_panel:
            raise NotImplementedError("Interpretability stage not implemented yet.")
        if protocol.visualization.plots:
            raise NotImplementedError("Visualization stage not implemented yet.")
        if protocol.reporting.format not in {"markdown", ""} or protocol.reporting.sections:
            if protocol.reporting.sections not in ([], ["summary", "metrics", "figures"]):
                raise NotImplementedError("Reporting stage not implemented yet.")
        if protocol.export.bundle:
            raise NotImplementedError("Export stage not implemented yet.")


__all__ = ["ExecutionEngine", "RunResult"]
