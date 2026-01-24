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

ArtifactRegistry manages standard run artifacts under a run directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class ArtifactRegistry:
    """Manage standard artifact locations and safe writes.

    Standard layout (within ``root``)::
        metrics.csv
        qc.csv
        predictions.csv
        plots/
        report.html
        report.pdf
        bundle/
        manifest.json
        logs.txt

    Examples
    --------
    Create a run layout and write metrics::

        reg = ArtifactRegistry(Path("/tmp/run"))
        reg.ensure_layout()
        reg.write_csv(reg.metrics_path, [{"metric": "accuracy", "value": 0.91}])
        reg.write_json(reg.manifest_path, {"version": "2.0.0"})
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    # Path helpers
    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.csv"

    @property
    def qc_path(self) -> Path:
        return self.root / "qc.csv"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.csv"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    @property
    def report_html_path(self) -> Path:
        return self.root / "report.html"

    @property
    def report_pdf_path(self) -> Path:
        return self.root / "report.pdf"

    @property
    def bundle_dir(self) -> Path:
        return self.root / "bundle"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs_path(self) -> Path:
        return self.root / "logs.txt"

    def ensure_layout(self) -> None:
        """Create root and standard directories if missing."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(self, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        """Write rows to CSV with headers inferred from first row."""

        self._ensure_parent(path)
        rows = list(rows)
        if not rows:
            path.write_text("")
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Write a mapping to JSON (pretty-printed)."""

        self._ensure_parent(path)
        path.write_text(json.dumps(payload, indent=2))

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)


__all__ = ["ArtifactRegistry"]
