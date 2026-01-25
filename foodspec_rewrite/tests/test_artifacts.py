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
"""

import json
from pathlib import Path

from foodspec.core.artifacts import ArtifactRegistry


def test_paths_and_layout(tmp_path: Path) -> None:
    reg = ArtifactRegistry(tmp_path)
    reg.ensure_layout()

    assert reg.metrics_path == tmp_path / "metrics.csv"
    assert reg.qc_path == tmp_path / "qc.csv"
    assert reg.predictions_path == tmp_path / "predictions.csv"
    assert reg.plots_dir.is_dir()
    assert reg.bundle_dir.is_dir()
    assert reg.report_html_path == tmp_path / "report.html"
    assert reg.report_pdf_path == tmp_path / "report.pdf"
    assert reg.manifest_path == tmp_path / "manifest.json"
    assert reg.logs_path == tmp_path / "logs.txt"


def test_write_csv_and_json(tmp_path: Path) -> None:
    reg = ArtifactRegistry(tmp_path)
    reg.ensure_layout()

    reg.write_csv(reg.metrics_path, [{"metric": "accuracy", "value": 0.9}])
    reg.write_json(reg.manifest_path, {"version": "2.0.0"})

    metrics = reg.metrics_path.read_text().strip().splitlines()
    assert metrics[0] == "metric,value"
    assert metrics[1] == "accuracy,0.9"

    manifest = json.loads(reg.manifest_path.read_text())
    assert manifest["version"] == "2.0.0"
