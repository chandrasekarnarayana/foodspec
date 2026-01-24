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

from pathlib import Path

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.reporting import generate_html_report


def test_generate_html_report_writes_file(tmp_path: Path) -> None:
    artifacts = ArtifactRegistry(tmp_path)
    artifacts.ensure_layout()

    manifest = {"version": "2.0.0", "seed": 42, "data_path": "data.csv"}
    dataset_summary = {"rows": 100, "features": 5}
    preprocessing_steps = ["normalize", "smooth"]
    qc_table = [{"metric": "snr", "value": 0.95}]
    metrics = [{"metric": "accuracy", "value": 0.9}]
    plots = ["plots/sample.png"]
    uncertainty = {"coverage": 0.9}

    path = generate_html_report(
        artifacts=artifacts,
        manifest=manifest,
        dataset_summary=dataset_summary,
        preprocessing_steps=preprocessing_steps,
        qc_table=qc_table,
        metrics=metrics,
        plots=plots,
        uncertainty=uncertainty,
    )

    assert path.exists()
    content = path.read_text()
    assert "FoodSpec Run Report" in content
    assert "normalize" in content
    assert "accuracy" in content
