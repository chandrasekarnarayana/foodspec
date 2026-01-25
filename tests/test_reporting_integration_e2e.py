from __future__ import annotations

import json
from pathlib import Path

import pytest

from foodspec.reporting import (
    ReportMode,
    ReportContext,
    ReportBuilder,
    build_experiment_card,
    ConfidenceLevel,
)
from foodspec.core.manifest import RunManifest

pytestmark = pytest.mark.no_cover


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_end_to_end_report_generation(tmp_path: Path) -> None:
    """End-to-end: context load → card → HTML/JSON/Markdown outputs."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal data file for fingerprinting
    data_path = run_dir / "data.csv"
    data_path.write_text("x\n1\n")

    # Metrics and trust outputs (good performance, low risk)
    metrics_rows = [
        {
            "macro_f1": 0.92,
            "auroc": 0.95,
            "ece": 0.05,
        }
    ]
    _write_csv(run_dir / "metrics.csv", metrics_rows)

    trust_payload = {
        "ece": 0.05,
        "coverage": 0.98,
        "abstain_rate": 0.02,
    }
    _write_json(run_dir / "trust_outputs.json", trust_payload)

    # Optional QC table
    qc_rows = [{"metric": "snr", "value": 0.95}]
    _write_csv(run_dir / "qc.csv", qc_rows)

    # Protocol snapshot and manifest
    protocol_snapshot = {
        "version": "1.0.0",
        "task": {"name": "oil-auth"},
        "modality": "raman",
        "model": {"name": "svm"},
        "validation": {"scheme": "cv"},
    }
    manifest = RunManifest.build(
        protocol_snapshot=protocol_snapshot,
        data_path=data_path,
        seed=42,
        artifacts={
            "metrics": "metrics.csv",
            "qc": "qc.csv",
            "trust_outputs": "trust_outputs.json",
        },
    )
    manifest.save(run_dir / "manifest.json")

    # Load context and build outputs
    context = ReportContext.load(run_dir)
    card = build_experiment_card(context, mode=ReportMode.RESEARCH)

    assert card.confidence_level is ConfidenceLevel.HIGH
    assert card.deployment_readiness.value == "ready"
    assert card.macro_f1 == 0.92
    assert card.auroc == 0.95
    assert card.ece == 0.05
    assert card.coverage == 0.98
    assert card.abstain_rate == 0.02

    # Exports
    html_path = run_dir / "report.html"
    ReportBuilder(context).build_html(html_path, mode=ReportMode.RESEARCH, title="Demo")
    assert html_path.exists()
    assert "Demo" in html_path.read_text()

    json_path = run_dir / "card.json"
    md_path = run_dir / "card.md"
    card.to_json(json_path)
    card.to_markdown(md_path)
    assert json_path.exists()
    assert md_path.exists()

    # JSON export converts enums to strings
    payload = json.loads(json_path.read_text())
    assert payload["confidence_level"] == "high"
    assert payload["deployment_readiness"] == "ready"

    # Markdown export contains key sections
    md_text = md_path.read_text()
    assert "Experiment Card" in md_text
    assert "Confidence" in md_text
    assert "Risks" in md_text
