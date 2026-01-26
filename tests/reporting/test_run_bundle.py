"""Tests for RunBundle artifact discovery and reporting API alignment."""
from __future__ import annotations

import json
from pathlib import Path

from foodspec.reporting.api import build_report_from_run
from foodspec.reporting.schema import RunBundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_runbundle_loads_trust_outputs(tmp_path: Path) -> None:
    _write_json(tmp_path / "manifest.json", {"protocol_snapshot": {}})
    _write_json(tmp_path / "run_summary.json", {})
    trust_dir = tmp_path / "trust"
    trust_dir.mkdir(parents=True, exist_ok=True)
    _write_json(trust_dir / "calibration.json", {"ece": 0.1})

    bundle = RunBundle.from_run_dir(tmp_path)

    assert "calibration" in bundle.trust_outputs
    assert bundle.trust_outputs["calibration"]["ece"] == 0.1


def test_runbundle_collects_figures(tmp_path: Path) -> None:
    _write_json(tmp_path / "manifest.json", {"protocol_snapshot": {}})
    _write_json(tmp_path / "run_summary.json", {})
    fig_dir = tmp_path / "plots" / "viz" / "uncertainty"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "calibration.png"
    fig_path.write_bytes(b"fake")

    bundle = RunBundle.from_run_dir(tmp_path)

    assert any(path.name == "calibration.png" for path in bundle.figures)


def test_build_report_from_run_uses_run_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "manifest.json", {"protocol_snapshot": {}})
    _write_json(run_dir / "run_summary.json", {})

    out_dir = tmp_path / "out"
    artifacts = build_report_from_run(run_dir, out_dir=out_dir, mode="research", pdf=False, title="Test")

    assert Path(artifacts["report_html"]).exists()
    assert (out_dir / "reports" / "report.html").exists()
    assert (out_dir / "card.json").exists()
    assert (out_dir / "cards" / "experiment_card.json").exists()
