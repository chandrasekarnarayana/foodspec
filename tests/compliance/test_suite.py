from __future__ import annotations

import json
from pathlib import Path

from foodspec.compliance import check_cfr_part11, check_iso_17025


def _write_minimal_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / "run.log").write_text("log\n")
    (run_dir / "run_summary.json").write_text(json.dumps({"status": "success"}))
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "tables" / "metrics.csv").write_text("metric,value\naccuracy,1.0\n")
    manifest = {"timestamp": "2025-01-01T00:00:00Z", "git_commit": "abc123", "protocol_snapshot": {"name": "demo"}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


def test_iso_17025_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    result = check_iso_17025(run_dir)
    assert result.standard == "ISO_17025"
    assert "manifest" in result.checks


def test_cfr_part11_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    result = check_cfr_part11(run_dir)
    assert result.standard == "FDA_21_CFR_PART_11"
    assert "timestamp" in result.checks
