from __future__ import annotations

import csv
import json
from pathlib import Path

from foodspec.reporting.html import HtmlReportBuilder
from foodspec.reporting.modes import ReportMode
from foodspec.reporting.schema import RunBundle


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "manifest.json",
        {
            "protocol_snapshot": {"task": {"name": "demo"}},
            "data_fingerprint": "deadbeef",
            "inputs": ["data.csv"],
        },
    )
    _write_json(run_dir / "run_summary.json", {"summary": "Demo summary"})
    _write_csv(run_dir / "tables" / "metrics.csv", [{"macro_f1": 0.92, "accuracy": 0.95}])
    _write_json(run_dir / "qc_report.json", {"status": "pass"})
    _write_json(run_dir / "trust_outputs.json", {"ece": 0.05})
    return run_dir


def test_html_report_mode_sections(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    bundle = RunBundle.from_run_dir(run_dir)

    html_path = HtmlReportBuilder(bundle, ReportMode.RESEARCH).build(run_dir)
    html_text = html_path.read_text(encoding="utf-8")
    assert "QC Report" not in html_text

    html_path = HtmlReportBuilder(bundle, ReportMode.REGULATORY).build(run_dir)
    html_text = html_path.read_text(encoding="utf-8")
    assert "QC Report" in html_text


def test_html_report_with_figures(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    bundle = RunBundle.from_run_dir(run_dir)
    html_path = HtmlReportBuilder(bundle, ReportMode.RESEARCH).build(run_dir, embed_images=False)
    html_text = html_path.read_text(encoding="utf-8")
    assert "<img" in html_text
    assert "figures/plot.png" in html_text
