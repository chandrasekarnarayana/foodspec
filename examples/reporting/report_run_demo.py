from __future__ import annotations

import csv
import json
from pathlib import Path

from foodspec.reporting import HtmlReportBuilder, ReportMode, RunBundle, ScientificDossierBuilder
from foodspec.reporting.cards import build_experiment_card_from_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run_dir = Path("examples/reporting/demo_run")
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "manifest.json",
        {
            "protocol_snapshot": {"task": {"name": "demo"}, "modality": "raman"},
            "data_fingerprint": "demo-fingerprint",
            "inputs": ["demo.csv"],
        },
    )
    _write_json(run_dir / "run_summary.json", {"summary": "Demo reporting run"})
    _write_csv(
        run_dir / "tables" / "metrics.csv",
        [{"macro_f1": 0.91, "accuracy": 0.94, "auroc": 0.95}],
    )
    _write_json(run_dir / "qc_report.json", {"status": "pass"})
    _write_json(run_dir / "trust_outputs.json", {"ece": 0.04, "coverage": 0.97})

    bundle = RunBundle.from_run_dir(run_dir)
    HtmlReportBuilder(bundle, ReportMode.RESEARCH).build(run_dir)

    card = build_experiment_card_from_bundle(bundle, mode=ReportMode.RESEARCH)
    cards_dir = run_dir / "cards"
    card.to_markdown(cards_dir / "experiment_card.md")
    card.to_json(cards_dir / "experiment_card.json")

    ScientificDossierBuilder().build(bundle, run_dir, mode="research")

    print(f"Report written to {run_dir / 'reports' / 'report.html'}")


if __name__ == "__main__":
    main()
