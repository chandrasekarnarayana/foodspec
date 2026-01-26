from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from foodspec.cli.main import app


def test_cli_report_run_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    df = pd.DataFrame(
        {
            "oil_type": ["A", "A", "B", "B"],
            "matrix": ["oil"] * 4,
            "heating_stage": [0, 1, 0, 1],
            "I_1742": [10, 9, 6, 5],
            "I_2720": [5, 5, 5, 5],
        }
    )
    df.to_csv(csv_path, index=False)

    proto_path = tmp_path / "proto.json"
    proto_path.write_text(
        json.dumps(
            {
                "name": "report_protocol",
                "steps": [
                    {
                        "type": "rq_analysis",
                        "params": {
                            "oil_col": "oil_type",
                            "matrix_col": "matrix",
                            "heating_col": "heating_stage",
                            "ratios": [
                                {
                                    "name": "1742/2720",
                                    "numerator": "I_1742",
                                    "denominator": "I_2720",
                                }
                            ],
                        },
                    },
                    {"type": "output", "params": {"output_dir": str(tmp_path / "runs")}},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "report_run"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "report",
            "run",
            "--input",
            str(csv_path),
            "--protocol",
            str(proto_path),
            "--outdir",
            str(out_dir),
            "--mode",
            "research",
        ],
    )
    assert result.exit_code == 0, result.output

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "run_summary.json").exists()
    assert (out_dir / "logs" / "run.log").exists()
    assert (out_dir / "reports" / "report.html").exists()
    assert (out_dir / "figures" / "metrics_overview.png").exists()
    assert (out_dir / "cards" / "experiment_card.md").exists()
    assert (out_dir / "dossier" / "dossier.md").exists()
    assert (out_dir / "dossier" / "appendices" / "qc.md").exists()
    assert (out_dir / "dossier" / "appendices" / "uncertainty.md").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("command") == "report.run"
    assert isinstance(manifest.get("inputs"), list)
    assert "python_version" in manifest
    assert "platform" in manifest
    assert "random_seeds" in manifest

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary.get("status") == "success"
