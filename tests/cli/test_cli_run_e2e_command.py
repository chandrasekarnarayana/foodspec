from __future__ import annotations

import json
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from foodspec.cli import app

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_oil_fixture.csv"


def test_run_e2e_command(tmp_path):
    assert FIXTURE_CSV.exists(), "Synthetic fixture missing"

    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        textwrap.dedent(
            """
            name: CLI Run E2E Test
            version: 0.0.1
            expected_columns:
              label_col: label
              group_col: stage
            steps:
              - type: preprocess
                params:
                  baseline_method: none
                  normalization: none
              - type: rq_analysis
                params:
                  model: logreg
                  validation:
                    scheme: loso
                    folds: 2
            features:
              type: peaks
              peaks:
                - name: peak_1000
                  center: 1000.0
                  window: 2.0
                - name: peak_1001
                  center: 1001.0
                  window: 2.0
                - name: peak_1002
                  center: 1002.0
                  window: 2.0
            task:
              type: classification
              target: label
            """
        ).strip()
    )

    run_dir = tmp_path / "run"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-e2e",
            "--csv",
            str(FIXTURE_CSV),
            "--protocol",
            str(protocol_path),
            "--outdir",
            str(run_dir),
            "--model",
            "logreg",
            "--features",
            "peaks",
            "--mode",
            "research",
            "--label-col",
            "label",
            "--group",
            "stage",
            "--seed",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "feature_info.json").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "trust" / "evaluation.json").exists()

    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary.get("status") == "success"
    assert summary.get("artifacts", {}).get("report")
