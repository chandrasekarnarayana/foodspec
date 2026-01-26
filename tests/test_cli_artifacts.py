import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from foodspec.cli.main import app


def test_io_validate_artifacts(tmp_path):
    runner = CliRunner()
    sample_path = Path(__file__).resolve().parent / "data" / "sample_wide.csv"
    run_dir = tmp_path / "io_validate"
    result = runner.invoke(
        app,
        ["io", "validate", str(sample_path), "--run-dir", str(run_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "logs" / "run.log").exists()


def test_train_run_emits_cards(tmp_path):
    runner = CliRunner()
    protocol_path = tmp_path / "protocol.json"
    protocol_payload = {
        "name": "TestProtocol",
        "description": "Minimal protocol for test coverage.",
        "when_to_use": "Unit test coverage.",
        "steps": [
            {
                "type": "qc_checks",
                "params": {"required_columns": ["label"], "class_col": "label"},
            }
        ],
        "qc": {"required": False},
    }
    protocol_path.write_text(json.dumps(protocol_payload), encoding="utf-8")

    data_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "feature1": [1.0, 1.5, 2.0, 2.5],
            "label": ["a", "a", "b", "b"],
        }
    )
    df.to_csv(data_path, index=False)

    run_dir = tmp_path / "train_run"
    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "--protocol",
            str(protocol_path),
            "--input",
            str(data_path),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "logs" / "run.log").exists()
    assert (run_dir / "model_card.md").exists()
    assert (run_dir / "dataset_card.md").exists()


def test_train_run_qc_required_fails(tmp_path):
    runner = CliRunner()
    protocol_path = tmp_path / "protocol.json"
    protocol_payload = {
        "name": "QcRequiredProtocol",
        "description": "QC required protocol.",
        "steps": [
            {
                "type": "qc_checks",
                "params": {"required_columns": ["label"], "class_col": "label"},
            }
        ],
        "qc": {"required": True},
    }
    protocol_path.write_text(json.dumps(protocol_payload), encoding="utf-8")

    data_path = tmp_path / "data.csv"
    df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
    df.to_csv(data_path, index=False)

    run_dir = tmp_path / "train_qc_required"
    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "--protocol",
            str(protocol_path),
            "--input",
            str(data_path),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 3, result.output
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "qc_report.json").exists()
