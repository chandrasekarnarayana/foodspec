"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from typer.testing import CliRunner

from foodspec.cli.main import app
from foodspec.deploy import save_bundle

runner = CliRunner()


def test_cli_run_minimal_protocol(tmp_path: Path) -> None:
    """Test CLI run command with minimal protocol."""
    # Create dummy data file
    data_file = tmp_path / "data.csv"
    data_file.write_text("id,value,modality\n1,2,raman\n")
    
    # Create minimal protocol
    protocol_path = tmp_path / "protocol.yaml"
    protocol_dict = {
        "version": "2.0.0",
        "task": {"name": "classification", "objective": "max"},
        "data": {
            "input": str(data_file),
            "modality": "raman",
            "label": "value",
            "metadata_map": {"sample_id": "id", "modality": "modality", "label": "value"},
        },
        "preprocess": {"recipe": None, "steps": []},
        "qc": {"thresholds": {}, "metrics": []},
        "features": {"modules": [], "strategy": "auto"},
        "model": {"estimator": "logreg"},
        "validation": {"scheme": "train_test_split"},
        "uncertainty": {"conformal": {}},
        "interpretability": {"methods": []},
        "visualization": {"plots": []},
        "reporting": {"format": "markdown", "sections": []},
        "export": {"bundle": {}},
    }

    protocol_path.write_text(yaml.safe_dump(protocol_dict, sort_keys=False))

    outdir = tmp_path / "run_output"

    # Run CLI
    result = runner.invoke(
        app,
        ["run", "--protocol", str(protocol_path), "--outdir", str(outdir), "--seed", "42"],
    )

    # Check output
    assert result.exit_code == 0
    assert "FoodSpec v2.0 - Analysis Run" in result.stdout
    assert "Analysis completed successfully" in result.stdout
    assert str(outdir) in result.stdout

    # Check output files
    assert outdir.exists()
    assert (outdir / "manifest.json").exists()
    assert (outdir / "logs.txt").exists()


def test_cli_run_missing_protocol(tmp_path: Path) -> None:
    """Test CLI run command with missing protocol file."""
    protocol_path = tmp_path / "nonexistent.yaml"
    outdir = tmp_path / "run_output"

    result = runner.invoke(
        app,
        ["run", "--protocol", str(protocol_path), "--outdir", str(outdir)],
        mix_stderr=False,
    )

    # Should fail with exit code 2 (Typer validation error) or 1 (FileNotFoundError)
    assert result.exit_code in (1, 2)
    # Check that an error was reported (either in stdout or exception)
    if result.stdout:
        output = result.stdout.lower()
        assert "error" in output or "does not exist" in output


def test_cli_predict_basic(tmp_path: Path) -> None:
    """Test CLI predict command."""
    # Create bundle
    X_train = np.random.randn(30, 10)
    y_train = np.random.randint(0, 2, 30)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
        label_encoder={0: "class_a", 1: "class_b"},
    )

    # Create input CSV
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(3):
        for feature_idx in range(10):
            test_data.append(
                {"sample_id": f"S{sample_idx}", "wavenumber": feature_idx, "intensity": 1.0}
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    outdir = tmp_path / "predictions"

    # Run CLI
    result = runner.invoke(
        app,
        [
            "predict",
            "--bundle",
            str(bundle_dir),
            "--input",
            str(input_csv),
            "--outdir",
            str(outdir),
        ],
    )

    # Check output
    assert result.exit_code == 0
    assert "FoodSpec v2.0 - Prediction" in result.stdout
    assert "Predictions completed successfully" in result.stdout
    assert "Samples processed: 3" in result.stdout

    # Check output files
    assert (outdir / "predictions.csv").exists()
    assert (outdir / "probabilities.csv").exists()


def test_cli_predict_no_probabilities(tmp_path: Path) -> None:
    """Test CLI predict command without probabilities."""
    # Create bundle
    X_train = np.random.randn(20, 5)
    y_train = np.random.randint(0, 2, 20)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
    )

    # Create input CSV
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(2):
        for feature_idx in range(5):
            test_data.append(
                {"sample_id": f"S{sample_idx}", "wavenumber": feature_idx, "intensity": 1.0}
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    outdir = tmp_path / "predictions"

    # Run CLI with --no-probabilities flag
    result = runner.invoke(
        app,
        [
            "predict",
            "--bundle",
            str(bundle_dir),
            "--input",
            str(input_csv),
            "--outdir",
            str(outdir),
            "--no-probabilities",
        ],
    )

    # Check output
    assert result.exit_code == 0
    assert (outdir / "predictions.csv").exists()
    assert not (outdir / "probabilities.csv").exists()


def test_cli_predict_missing_bundle(tmp_path: Path) -> None:
    """Test CLI predict command with missing bundle."""
    bundle_dir = tmp_path / "nonexistent_bundle"
    input_csv = tmp_path / "input.csv"
    outdir = tmp_path / "predictions"

    # Create dummy input
    pd.DataFrame({"sample_id": ["S1"], "wavenumber": [0], "intensity": [1.0]}).to_csv(
        input_csv, index=False
    )

    result = runner.invoke(
        app,
        [
            "predict",
            "--bundle",
            str(bundle_dir),
            "--input",
            str(input_csv),
            "--outdir",
            str(outdir),
        ],
    )

    # Should fail
    assert result.exit_code == 2  # Typer validation error for non-existent path


def test_cli_report_generation(tmp_path: Path) -> None:
    """Test CLI report command."""
    # Create minimal run directory with manifest
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir()

    # Create manifest
    manifest = {
        "run_id": "test_run_20260124",
        "protocol_hash": "abc123",
        "seed": 42,
        "artifacts": {},
    }
    import json

    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    output_path = tmp_path / "custom_report.html"

    # Run CLI
    result = runner.invoke(
        app,
        ["report", "--run-dir", str(run_dir), "--output", str(output_path)],
    )

    # Check output
    assert result.exit_code == 0
    assert "FoodSpec v2.0 - Report Generation" in result.stdout
    assert "Report generated successfully" in result.stdout
    assert str(output_path) in result.stdout

    # Check output file
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cli_report_missing_manifest(tmp_path: Path) -> None:
    """Test CLI report command with missing manifest."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    result = runner.invoke(app, ["report", "--run-dir", str(run_dir)])

    # Should fail with exit code 1
    assert result.exit_code == 1


def test_cli_version_command() -> None:
    """Test CLI version command."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "FoodSpec v2.0.0" in result.stdout
    assert "Spectral analysis framework" in result.stdout


def test_cli_help() -> None:
    """Test CLI help output."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "FoodSpec v2.0" in result.stdout
    assert "run" in result.stdout
    assert "predict" in result.stdout
    assert "report" in result.stdout


def test_cli_run_help() -> None:
    """Test CLI run command help."""
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "protocol" in result.stdout.lower()
    assert "outdir" in result.stdout.lower()
    assert "seed" in result.stdout.lower()


def test_cli_predict_help() -> None:
    """Test CLI predict command help."""
    result = runner.invoke(app, ["predict", "--help"])

    assert result.exit_code == 0
    assert "bundle" in result.stdout.lower()
    assert "input" in result.stdout.lower()


def test_cli_report_help() -> None:
    """Test CLI report command help."""
    result = runner.invoke(app, ["report", "--help"])

    assert result.exit_code == 0
    assert "run-dir" in result.stdout.lower()
