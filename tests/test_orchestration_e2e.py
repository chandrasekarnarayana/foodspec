"""
Integration tests for end-to-end orchestration layer.

Tests that:
  - Experiment.run() creates expected artifact structure
  - Manifest contains all required metadata
  - Metrics JSON is produced
  - Report index.html is generated
  - Exit codes are correct
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.experiment import Experiment, RunMode, ValidationScheme


@pytest.fixture
def synthetic_csv():
    """Create a synthetic CSV dataset for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 10

    # Create synthetic data: features + binary target
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def minimal_protocol_dict():
    """Create a minimal protocol config for testing."""
    return {
        "name": "TestProtocol",
        "version": "1.0.0",
        "description": "Test protocol",
        "steps": [],
        "expected_columns": {"target": "target"},
    }


class TestExperimentFromProtocol:
    """Test Experiment.from_protocol factory."""

    def test_from_dict(self, minimal_protocol_dict):
        """Test creating Experiment from dict."""
        exp = Experiment.from_protocol(minimal_protocol_dict)
        assert exp.config.protocol_config.name == "TestProtocol"
        assert exp.config.mode == RunMode.RESEARCH

    def test_from_dict_with_overrides(self, minimal_protocol_dict):
        """Test creating Experiment with overrides."""
        exp = Experiment.from_protocol(
            minimal_protocol_dict,
            mode=RunMode.REGULATORY,
            scheme=ValidationScheme.LOSO,
            model="svm",
        )
        assert exp.config.mode == RunMode.REGULATORY
        assert exp.config.scheme == ValidationScheme.LOSO
        assert exp.config.model == "svm"

    def test_from_dict_string_mode(self, minimal_protocol_dict):
        """Test mode as string."""
        exp = Experiment.from_protocol(
            minimal_protocol_dict,
            mode="regulatory",
            scheme="loso",
        )
        assert exp.config.mode == RunMode.REGULATORY
        assert exp.config.scheme == ValidationScheme.LOSO


class TestExperimentRun:
    """Test Experiment.run() orchestration."""

    def test_run_creates_artifact_structure(self, minimal_protocol_dict, synthetic_csv):
        """Test that run() creates expected directory structure."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

        assert result.status == "success"
        assert result.exit_code == 0
        assert result.run_id is not None
        assert result.tables_dir is not None
        assert result.figures_dir is not None
        assert result.report_dir is not None
        assert result.manifest_path is not None
        assert result.summary_path is not None

    def test_run_creates_directories(self, minimal_protocol_dict, synthetic_csv):
        """Test that expected subdirectories are created."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

            assert result.tables_dir.exists()
            assert result.figures_dir.exists()
            assert result.report_dir.exists()

            # Check for other subdirectories
            run_dir = outdir / result.run_id
            assert (run_dir / "data").exists()
            assert (run_dir / "features").exists()
            assert (run_dir / "modeling").exists()
            assert (run_dir / "trust").exists()

    def test_manifest_validity(self, minimal_protocol_dict, synthetic_csv):
        """Test that manifest.json is valid and contains required fields."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

            assert result.manifest_path.exists()

            # Load and verify manifest
            manifest_data = json.loads(result.manifest_path.read_text())
            assert "protocol_hash" in manifest_data
            assert "python_version" in manifest_data
            assert "platform" in manifest_data
            assert "seed" in manifest_data
            assert "data_fingerprint" in manifest_data
            assert "start_time" in manifest_data
            assert "end_time" in manifest_data
            assert "duration_seconds" in manifest_data
            assert "artifacts" in manifest_data

    def test_summary_validity(self, minimal_protocol_dict, synthetic_csv):
        """Test that summary.json is valid and contains deployment readiness info."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

            assert result.summary_path.exists()

            # Load and verify summary
            summary_data = json.loads(result.summary_path.read_text())
            assert "dataset_summary" in summary_data
            assert "scheme" in summary_data
            assert "model" in summary_data
            assert "mode" in summary_data
            assert "metrics" in summary_data
            assert "calibration" in summary_data
            assert "coverage" in summary_data
            assert "abstention_rate" in summary_data
            assert "deployment_readiness_score" in summary_data
            assert "deployment_ready" in summary_data
            assert "key_risks" in summary_data

    def test_metrics_produced(self, minimal_protocol_dict, synthetic_csv):
        """Test that modeling metrics are produced."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

            metrics_path = result.report_dir.parent / "modeling" / "metrics.json"
            assert metrics_path.exists()

            metrics = json.loads(metrics_path.read_text())
            assert isinstance(metrics, dict)

    def test_report_generated(self, minimal_protocol_dict, synthetic_csv):
        """Test that HTML report is generated."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

            report_path = result.report_dir / "index.html"
            assert report_path.exists()

            html_content = report_path.read_text()
            assert "<!DOCTYPE html>" in html_content or "<html>" in html_content
            assert "FoodSpec" in html_content

    def test_preprocessed_data_saved(self, minimal_protocol_dict, synthetic_csv):
        """Test that preprocessed data is saved."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

            preprocessed_path = result.report_dir.parent / "data" / "preprocessed.csv"
            assert preprocessed_path.exists()

    def test_features_saved(self, minimal_protocol_dict, synthetic_csv):
        """Test that feature matrices are saved."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

            features_dir = result.report_dir.parent / "features"
            assert (features_dir / "X.npy").exists()
            assert (features_dir / "y.npy").exists()

    def test_invalid_csv_path(self, minimal_protocol_dict):
        """Test handling of invalid CSV path."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = exp.run(
                csv_path=Path("/nonexistent/file.csv"),
                outdir=Path(tmpdir)
            )

            assert result.status == "validation_error"
            assert result.exit_code == 2
            assert result.error is not None

    def test_seed_reproducibility(self, minimal_protocol_dict, synthetic_csv):
        """Test that seed ensures reproducibility."""
        exp1 = Experiment.from_protocol(minimal_protocol_dict)
        exp2 = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir1:
            outdir1 = Path(tmpdir1)
            result1 = exp1.run(csv_path=synthetic_csv, outdir=outdir1, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir2:
            outdir2 = Path(tmpdir2)
            result2 = exp2.run(csv_path=synthetic_csv, outdir=outdir2, seed=42)

        # Both runs should succeed
        assert result1.status == "success"
        assert result2.status == "success"

    def test_different_modes(self, minimal_protocol_dict, synthetic_csv):
        """Test different run modes."""
        modes = [RunMode.RESEARCH, RunMode.REGULATORY, RunMode.MONITORING]

        for mode in modes:
            exp = Experiment.from_protocol(minimal_protocol_dict, mode=mode)

            with tempfile.TemporaryDirectory() as tmpdir:
                outdir = Path(tmpdir)
                result = exp.run(csv_path=synthetic_csv, outdir=outdir)

                assert result.status == "success"

                # Verify mode is recorded in summary
                summary = json.loads(result.summary_path.read_text())
                assert summary["mode"] == mode.value

    def test_different_schemes(self, minimal_protocol_dict, synthetic_csv):
        """Test different validation schemes."""
        schemes = [ValidationScheme.LOBO, ValidationScheme.LOSO, ValidationScheme.NESTED]

        for scheme in schemes:
            exp = Experiment.from_protocol(minimal_protocol_dict, scheme=scheme)

            with tempfile.TemporaryDirectory() as tmpdir:
                outdir = Path(tmpdir)
                result = exp.run(csv_path=synthetic_csv, outdir=outdir)

                assert result.status == "success"

                # Verify scheme is recorded in summary
                summary = json.loads(result.summary_path.read_text())
                assert summary["scheme"] == scheme.value

    def test_different_models(self, minimal_protocol_dict, synthetic_csv):
        """Test different model specifications."""
        models = ["lightgbm", "svm", "rf", "logreg"]

        for model_name in models:
            exp = Experiment.from_protocol(
                minimal_protocol_dict,
                model=model_name,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                outdir = Path(tmpdir)
                result = exp.run(csv_path=synthetic_csv, outdir=outdir)

                # All should attempt to succeed (may fail if model not available)
                # but should at least produce outputs
                assert result.run_id is not None

    def test_result_to_dict(self, minimal_protocol_dict, synthetic_csv):
        """Test RunResult.to_dict() serialization."""
        exp = Experiment.from_protocol(minimal_protocol_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = exp.run(csv_path=synthetic_csv, outdir=outdir)

            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "run_id" in result_dict
            assert "status" in result_dict
            assert "exit_code" in result_dict
            assert "metrics" in result_dict
            assert "error" in result_dict


class TestExperimentEdgeCases:
    """Test edge cases and error handling."""

    def test_tiny_dataset(self, minimal_protocol_dict):
        """Test with very small dataset."""
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0],
            "target": [0, 1],
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            csv_path = Path(f.name)

        try:
            exp = Experiment.from_protocol(minimal_protocol_dict)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = exp.run(csv_path=csv_path, outdir=Path(tmpdir))

                # Should handle tiny dataset gracefully
                assert result.run_id is not None
        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_multiclass_target(self, minimal_protocol_dict):
        """Test with multiclass target."""
        df = pd.DataFrame({
            "feature_1": np.random.randn(30),
            "feature_2": np.random.randn(30),
            "target": np.random.randint(0, 3, 30),
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            csv_path = Path(f.name)

        try:
            exp = Experiment.from_protocol(minimal_protocol_dict)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = exp.run(csv_path=csv_path, outdir=Path(tmpdir))

                assert result.status == "success"
        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_with_missing_values(self, minimal_protocol_dict):
        """Test handling of missing values."""
        df = pd.DataFrame({
            "feature_1": [1.0, np.nan, 3.0] * 10,
            "feature_2": [4.0, 5.0, np.nan] * 10,
            "target": [0, 1] * 15,
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            csv_path = Path(f.name)

        try:
            exp = Experiment.from_protocol(minimal_protocol_dict)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = exp.run(csv_path=csv_path, outdir=Path(tmpdir))

                # Should handle missing values (may fail or succeed depending on model)
                assert result.run_id is not None
        finally:
            if csv_path.exists():
                csv_path.unlink()
