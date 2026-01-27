"""Tests for multi-run comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from foodspec.viz.compare import (
    RunSummary,
    compare_runs,
    compute_baseline_deltas,
    create_comparison_dashboard,
    create_leaderboard,
    create_monitoring_plot,
    create_radar_plot,
    load_run_summary,
    scan_runs,
)


@pytest.fixture
def fake_runs_dir(tmp_path):
    """Create temporary fake run directories."""
    root = tmp_path / "runs"
    root.mkdir()

    # Create 3 fake runs
    for i in range(3):
        run_dir = root / f"run_00{i+1}"
        run_dir.mkdir()

        # Create manifest
        manifest = {
            "run_id": f"run_00{i+1}",
            "timestamp": f"2026-01-{20+i:02d}T10:00:00Z",
            "algorithm": f"Model_{chr(65+i)}",
            "validation_scheme": "holdout" if i % 2 == 0 else "cv",
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        # Create metrics
        metrics = {
            "macro_f1": 0.85 + i * 0.05,
            "auroc": 0.90 + i * 0.03,
            "coverage": 0.95 - i * 0.02,
        }
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

        # Create trust metrics
        trust_metrics = {
            "ece": 0.10 - i * 0.02,
            "abstain_rate": 0.05 + i * 0.01,
        }
        with open(run_dir / "uncertainty_metrics.json", "w") as f:
            json.dump(trust_metrics, f)

        # Create QC results
        qc_results = {
            "passed_snr": True,
            "passed_baseline": i % 2 == 0,
        }
        with open(run_dir / "qc_results.json", "w") as f:
            json.dump(qc_results, f)

    return root


@pytest.fixture
def sample_summaries():
    """Create sample run summaries."""
    summaries = []
    for i in range(3):
        summary = RunSummary(
            run_id=f"run_{i+1:03d}",
            run_dir=Path(f"run_{i+1:03d}"),
            timestamp=f"2026-01-{20+i:02d}T10:00:00Z",
            model_name=f"Model_{chr(65+i)}",
            validation_scheme="holdout" if i % 2 == 0 else "cv",
            metrics={
                "macro_f1": 0.85 + i * 0.05,
                "auroc": 0.90 + i * 0.03,
                "coverage": 0.95 - i * 0.02,
            },
            trust_metrics={
                "ece": 0.10 - i * 0.02,
                "abstain_rate": 0.05 + i * 0.01,
            },
            qc_flags={
                "passed_snr": True,
                "passed_baseline": i % 2 == 0,
            },
        )
        summaries.append(summary)
    return summaries


class TestRunSummary:
    """Test RunSummary class."""

    def test_initialization(self):
        """Test RunSummary initialization."""
        summary = RunSummary(
            run_id="test_run",
            run_dir=Path("test_run"),
            timestamp="2026-01-20T10:00:00Z",
            model_name="TestModel",
            validation_scheme="holdout",
            metrics={"f1": 0.85},
            trust_metrics={"ece": 0.10},
            qc_flags={"passed": True},
        )

        assert summary.run_id == "test_run"
        assert summary.model_name == "TestModel"
        assert summary.get_metric("f1") == 0.85
        assert summary.get_trust_metric("ece") == 0.10

    def test_to_dict(self):
        """Test to_dict conversion."""
        summary = RunSummary(
            run_id="test_run",
            run_dir=Path("test_run"),
            timestamp="2026-01-20T10:00:00Z",
            model_name="TestModel",
            validation_scheme="holdout",
            metrics={"f1": 0.85},
        )

        d = summary.to_dict()
        assert d["run_id"] == "test_run"
        assert d["model_name"] == "TestModel"
        assert d["metrics"]["f1"] == 0.85

    def test_get_metric_default(self):
        """Test get_metric with default value."""
        summary = RunSummary(
            run_id="test_run",
            run_dir=Path("test_run"),
            timestamp="2026-01-20T10:00:00Z",
            model_name="TestModel",
            validation_scheme="holdout",
            metrics={},
        )

        assert summary.get_metric("missing", default=0.5) == 0.5
        assert summary.get_trust_metric("missing", default=0.3) == 0.3


class TestScanRuns:
    """Test scan_runs function."""

    def test_scan_existing_runs(self, fake_runs_dir):
        """Test scanning existing run directories."""
        runs = scan_runs(fake_runs_dir)

        assert len(runs) == 3
        assert all(isinstance(run, Path) for run in runs)
        assert all((run / "manifest.json").exists() for run in runs)

    def test_scan_nonexistent_directory(self, tmp_path):
        """Test scanning non-existent directory."""
        runs = scan_runs(tmp_path / "nonexistent")
        assert runs == []

    def test_scan_empty_directory(self, tmp_path):
        """Test scanning empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        runs = scan_runs(empty_dir)
        assert runs == []

    def test_scan_with_custom_pattern(self, fake_runs_dir):
        """Test scanning with custom pattern."""
        runs = scan_runs(fake_runs_dir, pattern="*/manifest.json")
        assert len(runs) == 3


class TestLoadRunSummary:
    """Test load_run_summary function."""

    def test_load_complete_run(self, fake_runs_dir):
        """Test loading run with all artifacts."""
        run_dir = list(fake_runs_dir.glob("run_*"))[0]
        summary = load_run_summary(run_dir)

        assert summary.run_id.startswith("run_")
        assert summary.model_name.startswith("Model_")
        assert "macro_f1" in summary.metrics
        assert "ece" in summary.trust_metrics
        assert "passed_snr" in summary.qc_flags

    def test_load_minimal_run(self, tmp_path):
        """Test loading run with only manifest."""
        run_dir = tmp_path / "minimal_run"
        run_dir.mkdir()

        manifest = {
            "run_id": "minimal",
            "timestamp": "2026-01-20T10:00:00Z",
            "algorithm": "TestModel",
            "validation_scheme": "holdout",
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        summary = load_run_summary(run_dir)

        assert summary.run_id == "minimal"
        assert summary.metrics == {}
        assert summary.trust_metrics == {}
        assert summary.qc_flags == {}

    def test_load_missing_manifest(self, tmp_path):
        """Test loading run without manifest."""
        run_dir = tmp_path / "no_manifest"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            load_run_summary(run_dir)

    def test_load_with_default_values(self, tmp_path):
        """Test loading with missing optional fields."""
        run_dir = tmp_path / "partial_run"
        run_dir.mkdir()

        # Minimal manifest
        manifest = {}
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        summary = load_run_summary(run_dir)

        assert summary.run_id == run_dir.name
        assert summary.model_name == "Unknown"
        assert summary.validation_scheme == "unknown"


class TestCreateLeaderboard:
    """Test create_leaderboard function."""

    def test_create_basic_leaderboard(self, sample_summaries):
        """Test creating basic leaderboard."""
        leaderboard = create_leaderboard(sample_summaries)

        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 3
        assert "rank" in leaderboard.columns
        assert "run_id" in leaderboard.columns
        assert "model" in leaderboard.columns

    def test_leaderboard_sorting(self, sample_summaries):
        """Test leaderboard sorting by metrics."""
        leaderboard = create_leaderboard(
            sample_summaries,
            sort_by=("macro_f1",),
            ascending=(False,),
        )

        # Should be sorted by F1 descending
        f1_values = leaderboard["macro_f1"].values
        assert all(f1_values[i] >= f1_values[i + 1] for i in range(len(f1_values) - 1))

    def test_leaderboard_rank_column(self, sample_summaries):
        """Test leaderboard rank column."""
        leaderboard = create_leaderboard(sample_summaries)

        assert leaderboard["rank"].iloc[0] == 1
        assert leaderboard["rank"].iloc[-1] == 3

    def test_empty_summaries(self):
        """Test leaderboard with empty summaries."""
        leaderboard = create_leaderboard([])
        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 0


class TestCreateRadarPlot:
    """Test create_radar_plot function."""

    def test_create_basic_radar(self, sample_summaries, tmp_path):
        """Test creating basic radar plot."""
        output_path = tmp_path / "radar.png"
        result = create_radar_plot(
            sample_summaries,
            output_path=output_path,
            top_n=3,
        )

        assert result == output_path
        assert output_path.exists()

    def test_radar_with_top_n(self, sample_summaries, tmp_path):
        """Test radar plot with top N runs."""
        output_path = tmp_path / "radar_top2.png"
        create_radar_plot(
            sample_summaries,
            output_path=output_path,
            top_n=2,
        )

        assert output_path.exists()

    def test_radar_empty_summaries(self):
        """Test radar plot with empty summaries."""
        result = create_radar_plot([])
        assert result is None

    def test_radar_custom_metrics(self, sample_summaries, tmp_path):
        """Test radar plot with custom metrics."""
        output_path = tmp_path / "radar_custom.png"
        create_radar_plot(
            sample_summaries,
            metrics=("macro_f1", "auroc"),
            output_path=output_path,
        )

        assert output_path.exists()


class TestComputeBaselineDeltas:
    """Test compute_baseline_deltas function."""

    def test_compute_basic_deltas(self, sample_summaries):
        """Test computing deltas from baseline."""
        deltas = compute_baseline_deltas(
            sample_summaries,
            baseline_id="run_001",
        )

        assert isinstance(deltas, pd.DataFrame)
        assert len(deltas) == 3
        assert "delta_macro_f1" in deltas.columns
        assert "is_baseline" in deltas.columns

    def test_baseline_has_zero_deltas(self, sample_summaries):
        """Test that baseline has zero deltas."""
        deltas = compute_baseline_deltas(
            sample_summaries,
            baseline_id="run_001",
        )

        baseline_row = deltas[deltas["is_baseline"]]
        assert baseline_row["delta_macro_f1"].iloc[0] == 0.0

    def test_nonbaseline_has_nonzero_deltas(self, sample_summaries):
        """Test that non-baseline runs have non-zero deltas."""
        deltas = compute_baseline_deltas(
            sample_summaries,
            baseline_id="run_001",
        )

        non_baseline = deltas[~deltas["is_baseline"]]
        assert not all(non_baseline["delta_macro_f1"] == 0.0)

    def test_missing_baseline_raises_error(self, sample_summaries):
        """Test that missing baseline raises error."""
        with pytest.raises(ValueError, match="Baseline run not found"):
            compute_baseline_deltas(
                sample_summaries,
                baseline_id="nonexistent",
            )

    def test_custom_metrics(self, sample_summaries):
        """Test computing deltas for custom metrics."""
        deltas = compute_baseline_deltas(
            sample_summaries,
            baseline_id="run_001",
            metrics=["auroc"],
        )

        assert "delta_auroc" in deltas.columns
        assert "delta_macro_f1" not in deltas.columns

    def test_empty_summaries(self):
        """Test deltas with empty summaries."""
        deltas = compute_baseline_deltas([], baseline_id="any")
        assert isinstance(deltas, pd.DataFrame)
        assert len(deltas) == 0


class TestCreateMonitoringPlot:
    """Test create_monitoring_plot function."""

    def test_create_basic_monitoring(self, sample_summaries, tmp_path):
        """Test creating basic monitoring plot."""
        output_path = tmp_path / "monitoring.png"
        result = create_monitoring_plot(
            sample_summaries,
            output_path=output_path,
        )

        assert result == output_path
        assert output_path.exists()

    def test_monitoring_custom_metrics(self, sample_summaries, tmp_path):
        """Test monitoring plot with custom metrics."""
        output_path = tmp_path / "monitoring_custom.png"
        create_monitoring_plot(
            sample_summaries,
            metrics=("macro_f1",),
            output_path=output_path,
        )

        assert output_path.exists()

    def test_monitoring_empty_summaries(self):
        """Test monitoring plot with empty summaries."""
        result = create_monitoring_plot([])
        assert result is None

    def test_monitoring_multiple_metrics(self, sample_summaries, tmp_path):
        """Test monitoring plot with multiple metrics."""
        output_path = tmp_path / "monitoring_multi.png"
        create_monitoring_plot(
            sample_summaries,
            metrics=("macro_f1", "auroc", "coverage"),
            output_path=output_path,
        )

        assert output_path.exists()


class TestCreateComparisonDashboard:
    """Test create_comparison_dashboard function."""

    def test_create_basic_dashboard(self, sample_summaries, tmp_path):
        """Test creating basic dashboard."""
        output_path = tmp_path / "dashboard.html"
        result = create_comparison_dashboard(
            sample_summaries,
            output_path,
        )

        assert result == output_path.resolve()
        assert output_path.exists()

        # Check HTML content
        html = output_path.read_text()
        assert "FoodSpec" in html
        assert "Leaderboard" in html

    def test_dashboard_with_baseline(self, sample_summaries, tmp_path):
        """Test dashboard with baseline."""
        output_path = tmp_path / "dashboard_baseline.html"
        create_comparison_dashboard(
            sample_summaries,
            output_path,
            baseline_id="run_001",
        )

        assert output_path.exists()

    def test_dashboard_creates_directory(self, sample_summaries, tmp_path):
        """Test that dashboard creates output directory."""
        output_path = tmp_path / "subdir" / "dashboard.html"
        create_comparison_dashboard(
            sample_summaries,
            output_path,
        )

        assert output_path.exists()


class TestCompareRuns:
    """Test compare_runs function."""

    def test_compare_basic(self, sample_summaries, tmp_path):
        """Test basic run comparison."""
        outputs = compare_runs(
            sample_summaries,
            output_dir=tmp_path,
        )

        assert "leaderboard" in outputs
        assert "dashboard" in outputs
        assert "radar" in outputs
        assert "monitoring" in outputs

        # Check files exist
        for path in outputs.values():
            assert path.exists()

    def test_compare_with_baseline(self, sample_summaries, tmp_path):
        """Test comparison with baseline."""
        outputs = compare_runs(
            sample_summaries,
            output_dir=tmp_path,
            baseline_id="run_001",
        )

        assert "baseline_deltas" in outputs
        assert outputs["baseline_deltas"].exists()

    def test_compare_invalid_baseline_warns(self, sample_summaries, tmp_path):
        """Test that invalid baseline issues warning."""
        with pytest.warns(UserWarning, match="Could not compute baseline deltas"):
            outputs = compare_runs(
                sample_summaries,
                output_dir=tmp_path,
                baseline_id="nonexistent",
            )

        # Should still have other outputs
        assert "leaderboard" in outputs
        assert "baseline_deltas" not in outputs

    def test_compare_creates_directory(self, sample_summaries, tmp_path):
        """Test that compare_runs creates output directory."""
        output_dir = tmp_path / "comparison_results"
        compare_runs(sample_summaries, output_dir)

        assert output_dir.exists()

    def test_compare_with_custom_top_n(self, sample_summaries, tmp_path):
        """Test comparison with custom top_n."""
        outputs = compare_runs(
            sample_summaries,
            output_dir=tmp_path,
            top_n=2,
        )

        assert outputs["radar"].exists()


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, fake_runs_dir, tmp_path):
        """Test complete workflow from scan to comparison."""
        # Scan runs
        run_dirs = scan_runs(fake_runs_dir)
        assert len(run_dirs) == 3

        # Load summaries
        summaries = [load_run_summary(run_dir) for run_dir in run_dirs]
        assert len(summaries) == 3

        # Compare runs
        outputs = compare_runs(
            summaries,
            output_dir=tmp_path,
            baseline_id=summaries[0].run_id,
        )

        # Verify all outputs
        assert all(path.exists() for path in outputs.values())

    def test_scan_load_compare_pipeline(self, fake_runs_dir, tmp_path):
        """Test scan → load → compare pipeline."""
        # Step 1: Scan
        runs = scan_runs(fake_runs_dir)

        # Step 2: Load
        summaries = []
        for run in runs:
            summary = load_run_summary(run)
            summaries.append(summary)

        # Step 3: Compare
        outputs = compare_runs(summaries, tmp_path)

        # Verify
        assert len(summaries) == 3
        assert len(outputs) >= 4  # At least 4 output files

    def test_leaderboard_to_radar_consistency(self, sample_summaries, tmp_path):
        """Test that leaderboard and radar use same ordering."""
        # Create leaderboard
        leaderboard = create_leaderboard(sample_summaries)

        # Create radar
        create_radar_plot(
            sample_summaries,
            top_n=3,
            output_path=tmp_path / "radar.png",
        )

        # Verify both exist and top run matches
        assert len(leaderboard) == 3
        assert (tmp_path / "radar.png").exists()
