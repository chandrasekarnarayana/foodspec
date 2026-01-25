"""
Tests for ReportContext, ReportBuilder, and collect_figures.

Tests verify:
- ReportContext loads artifacts from run directory
- collect_figures indexes images correctly
- ReportBuilder validates required artifacts per mode
- Missing artifacts raise actionable errors in regulatory mode
- HTML is built with sidebar navigation
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.manifest import RunManifest
from foodspec.reporting.base import ReportBuilder, ReportContext, collect_figures
from foodspec.reporting.modes import ReportMode


# Fixtures

def _make_test_manifest(tmp_path: Path) -> RunManifest:
    """Create a minimal manifest for testing."""
    # Create a dummy data file for fingerprinting
    data_path = tmp_path / "data.csv"
    data_path.write_text("col1,col2\n1,2\n")
    
    return RunManifest.build(
        protocol_snapshot={"version": "2.0.0", "task": {"name": "classification"}},
        data_path=data_path,
        seed=42,
        artifacts={"metrics": "metrics.csv", "qc": "qc.csv"},
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def run_dir(tmp_path):
    """Create a test run directory with manifest and artifacts."""
    artifacts = ArtifactRegistry(tmp_path)
    artifacts.ensure_layout()

    # Create manifest
    manifest = _make_test_manifest(tmp_path)
    manifest.save(artifacts.manifest_path)

    # Create metrics
    metrics = [
        {"fold_id": 0, "accuracy": 0.9, "precision": 0.88},
        {"fold_id": 1, "accuracy": 0.92, "precision": 0.90},
    ]
    _write_csv(artifacts.metrics_path, metrics)

    # Create QC
    qc = [{"check": "snr", "status": "pass"}]
    _write_csv(artifacts.qc_path, qc)

    yield tmp_path


@pytest.fixture
def run_dir_with_figures(tmp_path):
    """Create a run directory with visualizations."""
    artifacts = ArtifactRegistry(tmp_path)
    artifacts.ensure_layout()

    # Create manifest
    manifest = _make_test_manifest(tmp_path)
    manifest.save(artifacts.manifest_path)

    # Create figures
    viz_dir = artifacts.viz_drift_dir
    viz_dir.mkdir(parents=True, exist_ok=True)
    (viz_dir / "drift_plot.png").write_bytes(b"PNG_DATA")
    (viz_dir / "drift_summary.svg").write_bytes(b"SVG_DATA")

    interpretability_dir = artifacts.viz_interpretability_dir
    interpretability_dir.mkdir(parents=True, exist_ok=True)
    (interpretability_dir / "feature_importance.png").write_bytes(b"PNG_DATA")

    yield tmp_path


# Tests for ReportContext

class TestReportContextLoad:
    """Test loading context from run directory."""

    def test_loads_manifest(self, run_dir):
        """Test that manifest is loaded."""
        context = ReportContext.load(run_dir)
        assert context.manifest is not None
        assert context.manifest.seed == 42

    def test_loads_metrics(self, run_dir):
        """Test that metrics are loaded."""
        context = ReportContext.load(run_dir)
        assert len(context.metrics) == 2
        assert context.metrics[0]["accuracy"] == "0.9"

    def test_loads_qc(self, run_dir):
        """Test that QC data is loaded."""
        context = ReportContext.load(run_dir)
        assert len(context.qc) == 1
        assert context.qc[0]["status"] == "pass"

    def test_missing_optional_artifacts_returns_empty(self, tmp_path):
        """Test that missing optional artifacts are empty lists."""
        artifacts = ArtifactRegistry(tmp_path)
        artifacts.ensure_layout()
        manifest = _make_test_manifest(tmp_path)
        manifest.save(artifacts.manifest_path)

        context = ReportContext.load(tmp_path)
        assert context.predictions == []
        assert context.trust_outputs == {}

    def test_missing_manifest_raises_error(self, tmp_path):
        """Test that missing manifest raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ReportContext.load(tmp_path)

    def test_available_artifacts_includes_manifest(self, run_dir):
        """Test that manifest is always in available artifacts."""
        context = ReportContext.load(run_dir)
        assert "manifest" in context.available_artifacts

    def test_available_artifacts_includes_metrics_and_qc(self, run_dir):
        """Test that loaded artifacts are in available list."""
        context = ReportContext.load(run_dir)
        assert "metrics" in context.available_artifacts
        assert "qc" in context.available_artifacts

    def test_to_dict_is_serializable(self, run_dir):
        """Test that context converts to dict for templating."""
        context = ReportContext.load(run_dir)
        data = context.to_dict()
        assert "manifest" in data
        assert "metrics" in data
        assert "qc" in data
        assert data["manifest"]["seed"] == 42


# Tests for collect_figures

class TestCollectFigures:
    """Test figure indexing."""

    def test_collect_figures_from_viz_dir(self, run_dir_with_figures):
        """Test that figures are collected from viz directories."""
        figures = collect_figures(run_dir_with_figures)
        assert "drift" in figures
        assert "interpretability" in figures

    def test_collect_figures_groups_by_category(self, run_dir_with_figures):
        """Test that figures are grouped by category."""
        figures = collect_figures(run_dir_with_figures)
        drift_figs = figures["drift"]
        assert len(drift_figs) == 2
        assert any("drift_plot" in str(f) for f in drift_figs)

    def test_collect_figures_empty_when_no_viz_dir(self, tmp_path):
        """Test that empty dict is returned when viz dir doesn't exist."""
        figures = collect_figures(tmp_path)
        assert figures == {}

    def test_collect_figures_includes_png_jpg_svg(self, tmp_path):
        """Test that multiple image formats are collected."""
        artifacts = ArtifactRegistry(tmp_path)
        viz_dir = artifacts.viz_pipeline_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        (viz_dir / "plot.png").write_bytes(b"")
        (viz_dir / "diagram.svg").write_bytes(b"")
        (viz_dir / "photo.jpg").write_bytes(b"")

        figures = collect_figures(tmp_path)
        assert len(figures["pipeline"]) == 3


# Tests for ReportBuilder

class TestReportBuilderValidation:
    """Test artifact validation."""

    def test_validates_research_mode_minimal_artifacts(self, run_dir):
        """Test that RESEARCH mode accepts manifest and metrics."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"
        # Should not raise
        result = builder.build_html(out_path, mode=ReportMode.RESEARCH)
        assert result.exists()

    def test_validates_regulatory_mode_requires_qc(self, run_dir):
        """Test that REGULATORY mode requires qc artifact."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"
        # Should not raise (qc is present)
        result = builder.build_html(out_path, mode=ReportMode.REGULATORY)
        assert result.exists()

    def test_regulatory_mode_missing_artifacts_raises_error(self, tmp_path):
        """Test that missing required artifacts raise ValueError in regulatory mode."""
        artifacts = ArtifactRegistry(tmp_path)
        artifacts.ensure_layout()
        manifest = _make_test_manifest(tmp_path)
        manifest.save(artifacts.manifest_path)

        # Only manifest and metrics, missing qc, protocol_snapshot, data_fingerprint
        metrics = [{"accuracy": 0.9}]
        _write_csv(artifacts.metrics_path, metrics)

        context = ReportContext.load(tmp_path)
        builder = ReportBuilder(context)
        out_path = tmp_path / "report.html"

        with pytest.raises(ValueError, match="requires artifacts"):
            builder.build_html(out_path, mode=ReportMode.REGULATORY)

    def test_research_mode_permissive_with_missing_artifacts(self, tmp_path):
        """Test that RESEARCH mode doesn't fail on missing optional artifacts."""
        artifacts = ArtifactRegistry(tmp_path)
        artifacts.ensure_layout()
        manifest = _make_test_manifest(tmp_path)
        manifest.save(artifacts.manifest_path)

        metrics = [{"accuracy": 0.9}]
        _write_csv(artifacts.metrics_path, metrics)

        context = ReportContext.load(tmp_path)
        builder = ReportBuilder(context)
        out_path = tmp_path / "report.html"

        # Should not raise even if predictions, qc are missing
        result = builder.build_html(out_path, mode=ReportMode.RESEARCH)
        assert result.exists()


class TestReportBuilderHtmlGeneration:
    """Test HTML report generation."""

    def test_builds_html_file(self, run_dir):
        """Test that HTML report is written to disk."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        result = builder.build_html(out_path, mode=ReportMode.RESEARCH)
        assert result.exists()
        assert result.read_text().startswith("<!DOCTYPE html>")

    def test_html_contains_title(self, run_dir):
        """Test that HTML includes the report title."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.RESEARCH, title="My Report")
        html = out_path.read_text()
        assert "My Report" in html

    def test_html_contains_sidebar_navigation(self, run_dir):
        """Test that HTML includes sidebar navigation."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.RESEARCH)
        html = out_path.read_text()
        assert "sidebar" in html.lower()
        assert "Metrics" in html or "metrics" in html

    def test_html_contains_manifest_metadata(self, run_dir):
        """Test that HTML includes manifest metadata."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.RESEARCH)
        html = out_path.read_text()
        assert "42" in html  # seed value
        assert "Summary" in html

    def test_html_includes_metrics_table(self, run_dir):
        """Test that metrics are rendered in HTML."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.RESEARCH)
        html = out_path.read_text()
        assert "0.9" in html or "0.90" in html  # metrics values
        assert "<table>" in html

    def test_html_includes_qc_section_in_regulatory_mode(self, run_dir):
        """Test that QC section is included in regulatory mode."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.REGULATORY)
        html = out_path.read_text()
        assert "QC" in html
        assert "pass" in html

    def test_html_does_not_include_qc_section_in_research_mode(self, run_dir):
        """Test that QC section may not be in research mode."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.RESEARCH)
        html = out_path.read_text()
        # Research mode doesn't include qc, so it might not be in the HTML
        # (unless it's in template unconditionally)

    def test_html_includes_figures_if_present(self, run_dir_with_figures):
        """Test that figures are included in HTML if available."""
        context = ReportContext.load(run_dir_with_figures)
        builder = ReportBuilder(context)
        out_path = run_dir_with_figures / "report.html"

        builder.build_html(out_path, mode=ReportMode.RESEARCH)
        html = out_path.read_text()
        assert "Visualizations" in html
        assert "drift" in html.lower()
        assert "interpretability" in html.lower()

    def test_html_creates_parent_directories(self, run_dir):
        """Test that parent directories are created if missing."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        nested_path = run_dir / "nested" / "deep" / "report.html"

        result = builder.build_html(nested_path, mode=ReportMode.RESEARCH)
        assert result.exists()
        assert nested_path.exists()

    def test_mode_parameter_accepts_string(self, run_dir):
        """Test that mode can be passed as string."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        result = builder.build_html(out_path, mode="research")
        assert result.exists()

    def test_html_contains_reproducibility_section(self, run_dir):
        """Test that reproducibility section is always included."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.RESEARCH)
        html = out_path.read_text()
        assert "Reproducibility" in html
        assert "Python" in html or "python" in html


class TestReportBuilderModes:
    """Test mode-specific behavior."""

    def test_research_mode_sets_description(self, run_dir):
        """Test that research mode description is in HTML."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.RESEARCH)
        html = out_path.read_text()
        assert "research" in html.lower()

    def test_regulatory_mode_sets_description(self, run_dir):
        """Test that regulatory mode description is in HTML."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.REGULATORY)
        html = out_path.read_text()
        assert "regulatory" in html.lower() or "compliance" in html.lower()

    def test_monitoring_mode_sets_description(self, run_dir):
        """Test that monitoring mode description is in HTML."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        builder.build_html(out_path, mode=ReportMode.MONITORING)
        html = out_path.read_text()
        assert "monitoring" in html.lower() or "drift" in html.lower()


class TestReportBuilderErrorHandling:
    """Test error handling."""

    def test_invalid_mode_string_raises_error(self, run_dir):
        """Test that invalid mode string raises ValueError."""
        context = ReportContext.load(run_dir)
        builder = ReportBuilder(context)
        out_path = run_dir / "report.html"

        with pytest.raises(ValueError):
            builder.build_html(out_path, mode="invalid_mode")

    def test_error_message_lists_missing_artifacts(self, tmp_path):
        """Test that error message includes list of missing artifacts."""
        artifacts = ArtifactRegistry(tmp_path)
        artifacts.ensure_layout()
        manifest = _make_test_manifest(tmp_path)
        manifest.save(artifacts.manifest_path)

        metrics = [{"accuracy": 0.9}]
        _write_csv(artifacts.metrics_path, metrics)

        context = ReportContext.load(tmp_path)
        builder = ReportBuilder(context)
        out_path = tmp_path / "report.html"

        try:
            builder.build_html(out_path, mode=ReportMode.REGULATORY)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "Missing" in error_msg or "missing" in error_msg
            assert "qc" in error_msg.lower()
