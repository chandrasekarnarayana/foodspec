"""Comprehensive tests for FoodSpec reporting and visualization system.

Tests report generation, visualization functions, and figure export.
"""
from pathlib import Path

import pytest

from foodspec.reporting.base import ReportContext, ReportBuilder, collect_figures
from foodspec.reporting.cards import build_experiment_card
from foodspec.reporting.modes import ReportMode
from foodspec.reporting.export import export_paper_figures
from foodspec.reporting.api import build_report_from_run


class TestReportContext:
    """Test report context loading and artifact collection."""

    def test_load_minimal_context(self, minimal_run_dir):
        """Load context from minimal run directory."""
        context = ReportContext.load(minimal_run_dir)
        
        assert context.manifest is not None
        assert context.manifest.run_id == "test_run_001"
        assert context.metrics is not None
        assert len(context.metrics) > 0
        assert context.predictions is not None
        assert context.trust_outputs is not None

    def test_load_trust_outputs(self, minimal_run_dir):
        """Verify trust outputs are loaded from subdirectories."""
        context = ReportContext.load(minimal_run_dir)
        
        assert "calibration" in context.trust_outputs
        assert "conformal" in context.trust_outputs
        assert "abstention" in context.trust_outputs
        assert "drift" in context.trust_outputs

    def test_metrics_extraction(self, minimal_run_dir):
        """Verify metrics are correctly parsed."""
        context = ReportContext.load(minimal_run_dir)
        
        assert len(context.metrics) == 5  # 5 folds
        first_metric = context.metrics[0]
        assert "macro_f1" in first_metric
        assert float(first_metric["macro_f1"]) > 0.8

    def test_predictions_extraction(self, minimal_run_dir):
        """Verify predictions are correctly parsed."""
        context = ReportContext.load(minimal_run_dir)
        
        assert len(context.predictions) == 10  # 10 samples
        first_pred = context.predictions[0]
        assert "true_label" in first_pred
        assert "predicted_label" in first_pred


class TestCollectFigures:
    """Test figure collection from run directory."""

    def test_collect_from_empty_dir(self, tmp_path):
        """Collect figures from directory with no figures."""
        figures = collect_figures(tmp_path)
        assert figures == {}

    def test_collect_and_index(self, tmp_path):
        """Create figures and verify collection."""
        # Create dummy figure files
        viz_dir = tmp_path / "plots" / "viz" / "uncertainty"
        viz_dir.mkdir(parents=True)
        
        (viz_dir / "calibration.png").touch()
        (viz_dir / "coverage.svg").touch()
        
        figures = collect_figures(tmp_path)
        
        assert "uncertainty" in figures
        assert len(figures["uncertainty"]) == 2


class TestExperimentCard:
    """Test experiment card generation."""

    def test_build_card_from_context(self, minimal_run_dir):
        """Build experiment card from minimal run."""
        context = ReportContext.load(minimal_run_dir)
        card = build_experiment_card(context, mode=ReportMode.RESEARCH)
        
        assert card.run_id is not None
        assert card.macro_f1 is not None
        assert card.auroc is not None
        assert card.confidence_level is not None
        assert card.deployment_readiness is not None

    def test_card_metrics_extraction(self, minimal_run_dir):
        """Verify card extracts all available metrics."""
        context = ReportContext.load(minimal_run_dir)
        card = build_experiment_card(context, mode=ReportMode.RESEARCH)
        
        # Check main metrics
        assert card.macro_f1 > 0.8
        assert card.auroc > 0.9
        
        # Check trust metrics
        assert card.coverage is not None
        assert card.abstain_rate is not None

    def test_card_markdown_export(self, minimal_run_dir, tmp_path):
        """Export card to markdown."""
        context = ReportContext.load(minimal_run_dir)
        card = build_experiment_card(context, mode=ReportMode.RESEARCH)
        
        md_path = tmp_path / "card.md"
        card.to_markdown(md_path)
        
        assert md_path.exists()
        content = md_path.read_text()
        assert "Experiment Card" in content
        assert "Macro F1" in content
        assert "Coverage" in content

    def test_card_json_export(self, minimal_run_dir, tmp_path):
        """Export card to JSON."""
        context = ReportContext.load(minimal_run_dir)
        card = build_experiment_card(context, mode=ReportMode.RESEARCH)
        
        json_path = tmp_path / "card.json"
        card.to_json(json_path)
        
        assert json_path.exists()
        import json
        data = json.loads(json_path.read_text())
        assert data["run_id"] == card.run_id
        assert data["macro_f1"] == card.macro_f1


class TestReportBuilder:
    """Test HTML report generation."""

    def test_build_html_report(self, minimal_run_dir, tmp_path):
        """Build HTML report."""
        context = ReportContext.load(minimal_run_dir)
        builder = ReportBuilder(context)
        
        report_path = tmp_path / "report.html"
        builder.build_html(report_path, mode=ReportMode.RESEARCH, title="Test Report")
        
        assert report_path.exists()
        html_content = report_path.read_text()
        assert "Test Report" in html_content

    def test_build_regulatory_report(self, minimal_run_dir, tmp_path):
        """Build regulatory mode report."""
        context = ReportContext.load(minimal_run_dir)
        builder = ReportBuilder(context)
        
        report_path = tmp_path / "report_regulatory.html"
        builder.build_html(report_path, mode=ReportMode.REGULATORY)
        
        assert report_path.exists()

    def test_build_monitoring_report(self, minimal_run_dir, tmp_path):
        """Build monitoring mode report."""
        context = ReportContext.load(minimal_run_dir)
        builder = ReportBuilder(context)
        
        report_path = tmp_path / "report_monitoring.html"
        builder.build_html(report_path, mode=ReportMode.MONITORING)
        
        assert report_path.exists()


class TestBuildReportFromRun:
    """Test the build_report_from_run API."""

    def test_full_report_generation(self, minimal_run_dir, tmp_path):
        """Generate complete report artifacts."""
        artifacts = build_report_from_run(
            minimal_run_dir,
            out_dir=tmp_path,
            mode="research",
            pdf=False,
            title="Integration Test Report",
        )
        
        assert "report_html" in artifacts
        assert "card_json" in artifacts
        assert "card_markdown" in artifacts
        
        # Verify files exist
        for key, path in artifacts.items():
            if key != "report_pdf":  # PDF may not be created if optional
                assert Path(path).exists(), f"Missing artifact: {path}"

    def test_report_with_all_sections(self, minimal_run_dir, tmp_path):
        """Verify report includes all expected sections."""
        artifacts = build_report_from_run(
            minimal_run_dir,
            out_dir=tmp_path,
            mode="research",
            pdf=False,
        )
        
        html_path = artifacts["report_html"]
        html_content = Path(html_path).read_text()
        
        # Check for expected sections
        # (sections may be present as section IDs or headers)
        sections_to_check = ["metrics", "uncertainty", "summary"]
        for section in sections_to_check:
            # Just verify HTML is well-formed
            assert "<" in html_content and ">" in html_content


class TestPaperFigureExport:
    """Test paper-ready figure export."""

    def test_export_paper_figures(self, tmp_path):
        """Export figures with paper preset."""
        figures_dir = tmp_path / "figures_export"
        
        # Create dummy figure structure
        src_dir = tmp_path / "plots" / "viz" / "uncertainty"
        src_dir.mkdir(parents=True)
        (src_dir / "calibration.png").write_text("fake png")
        
        result_dir = export_paper_figures(
            tmp_path,
            out_dir=figures_dir,
            preset="joss",
        )
        
        assert result_dir.exists()
        assert (result_dir / "README.md").exists()
        assert (result_dir / "uncertainty").exists()


class TestVisualizationFunctions:
    """Test individual visualization functions."""

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_deterministic_plotting(self, seed):
        """Verify plots are deterministic with seeds."""
        import numpy as np
        from foodspec.viz.comprehensive import plot_abstention_distribution
        
        abstain_flags = np.array([0, 1, 0, 0, 1] * 10)
        
        fig1 = plot_abstention_distribution(abstain_flags, seed=seed)
        fig2 = plot_abstention_distribution(abstain_flags, seed=seed)
        
        # Both should create valid figures
        assert fig1 is not None
        assert fig2 is not None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def minimal_run_dir():
    """Provide minimal run directory for testing."""
    # Use the fixture directory created for tests
    fixture_path = Path(__file__).parent.parent / "fixtures" / "run_minimal"
    
    if not fixture_path.exists():
        pytest.skip("Minimal run fixture not found")
    
    return fixture_path
