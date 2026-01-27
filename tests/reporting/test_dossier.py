"""Tests for scientific dossier generator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from foodspec.reporting.dossier import DossierBuilder


class TestDossierBuilder:
    """Test scientific dossier generation."""

    @pytest.fixture
    def temp_run_dir(self, tmp_path: Path) -> Path:
        """Create a temporary run directory with artifacts."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir(parents=True)

        # Create manifest
        manifest_data = {
            "run_id": "test_run_001",
            "execution_timestamp": "2026-01-25T12:00:00",
            "protocol_name": "Test Protocol",
            "protocol_version": "1.0.0",
            "sample_count": 100,
            "data_source": "Test Dataset",
            "model_type": "random_forest",
            "cv_scheme": "stratified_k_fold",
            "foodspec_version": "1.0.0",
            "python_version": "3.12.9",
            "random_seed": 42,
            "cv_seed": 123,
            "data_hash": "abc123def456",
            "config_hash": "xyz789uvw456",
            "command_line": "foodspec analyze --config test.yaml",
            "platform": "linux",
            "hostname": "test-machine",
        }
        manifest_path = run_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        # Create protocol snapshot
        protocol_snapshot = {
            "steps": [
                {
                    "name": "Preprocessing",
                    "type": "baseline_correction",
                    "description": "Applied baseline correction",
                    "parameters": {"method": "als", "lam": 100},
                },
                {
                    "name": "Normalization",
                    "type": "normalization",
                    "description": "Applied normalization",
                    "parameters": {"method": "minmax"},
                },
            ]
        }
        snapshot_path = run_dir / "protocol_snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump(protocol_snapshot, f)

        # Create metrics
        metrics_data = {
            "summary": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1": 0.95,
            },
            "fold_metrics": [
                {"accuracy": 0.94, "precision": 0.93, "recall": 0.95, "f1": 0.94},
                {"accuracy": 0.95, "precision": 0.95, "recall": 0.96, "f1": 0.95},
                {"accuracy": 0.96, "precision": 0.96, "recall": 0.97, "f1": 0.96},
            ],
        }
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f)

        # Create QC results
        qc_results = {
            "summary": {"total_checks": 5, "passed": 4, "failed": 0, "warnings": 1},
            "checks": [
                {"name": "Data Quality", "status": "passed", "details": "OK"},
                {"name": "Outliers", "status": "passed", "details": "2 removed"},
                {"name": "Missing Values", "status": "passed", "details": "None"},
                {"name": "Drift", "status": "warning", "details": "Minor drift detected"},
                {"name": "Replicates", "status": "passed", "details": "Consistent"},
            ],
            "drift": {"detected": True, "drift_metrics": {"ks_statistic": 0.15}},
            "failures": [{"reason": "Outlier removal", "count": 2}],
        }
        qc_path = run_dir / "qc_results.json"
        with open(qc_path, "w") as f:
            json.dump(qc_results, f)

        # Create uncertainty metrics
        uncertainty_data = {
            "reliability": {"calibration_error": 0.05, "sharpness": 0.88},
            "conformal": {
                "coverage": 0.95,
                "average_set_size": 1.15,
                "coverage_by_size": {"1": 0.88, "2": 0.99},
            },
            "abstention": {
                "rate": 0.08,
                "accuracy_when_predicting": 0.97,
                "coverage_when_predicting": 0.99,
            },
        }
        uncertainty_path = run_dir / "uncertainty_metrics.json"
        with open(uncertainty_path, "w") as f:
            json.dump(uncertainty_data, f)

        return run_dir

    @pytest.fixture
    def temp_output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory."""
        output_dir = tmp_path / "dossier"
        return output_dir

    def test_build_creates_all_files(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that build() creates all required files."""
        builder = DossierBuilder()
        result = builder.build(
            run_dir=temp_run_dir, out_dir=temp_output_dir, mode="regulatory"
        )

        # Check all files exist
        assert (temp_output_dir / "methods.md").exists()
        assert (temp_output_dir / "results.md").exists()
        assert (temp_output_dir / "appendix_qc.md").exists()
        assert (temp_output_dir / "appendix_uncertainty.md").exists()
        assert (temp_output_dir / "appendix_reproducibility.md").exists()
        assert (temp_output_dir / "dossier_index.html").exists()

        # Check return value
        assert result == temp_output_dir / "dossier_index.html"

    def test_methods_md_contains_required_sections(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that methods.md has required sections."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        methods_file = temp_output_dir / "methods.md"
        content = methods_file.read_text()

        # Check required headings
        assert "# Methods" in content
        assert "## Protocol Specification" in content
        assert "## Data Source" in content
        assert "## Processing Pipeline" in content
        assert "## Model Configuration" in content

        # Check protocol-derived content (no hallucinations)
        assert "Test Protocol" in content
        assert "1.0.0" in content  # Version from protocol
        assert "100" in content  # Sample count
        assert "Preprocessing" in content  # From protocol snapshot
        assert "Normalization" in content  # From protocol snapshot

    def test_results_md_contains_required_sections(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that results.md has required sections."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        results_file = temp_output_dir / "results.md"
        content = results_file.read_text()

        # Check required headings
        assert "# Results" in content
        assert "## Summary Performance" in content
        assert "## Cross-Validation Stability" in content
        assert "## Key Findings" in content

        # Check metrics are included
        assert "0.95" in content  # Accuracy
        assert "| Fold |" in content  # Fold table
        assert "F1" in content

    def test_results_md_fold_table(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that results.md includes fold stability table."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        results_file = temp_output_dir / "results.md"
        content = results_file.read_text()

        # Check table structure
        assert "|" in content  # Table characters
        assert "| 1 |" in content  # Fold 1
        assert "| 2 |" in content  # Fold 2
        assert "| 3 |" in content  # Fold 3

    def test_appendix_qc_contains_required_sections(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that appendix_qc.md has required sections."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        qc_file = temp_output_dir / "appendix_qc.md"
        content = qc_file.read_text()

        # Check required headings
        assert "# Appendix: Quality Control" in content
        assert "## QC Summary" in content
        assert "## QC Details" in content
        assert "## Drift Analysis" in content
        assert "## Failure Analysis" in content

        # Check QC content
        assert "5" in content  # Total checks
        assert "4" in content  # Passed
        assert "Data Quality" in content  # Check name
        assert "Outlier removal" in content  # Failure reason

    def test_appendix_qc_details_table(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that appendix_qc.md includes QC details table."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        qc_file = temp_output_dir / "appendix_qc.md"
        content = qc_file.read_text()

        # Check table exists
        assert "| Check | Status | Details |" in content
        assert "| Data Quality | passed | OK |" in content

    def test_appendix_uncertainty_contains_required_sections(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that appendix_uncertainty.md has required sections."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        uncertainty_file = temp_output_dir / "appendix_uncertainty.md"
        content = uncertainty_file.read_text()

        # Check required headings
        assert "# Appendix: Uncertainty Quantification" in content
        assert "## Reliability Analysis" in content
        assert "## Conformal Prediction Coverage" in content
        assert "## Abstention Analysis" in content

        # Check uncertainty content
        assert "0.05" in content  # Calibration error
        assert "0.95" in content  # Coverage
        assert "1.15" in content  # Average set size
        assert "0.08" in content  # Abstention rate

    def test_appendix_reproducibility_contains_required_sections(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that appendix_reproducibility.md has required sections."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        repro_file = temp_output_dir / "appendix_reproducibility.md"
        content = repro_file.read_text()

        # Check required headings
        assert "# Appendix: Reproducibility" in content
        assert "## Execution Details" in content
        assert "## Software Versions" in content
        assert "## Random Seeds" in content
        assert "## Data Integrity" in content
        assert "## Command Line Execution" in content
        assert "## Environment" in content

        # Check reproducibility content
        assert "test_run_001" in content  # Run ID
        assert "42" in content  # Random seed
        assert "abc123def456" in content  # Data hash
        assert "xyz789uvw456" in content  # Config hash
        assert "foodspec analyze --config test.yaml" in content  # Command line
        assert "3.12.9" in content  # Python version

    def test_dossier_index_html_valid(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that dossier_index.html is valid HTML."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        index_file = temp_output_dir / "dossier_index.html"
        content = index_file.read_text()

        # Check HTML structure
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "</html>" in content

    def test_dossier_index_html_contains_links(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that dossier_index.html links to all documents."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        index_file = temp_output_dir / "dossier_index.html"
        content = index_file.read_text()

        # Check links to all documents
        assert 'href="methods.md"' in content
        assert 'href="results.md"' in content
        assert 'href="appendix_qc.md"' in content
        assert 'href="appendix_uncertainty.md"' in content
        assert 'href="appendix_reproducibility.md"' in content

    def test_dossier_index_html_contains_descriptions(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that dossier_index.html has document descriptions."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        index_file = temp_output_dir / "dossier_index.html"
        content = index_file.read_text()

        # Check descriptions
        assert "Methods" in content
        assert "Results" in content
        assert "QC Appendix" in content
        assert "Uncertainty Appendix" in content
        assert "Reproducibility Appendix" in content

    def test_missing_run_directory_raises_error(self, temp_output_dir: Path) -> None:
        """Test that missing run directory raises FileNotFoundError."""
        builder = DossierBuilder()

        with pytest.raises(FileNotFoundError):
            builder.build(
                run_dir="/nonexistent/path", out_dir=temp_output_dir, mode="regulatory"
            )

    def test_missing_manifest_raises_error(self, temp_output_dir: Path) -> None:
        """Test that missing manifest raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            builder = DossierBuilder()

            with pytest.raises(FileNotFoundError):
                builder.build(run_dir=run_dir, out_dir=temp_output_dir)

    def test_invalid_manifest_raises_error(self, temp_output_dir: Path) -> None:
        """Test that invalid manifest raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            # Create invalid manifest
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text("{ invalid json }")

            builder = DossierBuilder()

            with pytest.raises(ValueError):
                builder.build(run_dir=run_dir, out_dir=temp_output_dir)

    def test_build_with_missing_artifacts(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that build handles missing artifacts gracefully."""
        # Remove some artifacts
        (temp_run_dir / "metrics.json").unlink()
        (temp_run_dir / "qc_results.json").unlink()

        builder = DossierBuilder()
        result = builder.build(
            run_dir=temp_run_dir, out_dir=temp_output_dir, mode="research"
        )

        # Should still create all files
        assert (temp_output_dir / "results.md").exists()
        assert (temp_output_dir / "appendix_qc.md").exists()
        assert result.exists()

    def test_build_creates_output_directory(
        self, temp_run_dir: Path, tmp_path: Path
    ) -> None:
        """Test that build creates output directory if it doesn't exist."""
        output_dir = tmp_path / "new" / "nested" / "dossier"
        assert not output_dir.exists()

        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=output_dir)

        assert output_dir.exists()
        assert (output_dir / "dossier_index.html").exists()

    def test_all_documents_have_content(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that all generated documents are not empty."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        for filename in [
            "methods.md",
            "results.md",
            "appendix_qc.md",
            "appendix_uncertainty.md",
            "appendix_reproducibility.md",
            "dossier_index.html",
        ]:
            file_path = temp_output_dir / filename
            content = file_path.read_text()
            assert len(content) > 0, f"{filename} is empty"

    def test_dossier_with_research_mode(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that dossier is created with research mode."""
        builder = DossierBuilder()
        result = builder.build(
            run_dir=temp_run_dir, out_dir=temp_output_dir, mode="research"
        )

        assert result.exists()
        assert (temp_output_dir / "methods.md").exists()

    def test_dossier_with_regulatory_mode(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that dossier is created with regulatory mode."""
        builder = DossierBuilder()
        result = builder.build(
            run_dir=temp_run_dir, out_dir=temp_output_dir, mode="regulatory"
        )

        assert result.exists()
        assert (temp_output_dir / "methods.md").exists()

    def test_dossier_with_monitoring_mode(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that dossier is created with monitoring mode."""
        builder = DossierBuilder()
        result = builder.build(
            run_dir=temp_run_dir, out_dir=temp_output_dir, mode="monitoring"
        )

        assert result.exists()
        assert (temp_output_dir / "methods.md").exists()

    def test_protocol_derived_not_hallucinated(
        self, temp_run_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test that methods are derived from protocol, not hallucinated."""
        builder = DossierBuilder()
        builder.build(run_dir=temp_run_dir, out_dir=temp_output_dir)

        methods_file = temp_output_dir / "methods.md"
        content = methods_file.read_text()

        # Should contain exact values from protocol
        assert "Test Protocol" in content  # From protocol
        assert "1.0.0" in content  # Version from protocol
        assert "Test Dataset" in content  # Data source

        # Should not contain random made-up stuff
        assert "random forest" not in content or "random_forest" in content


class TestDossierIntegration:
    """Integration tests for complete dossier workflow."""

    def test_complete_dossier_generation(self, tmp_path: Path) -> None:
        """Test generating a complete dossier."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Create minimal artifacts
        manifest = {
            "run_id": "integration_test",
            "execution_timestamp": "2026-01-25T12:00:00",
            "protocol_name": "Integration Test",
            "protocol_version": "1.0.0",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        output_dir = tmp_path / "dossier"
        builder = DossierBuilder()
        result = builder.build(run_dir=run_dir, out_dir=output_dir)

        # Verify all files created
        assert result.exists()
        assert (output_dir / "methods.md").exists()
        assert (output_dir / "results.md").exists()
        assert (output_dir / "appendix_qc.md").exists()
        assert (output_dir / "appendix_uncertainty.md").exists()
        assert (output_dir / "appendix_reproducibility.md").exists()
        assert (output_dir / "dossier_index.html").exists()
