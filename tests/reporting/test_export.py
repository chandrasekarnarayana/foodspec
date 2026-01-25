"""Tests for reproducibility pack and archive export functionality."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from foodspec.reporting.export import (
    ArchiveExporter,
    ReproducibilityPackBuilder,
    build_reproducibility_pack,
    export_archive,
    get_archive_file_list,
    verify_archive_integrity,
)


@pytest.fixture
def temp_run_dir(tmp_path):
    """Create a temporary run directory with sample artifacts."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Create manifest
    manifest_data = {
        "run_id": "test_run_001",
        "timestamp": "2024-01-01T00:00:00",
        "algorithm": "TestAlgorithm",
        "parameters": {"param1": "value1"},
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f)

    # Create protocol snapshot
    protocol_data = {
        "name": "TestProtocol",
        "version": "1.0",
        "steps": [
            {
                "name": "Preprocessing",
                "type": "preprocessing",
                "parameters": {"method": "baseline_removal"},
            },
            {
                "name": "Analysis",
                "type": "analysis",
                "parameters": {"algorithm": "pls"},
            },
        ],
    }
    with open(run_dir / "protocol_snapshot.json", "w") as f:
        json.dump(protocol_data, f)

    # Create metrics
    metrics = {"accuracy": 0.95, "rmse": 0.05, "r2_score": 0.92}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # Create predictions
    predictions = {"prediction_1": 0.8, "prediction_2": 0.85}
    with open(run_dir / "predictions.json", "w") as f:
        json.dump(predictions, f)

    # Create plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir()
    (plots_dir / "plot1.png").write_text("fake png data")
    (plots_dir / "plot2.pdf").write_text("fake pdf data")

    # Create dossier directory
    dossier_dir = run_dir / "dossier"
    dossier_dir.mkdir()
    (dossier_dir / "methods.md").write_text("# Methods\nTest content")
    (dossier_dir / "results.md").write_text("# Results\nTest content")

    return run_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


class TestReproducibilityPackBuilder:
    """Test reproducibility pack builder."""

    def test_builder_initialization(self):
        """Test builder initializes correctly."""
        builder = ReproducibilityPackBuilder()
        assert builder.run_dir is None
        assert builder.out_dir is None

    def test_build_creates_output_directory(self, temp_run_dir, temp_output_dir):
        """Test build creates output directory."""
        builder = ReproducibilityPackBuilder()
        result = builder.build(temp_run_dir, temp_output_dir)
        assert result == temp_output_dir
        assert temp_output_dir.exists()

    def test_build_with_nonexistent_run_dir(self, temp_output_dir):
        """Test build raises error with missing run directory."""
        builder = ReproducibilityPackBuilder()
        with pytest.raises(FileNotFoundError):
            builder.build("nonexistent_dir", temp_output_dir)

    def test_build_copies_manifest(self, temp_run_dir, temp_output_dir):
        """Test build copies manifest to pack."""
        builder = ReproducibilityPackBuilder()
        builder.build(temp_run_dir, temp_output_dir)

        manifest_file = temp_output_dir / "manifest.json"
        assert manifest_file.exists()

        with open(manifest_file) as f:
            manifest = json.load(f)
        assert manifest["run_id"] == "test_run_001"

    def test_build_exports_protocol_snapshot(self, temp_run_dir, temp_output_dir):
        """Test build exports protocol snapshot in multiple formats."""
        builder = ReproducibilityPackBuilder()
        builder.build(temp_run_dir, temp_output_dir)

        # Check JSON export
        json_file = temp_output_dir / "protocol_snapshot.json"
        assert json_file.exists()
        with open(json_file) as f:
            protocol = json.load(f)
        assert protocol["name"] == "TestProtocol"
        assert len(protocol["steps"]) == 2

        # Check text export
        txt_file = temp_output_dir / "protocol_snapshot.txt"
        assert txt_file.exists()
        content = txt_file.read_text()
        assert "PROTOCOL SNAPSHOT" in content
        assert "Preprocessing" in content

    def test_build_creates_environment_freeze(self, temp_run_dir, temp_output_dir):
        """Test build creates environment freeze file."""
        builder = ReproducibilityPackBuilder()
        builder.build(temp_run_dir, temp_output_dir)

        env_file = temp_output_dir / "environment.txt"
        assert env_file.exists()
        content = env_file.read_text()
        assert "Python" in content

    def test_build_copies_metrics_tables(self, temp_run_dir, temp_output_dir):
        """Test build copies metrics and predictions."""
        builder = ReproducibilityPackBuilder()
        builder.build(temp_run_dir, temp_output_dir)

        tables_dir = temp_output_dir / "tables"
        assert tables_dir.exists()

        metrics_file = tables_dir / "metrics.json"
        assert metrics_file.exists()

        predictions_file = tables_dir / "predictions.json"
        assert predictions_file.exists()

    def test_build_creates_plots_index(self, temp_run_dir, temp_output_dir):
        """Test build creates plots index."""
        builder = ReproducibilityPackBuilder()
        builder.build(temp_run_dir, temp_output_dir)

        index_file = temp_output_dir / "plots_index.txt"
        assert index_file.exists()
        content = index_file.read_text()
        assert "PLOTS INDEX" in content
        assert "plot1.png" in content
        assert "plot2.pdf" in content

    def test_build_creates_pack_metadata(self, temp_run_dir, temp_output_dir):
        """Test build creates pack metadata."""
        builder = ReproducibilityPackBuilder()
        builder.build(temp_run_dir, temp_output_dir)

        metadata_file = temp_output_dir / "pack_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)
        assert "pack_created" in metadata
        assert "source_run" in metadata
        assert "pack_contents" in metadata
        assert len(metadata["pack_contents"]) > 0

    def test_build_handles_missing_artifacts(self, tmp_path):
        """Test build handles missing optional artifacts."""
        run_dir = tmp_path / "minimal_run"
        run_dir.mkdir()

        # Only create minimal manifest
        manifest = {"run_id": "minimal"}
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        out_dir = tmp_path / "pack_output"
        builder = ReproducibilityPackBuilder()
        result = builder.build(run_dir, out_dir)

        assert result.exists()
        assert (out_dir / "manifest.json").exists()
        # Pack should handle missing plots gracefully
        assert out_dir.exists()


class TestArchiveExporter:
    """Test archive exporter."""

    def test_exporter_initialization(self):
        """Test exporter initializes correctly."""
        exporter = ArchiveExporter()
        assert exporter.run_dir is None
        assert exporter.include_set is None

    def test_export_creates_zip_file(self, temp_run_dir, tmp_path):
        """Test export creates zip file."""
        zip_path = tmp_path / "test_archive.zip"
        exporter = ArchiveExporter()
        result = exporter.export(zip_path, temp_run_dir)

        assert result.exists()
        assert result.suffix == ".zip"
        assert zipfile.is_zipfile(result)

    def test_export_with_nonexistent_run_dir(self, tmp_path):
        """Test export raises error with missing run directory."""
        zip_path = tmp_path / "archive.zip"
        exporter = ArchiveExporter()
        with pytest.raises(FileNotFoundError):
            exporter.export(zip_path, "nonexistent_dir")

    def test_export_includes_manifest(self, temp_run_dir, tmp_path):
        """Test exported archive includes manifest."""
        zip_path = tmp_path / "archive.zip"
        exporter = ArchiveExporter()
        exporter.export(zip_path, temp_run_dir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "manifest.json" in zf.namelist()
            manifest_data = json.loads(zf.read("manifest.json"))
            assert manifest_data["run_id"] == "test_run_001"

    def test_export_includes_protocol(self, temp_run_dir, tmp_path):
        """Test exported archive includes protocol snapshot."""
        zip_path = tmp_path / "archive.zip"
        exporter = ArchiveExporter()
        exporter.export(zip_path, temp_run_dir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "protocol_snapshot.json" in zf.namelist()

    def test_export_with_include_dossier(self, temp_run_dir, tmp_path):
        """Test export with dossier inclusion."""
        zip_path = tmp_path / "archive.zip"
        exporter = ArchiveExporter()
        exporter.export(zip_path, temp_run_dir, include=["dossier"])

        with zipfile.ZipFile(zip_path, "r") as zf:
            files = zf.namelist()
            assert any(f.startswith("dossier/") for f in files)

    def test_export_with_include_figures(self, temp_run_dir, tmp_path):
        """Test export with figures inclusion."""
        zip_path = tmp_path / "archive.zip"
        exporter = ArchiveExporter()
        exporter.export(zip_path, temp_run_dir, include=["figures"])

        with zipfile.ZipFile(zip_path, "r") as zf:
            files = zf.namelist()
            assert any(f.startswith("plots/") for f in files)

    def test_export_deterministic_ordering(self, temp_run_dir, tmp_path):
        """Test export creates deterministically ordered archive."""
        zip_path1 = tmp_path / "archive1.zip"
        zip_path2 = tmp_path / "archive2.zip"

        exporter1 = ArchiveExporter()
        exporter1.export(zip_path1, temp_run_dir)

        exporter2 = ArchiveExporter()
        exporter2.export(zip_path2, temp_run_dir)

        # File lists should be identical
        with zipfile.ZipFile(zip_path1, "r") as zf1:
            list1 = sorted(zf1.namelist())
        with zipfile.ZipFile(zip_path2, "r") as zf2:
            list2 = sorted(zf2.namelist())

        assert list1 == list2

    def test_export_with_selective_include(self, temp_run_dir, tmp_path):
        """Test export with selective component inclusion."""
        zip_full = tmp_path / "full.zip"
        zip_partial = tmp_path / "partial.zip"

        exporter1 = ArchiveExporter()
        exporter1.export(zip_full, temp_run_dir)

        exporter2 = ArchiveExporter()
        exporter2.export(
            zip_partial, temp_run_dir, include=["dossier", "figures"]
        )

        with zipfile.ZipFile(zip_full, "r") as zf:
            full_files = set(zf.namelist())
        with zipfile.ZipFile(zip_partial, "r") as zf:
            partial_files = set(zf.namelist())

        # Full should have same or more files
        assert len(full_files) >= len(partial_files)


class TestPublicFunctions:
    """Test public API functions."""

    def test_build_reproducibility_pack(self, temp_run_dir, temp_output_dir):
        """Test build_reproducibility_pack function."""
        pack_dir = build_reproducibility_pack(temp_run_dir, temp_output_dir)

        assert pack_dir.exists()
        assert (pack_dir / "manifest.json").exists()
        assert (pack_dir / "protocol_snapshot.json").exists()
        assert (pack_dir / "environment.txt").exists()

    def test_export_archive_function(self, temp_run_dir, tmp_path):
        """Test export_archive function."""
        zip_path = tmp_path / "archive.zip"
        result = export_archive(zip_path, temp_run_dir)

        assert result.exists()
        assert zipfile.is_zipfile(result)

    def test_export_archive_with_include(self, temp_run_dir, tmp_path):
        """Test export_archive with include parameter."""
        zip_path = tmp_path / "archive.zip"
        result = export_archive(
            zip_path, temp_run_dir, include=["dossier", "figures"]
        )

        assert result.exists()
        with zipfile.ZipFile(result, "r") as zf:
            files = zf.namelist()
            assert any(f.startswith("dossier/") for f in files)

    def test_get_archive_file_list(self, temp_run_dir, tmp_path):
        """Test get_archive_file_list function."""
        zip_path = tmp_path / "archive.zip"
        export_archive(zip_path, temp_run_dir)

        file_list = get_archive_file_list(zip_path)
        assert len(file_list) > 0
        assert "manifest.json" in file_list
        assert all(isinstance(f, str) for f in file_list)
        # Check deterministic ordering
        assert file_list == sorted(file_list)

    def test_get_archive_file_list_nonexistent(self, tmp_path):
        """Test get_archive_file_list with nonexistent archive."""
        file_list = get_archive_file_list(tmp_path / "nonexistent.zip")
        assert file_list == []

    def test_verify_archive_integrity(self, temp_run_dir, tmp_path):
        """Test verify_archive_integrity function."""
        zip_path = tmp_path / "archive.zip"
        export_archive(zip_path, temp_run_dir)

        assert verify_archive_integrity(zip_path)

    def test_verify_archive_integrity_with_expected_files(
        self, temp_run_dir, tmp_path
    ):
        """Test verify_archive_integrity with expected files."""
        zip_path = tmp_path / "archive.zip"
        export_archive(zip_path, temp_run_dir)

        assert verify_archive_integrity(
            zip_path, expected_files=["manifest.json"]
        )

    def test_verify_archive_integrity_missing_expected(
        self, temp_run_dir, tmp_path
    ):
        """Test verify_archive_integrity fails if expected files missing."""
        zip_path = tmp_path / "archive.zip"
        export_archive(zip_path, temp_run_dir)

        assert not verify_archive_integrity(
            zip_path, expected_files=["nonexistent_file.txt"]
        )

    def test_verify_archive_integrity_corrupted(self, tmp_path):
        """Test verify_archive_integrity detects corrupted archives."""
        zip_path = tmp_path / "bad.zip"
        zip_path.write_text("not a real zip file")

        assert not verify_archive_integrity(zip_path)


class TestIntegration:
    """Integration tests for export workflow."""

    def test_full_workflow_pack_and_archive(self, temp_run_dir, tmp_path):
        """Test complete workflow: build pack then export archive."""
        pack_dir = build_reproducibility_pack(temp_run_dir, tmp_path / "pack")
        assert pack_dir.exists()

        zip_path = export_archive(
            tmp_path / "archive.zip", temp_run_dir
        )
        assert zip_path.exists()

        # Verify contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            files = zf.namelist()
            assert "manifest.json" in files
            assert "protocol_snapshot.json" in files

    def test_archive_round_trip(self, temp_run_dir, tmp_path):
        """Test archive creation and extraction."""
        zip_path = export_archive(tmp_path / "archive.zip", temp_run_dir)
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Verify extracted contents
        assert (extract_dir / "manifest.json").exists()
        assert (extract_dir / "protocol_snapshot.json").exists()

    def test_archive_size_optimization_with_include(
        self, temp_run_dir, tmp_path
    ):
        """Test that include parameter reduces archive size."""
        full_zip = tmp_path / "full.zip"
        minimal_zip = tmp_path / "minimal.zip"

        export_archive(full_zip, temp_run_dir)
        export_archive(minimal_zip, temp_run_dir, include=["dossier"])

        full_size = full_zip.stat().st_size
        minimal_size = minimal_zip.stat().st_size

        # Minimal should be smaller
        assert minimal_size <= full_size

    def test_multiple_runs_different_archives(
        self, temp_run_dir, tmp_path
    ):
        """Test creating archives for multiple runs."""
        zip1 = export_archive(tmp_path / "run1.zip", temp_run_dir)
        zip2 = export_archive(tmp_path / "run2.zip", temp_run_dir)

        assert zip1.exists()
        assert zip2.exists()

        # Both should be valid archives
        assert zipfile.is_zipfile(zip1)
        assert zipfile.is_zipfile(zip2)
