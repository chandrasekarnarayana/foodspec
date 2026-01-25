"""Tests for parameter map and data lineage visualizations."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from foodspec.viz.parameters import (
    get_parameter_summary,
    plot_parameter_map,
    _flatten_protocol,
    _identify_non_defaults,
)
from foodspec.viz.lineage import (
    get_lineage_summary,
    plot_data_lineage,
    _extract_lineage_from_manifest,
)


def _make_mock_protocol():
    """Create a mock protocol for testing."""
    protocol = MagicMock()
    protocol.data = MagicMock()
    protocol.data.input = "/data/test.csv"
    protocol.data.format = "csv"
    
    protocol.preprocess = MagicMock()
    protocol.preprocess.recipe = "standard"
    protocol.preprocess.steps = ["baseline", "normalize"]
    
    protocol.qc = MagicMock()
    protocol.qc.thresholds = {"snr": 50}
    protocol.qc.metrics = ["snr", "fwhm"]
    
    protocol.features = MagicMock()
    protocol.features.modules = ["pca"]
    protocol.features.strategy = "pca"
    
    protocol.model = MagicMock()
    protocol.model.estimator = "ensemble"
    protocol.model.hyperparameters = {}
    
    protocol.uncertainty = MagicMock()
    protocol.uncertainty.conformal = {"method": "mondrian"}
    
    protocol.interpretability = MagicMock()
    protocol.interpretability.methods = ["shap"]
    protocol.interpretability.marker_panel = ["panel1"]
    
    protocol.reporting = MagicMock()
    protocol.reporting.format = "html"
    protocol.reporting.sections = ["summary"]
    
    protocol.export = MagicMock()
    protocol.export.bundle = True
    
    return protocol


def _make_mock_manifest():
    """Create a mock manifest for testing."""
    manifest = MagicMock()
    
    manifest.inputs = [
        {
            "path": "/data/samples.csv",
            "hash": "abc123def456",
            "timestamp": "2026-01-25T10:00:00"
        },
        {
            "path": "/data/reference.csv",
            "hash": "xyz789uvw123",
            "timestamp": "2026-01-25T10:01:00"
        }
    ]
    
    manifest.preprocessing = [
        {"name": "baseline_correction", "duration": 2.5},
        {"name": "normalization", "duration": 1.2},
    ]
    
    manifest.processing = [
        {"name": "pca", "n_components": 10, "duration": 5.3},
        {"name": "regression", "model": "ensemble", "duration": 8.1},
    ]
    
    manifest.outputs = [
        {
            "path": "/output/predictions.csv",
            "hash": "output123hash456",
            "timestamp": "2026-01-25T10:30:00"
        },
        {
            "path": "/output/scores.json",
            "hash": "output789hash012",
            "timestamp": "2026-01-25T10:31:00"
        }
    ]
    
    return manifest


# ============================================================================
# Parameter Map Tests
# ============================================================================


class TestParameterFlatten:
    """Test protocol flattening."""

    def test_flatten_extracts_all_parameters(self):
        """Test that flattening extracts all expected parameters."""
        protocol = _make_mock_protocol()
        flattened = _flatten_protocol(protocol)
        
        assert len(flattened) >= 16
        assert "data.input" in flattened
        assert "preprocess.recipe" in flattened
        assert "model.estimator" in flattened

    def test_flatten_values_match_protocol(self):
        """Test that flattened values match protocol."""
        protocol = _make_mock_protocol()
        flattened = _flatten_protocol(protocol)
        
        assert flattened["data.input"] == "/data/test.csv"
        assert flattened["preprocess.recipe"] == "standard"
        assert flattened["model.estimator"] == "ensemble"

    def test_flatten_handles_missing_attributes(self):
        """Test that flattening handles missing attributes gracefully."""
        protocol = MagicMock()
        protocol.data = MagicMock()
        protocol.data.input = "/data/test.csv"
        # Don't set other attributes, should use defaults/None
        
        flattened = _flatten_protocol(protocol)
        
        assert "data.input" in flattened
        assert isinstance(flattened, dict)


class TestNonDefaultDetection:
    """Test non-default parameter detection."""

    def test_detects_non_defaults(self):
        """Test that non-defaults are identified correctly."""
        protocol = _make_mock_protocol()
        non_defaults = _identify_non_defaults(protocol)
        
        # Parameters that differ from defaults
        assert non_defaults["data.input"] is True
        assert non_defaults["preprocess.recipe"] is True
        assert non_defaults["model.estimator"] is True
        assert non_defaults["features.strategy"] is True

    def test_detects_defaults(self):
        """Test that default values are identified."""
        protocol = _make_mock_protocol()
        protocol.features.strategy = "auto"  # This is the default
        non_defaults = _identify_non_defaults(protocol)
        
        assert non_defaults["features.strategy"] is False

    def test_all_parameters_evaluated(self):
        """Test that all parameters are evaluated."""
        protocol = _make_mock_protocol()
        non_defaults = _identify_non_defaults(protocol)
        
        assert len(non_defaults) >= 16


class TestParameterMapPlotting:
    """Test parameter map visualization."""

    def test_plot_returns_figure(self):
        """Test that plot returns matplotlib Figure."""
        protocol = _make_mock_protocol()
        fig = plot_parameter_map(protocol)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_has_axes(self):
        """Test that figure has axes."""
        protocol = _make_mock_protocol()
        fig = plot_parameter_map(protocol)
        
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_with_custom_size(self):
        """Test custom figure size."""
        protocol = _make_mock_protocol()
        custom_size = (12, 8)
        
        fig = plot_parameter_map(protocol, figure_size=custom_size)
        
        w, h = fig.get_size_inches()
        assert np.isclose(w, custom_size[0])
        assert np.isclose(h, custom_size[1])
        
        plt.close(fig)

    def test_plot_saves_png(self):
        """Test PNG file creation."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_parameter_map(protocol, save_path=tmpdir)
            
            png_path = tmpdir / "parameter_map.png"
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            
            plt.close(fig)

    def test_plot_saves_json(self):
        """Test JSON snapshot creation."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_parameter_map(protocol, save_path=tmpdir)
            
            json_path = tmpdir / "parameter_map.json"
            assert json_path.exists()
            
            with open(json_path) as f:
                data = json.load(f)
            
            assert "parameters" in data
            assert "non_defaults" in data
            assert "summary" in data
            
            plt.close(fig)

    def test_plot_json_contains_summary(self):
        """Test that JSON has complete summary."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_parameter_map(protocol, save_path=tmpdir)
            
            json_path = tmpdir / "parameter_map.json"
            with open(json_path) as f:
                data = json.load(f)
            
            summary = data["summary"]
            assert "total_parameters" in summary
            assert "non_default_count" in summary
            assert "non_default_percentage" in summary
            
            plt.close(fig)

    def test_plot_with_custom_dpi(self):
        """Test custom DPI."""
        protocol = _make_mock_protocol()
        custom_dpi = 150
        
        fig = plot_parameter_map(protocol, dpi=custom_dpi)
        
        assert fig.dpi == custom_dpi
        
        plt.close(fig)


class TestParameterSummary:
    """Test parameter summary extraction."""

    def test_summary_has_required_fields(self):
        """Test that summary has all required fields."""
        protocol = _make_mock_protocol()
        summary = get_parameter_summary(protocol)
        
        assert "total_parameters" in summary
        assert "non_default_parameters" in summary
        assert "non_default_percentage" in summary
        assert "non_defaults" in summary
        assert "all_parameters" in summary

    def test_summary_counts_correct(self):
        """Test that counts are correct."""
        protocol = _make_mock_protocol()
        summary = get_parameter_summary(protocol)
        
        assert summary["total_parameters"] > 0
        assert summary["non_default_parameters"] >= 0
        assert summary["non_default_parameters"] <= summary["total_parameters"]

    def test_summary_percentage_valid(self):
        """Test that percentage is valid."""
        protocol = _make_mock_protocol()
        summary = get_parameter_summary(protocol)
        
        percentage = summary["non_default_percentage"]
        assert 0 <= percentage <= 100


# ============================================================================
# Data Lineage Tests
# ============================================================================


class TestLineageExtraction:
    """Test lineage extraction from manifest."""

    def test_extract_all_stages(self):
        """Test that all stages are extracted."""
        manifest = _make_mock_manifest()
        lineage = _extract_lineage_from_manifest(manifest)
        
        assert "inputs" in lineage
        assert "preprocessing" in lineage
        assert "processing" in lineage
        assert "outputs" in lineage

    def test_extract_inputs(self):
        """Test that inputs are extracted correctly."""
        manifest = _make_mock_manifest()
        lineage = _extract_lineage_from_manifest(manifest)
        
        assert len(lineage["inputs"]) == 2
        assert lineage["inputs"][0]["path"] == "/data/samples.csv"

    def test_extract_with_hashes(self):
        """Test that hashes are preserved."""
        manifest = _make_mock_manifest()
        lineage = _extract_lineage_from_manifest(manifest)
        
        input_item = lineage["inputs"][0]
        assert "hash" in input_item
        assert input_item["hash"] == "abc123def456"

    def test_extract_with_timestamps(self):
        """Test that timestamps are preserved."""
        manifest = _make_mock_manifest()
        lineage = _extract_lineage_from_manifest(manifest)
        
        input_item = lineage["inputs"][0]
        assert "timestamp" in input_item
        assert "2026-01-25" in input_item["timestamp"]

    def test_extract_processing_steps(self):
        """Test that processing steps are extracted."""
        manifest = _make_mock_manifest()
        lineage = _extract_lineage_from_manifest(manifest)
        
        assert len(lineage["processing"]) >= 2


class TestDataLineagePlotting:
    """Test data lineage visualization."""

    def test_plot_returns_figure(self):
        """Test that plot returns matplotlib Figure."""
        manifest = _make_mock_manifest()
        fig = plot_data_lineage(manifest)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_has_axes(self):
        """Test that figure has axes."""
        manifest = _make_mock_manifest()
        fig = plot_data_lineage(manifest)
        
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_saves_png(self):
        """Test PNG file creation."""
        manifest = _make_mock_manifest()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_data_lineage(manifest, save_path=tmpdir)
            
            png_path = tmpdir / "data_lineage.png"
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            
            plt.close(fig)

    def test_plot_saves_json(self):
        """Test JSON snapshot creation."""
        manifest = _make_mock_manifest()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_data_lineage(manifest, save_path=tmpdir)
            
            json_path = tmpdir / "data_lineage.json"
            assert json_path.exists()
            
            with open(json_path) as f:
                data = json.load(f)
            
            assert "lineage" in data
            assert "summary" in data
            
            plt.close(fig)

    def test_plot_json_has_summary(self):
        """Test that JSON has complete summary."""
        manifest = _make_mock_manifest()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_data_lineage(manifest, save_path=tmpdir)
            
            json_path = tmpdir / "data_lineage.json"
            with open(json_path) as f:
                data = json.load(f)
            
            summary = data["summary"]
            assert "inputs" in summary
            assert "preprocessing_steps" in summary
            assert "processing_steps" in summary
            assert "outputs" in summary
            assert "total_items" in summary
            
            plt.close(fig)

    def test_plot_with_custom_size(self):
        """Test custom figure size."""
        manifest = _make_mock_manifest()
        custom_size = (14, 8)
        
        fig = plot_data_lineage(manifest, figure_size=custom_size)
        
        w, h = fig.get_size_inches()
        assert np.isclose(w, custom_size[0])
        assert np.isclose(h, custom_size[1])
        
        plt.close(fig)

    def test_plot_with_custom_dpi(self):
        """Test custom DPI."""
        manifest = _make_mock_manifest()
        custom_dpi = 150
        
        fig = plot_data_lineage(manifest, dpi=custom_dpi)
        
        assert fig.dpi == custom_dpi
        
        plt.close(fig)


class TestLineageSummary:
    """Test lineage summary extraction."""

    def test_summary_has_required_fields(self):
        """Test that summary has all required fields."""
        manifest = _make_mock_manifest()
        summary = get_lineage_summary(manifest)
        
        assert "input_count" in summary
        assert "preprocessing_steps" in summary
        assert "processing_steps" in summary
        assert "output_count" in summary
        assert "total_items" in summary
        assert "lineage" in summary

    def test_summary_counts_correct(self):
        """Test that counts match manifest."""
        manifest = _make_mock_manifest()
        summary = get_lineage_summary(manifest)
        
        assert summary["input_count"] == len(manifest.inputs)
        assert summary["output_count"] == len(manifest.outputs)
        assert summary["preprocessing_steps"] == len(manifest.preprocessing)

    def test_summary_total_items_correct(self):
        """Test that total items is sum of all stages."""
        manifest = _make_mock_manifest()
        summary = get_lineage_summary(manifest)
        
        expected_total = (
            summary["input_count"] +
            summary["preprocessing_steps"] +
            summary["processing_steps"] +
            summary["output_count"]
        )
        assert summary["total_items"] == expected_total


class TestIntegration:
    """Integration tests for both visualizations."""

    def test_parameter_map_full_workflow(self):
        """Test complete parameter map workflow."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            fig = plot_parameter_map(protocol, save_path=tmpdir)
            summary = get_parameter_summary(protocol)
            
            assert (tmpdir / "parameter_map.png").exists()
            assert (tmpdir / "parameter_map.json").exists()
            assert summary["total_parameters"] > 0
            
            plt.close(fig)

    def test_data_lineage_full_workflow(self):
        """Test complete data lineage workflow."""
        manifest = _make_mock_manifest()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            fig = plot_data_lineage(manifest, save_path=tmpdir)
            summary = get_lineage_summary(manifest)
            
            assert (tmpdir / "data_lineage.png").exists()
            assert (tmpdir / "data_lineage.json").exists()
            assert summary["total_items"] > 0
            
            plt.close(fig)

    def test_both_visualizations_together(self):
        """Test using both visualizations together."""
        protocol = _make_mock_protocol()
        manifest = _make_mock_manifest()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            fig1 = plot_parameter_map(protocol, save_path=tmpdir)
            fig2 = plot_data_lineage(manifest, save_path=tmpdir)
            
            assert (tmpdir / "parameter_map.png").exists()
            assert (tmpdir / "data_lineage.png").exists()
            assert (tmpdir / "parameter_map.json").exists()
            assert (tmpdir / "data_lineage.json").exists()
            
            plt.close(fig1)
            plt.close(fig2)
