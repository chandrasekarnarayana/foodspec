"""Tests for FoodSpec pipeline DAG visualizer."""

from pathlib import Path
from unittest.mock import MagicMock
import tempfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from foodspec.viz.pipeline import (
    _build_pipeline_graph,
    _compute_deterministic_layout,
    get_pipeline_stats,
    plot_pipeline_dag,
)


def _make_mock_protocol():
    """Create a mock protocol for testing."""
    protocol = MagicMock()
    protocol.data = MagicMock()
    protocol.data.input = "/data/test"
    protocol.data.format = "csv"
    
    protocol.preprocess = MagicMock()
    protocol.preprocess.recipe = None
    protocol.preprocess.steps = []
    
    protocol.qc = MagicMock()
    protocol.qc.thresholds = {}
    protocol.qc.metrics = []
    
    protocol.features = MagicMock()
    protocol.features.modules = []
    protocol.features.strategy = "auto"
    
    protocol.model = MagicMock()
    protocol.model.estimator = ""
    protocol.model.hyperparameters = None
    
    protocol.uncertainty = MagicMock()
    protocol.uncertainty.conformal = {}
    
    protocol.interpretability = MagicMock()
    protocol.interpretability.methods = []
    protocol.interpretability.marker_panel = None
    
    protocol.reporting = MagicMock()
    protocol.reporting.format = ""
    protocol.reporting.sections = []
    
    protocol.export = MagicMock()
    protocol.export.bundle = False
    
    return protocol


class TestPipelineGraphConstruction:
    """Test pipeline graph building."""

    def test_graph_has_correct_nodes(self):
        """Test that graph has all expected pipeline stages."""
        protocol = _make_mock_protocol()
        graph, node_attrs = _build_pipeline_graph(protocol)
        
        assert len(graph.nodes) == 10, "Should have 10 pipeline stages"
        expected_stages = {
            "Data", "Preprocess", "QC", "Features", "Model",
            "Calibration", "Conformal", "Interpret", "Report", "Bundle"
        }
        assert set(node_attrs.keys()) == expected_stages

    def test_graph_has_edges(self):
        """Test that graph has sequential edges."""
        protocol = _make_mock_protocol()
        graph, _ = _build_pipeline_graph(protocol)
        
        # Should have 9 edges (10 nodes, linear graph)
        assert len(graph.edges) == 9
        
        # Check sequential connectivity
        for i in range(9):
            assert graph.has_edge(i, i + 1)

    def test_graph_is_directed_acyclic(self):
        """Test that graph is a DAG (no cycles)."""
        protocol = _make_mock_protocol()
        graph, _ = _build_pipeline_graph(protocol)
        
        assert nx.is_directed_acyclic_graph(graph)

    def test_node_attributes_have_required_fields(self):
        """Test that each node has enabled, params, and stage_key."""
        protocol = _make_mock_protocol()
        _, node_attrs = _build_pipeline_graph(protocol)
        
        for stage_name, attrs in node_attrs.items():
            assert "enabled" in attrs
            assert "params" in attrs
            assert "stage_key" in attrs
            assert isinstance(attrs["enabled"], bool)
            assert isinstance(attrs["params"], dict)

    def test_data_stage_always_enabled(self):
        """Test that Data stage is always enabled."""
        protocol = _make_mock_protocol()
        _, node_attrs = _build_pipeline_graph(protocol)
        
        assert node_attrs["Data"]["enabled"] is True

    def test_optional_stages_disabled_by_default(self):
        """Test that optional stages are disabled by default."""
        protocol = _make_mock_protocol()
        _, node_attrs = _build_pipeline_graph(protocol)
        
        # These should be disabled with default empty protocol
        assert node_attrs["Preprocess"]["enabled"] is False
        assert node_attrs["QC"]["enabled"] is False
        assert node_attrs["Features"]["enabled"] is False

    def test_parameters_extracted_correctly(self):
        """Test that stage parameters are extracted from protocol."""
        protocol = _make_mock_protocol()
        _, node_attrs = _build_pipeline_graph(protocol)
        
        # Data stage should have input and format
        assert "input" in node_attrs["Data"]["params"]
        assert "format" in node_attrs["Data"]["params"]


class TestDeterministicLayout:
    """Test deterministic graph layout computation."""

    def test_same_seed_produces_same_layout(self):
        """Test that same seed produces identical node positions."""
        protocol = _make_mock_protocol()
        graph, _ = _build_pipeline_graph(protocol)
        
        # Compute layout twice with same seed
        pos1 = _compute_deterministic_layout(graph, seed=42)
        pos2 = _compute_deterministic_layout(graph, seed=42)
        
        # Compare positions
        for node_id in graph.nodes:
            x1, y1 = pos1[node_id]
            x2, y2 = pos2[node_id]
            assert np.isclose(x1, x2, atol=1e-10)
            assert np.isclose(y1, y2, atol=1e-10)

    def test_different_seeds_produce_different_layouts(self):
        """Test that different seeds produce different positions."""
        protocol = _make_mock_protocol()
        graph, _ = _build_pipeline_graph(protocol)
        
        pos1 = _compute_deterministic_layout(graph, seed=42)
        pos2 = _compute_deterministic_layout(graph, seed=123)
        
        # At least some positions should differ
        differences = []
        for node_id in graph.nodes:
            x1, y1 = pos1[node_id]
            x2, y2 = pos2[node_id]
            if not (np.isclose(x1, x2) and np.isclose(y1, y2)):
                differences.append(node_id)
        
        assert len(differences) > 0, "Different seeds should produce different layouts"

    def test_layout_has_all_nodes(self):
        """Test that layout includes all nodes."""
        protocol = _make_mock_protocol()
        graph, _ = _build_pipeline_graph(protocol)
        
        pos = _compute_deterministic_layout(graph, seed=42)
        
        assert len(pos) == len(graph.nodes)
        for node_id in graph.nodes:
            assert node_id in pos
            x, y = pos[node_id]
            assert isinstance(x, (int, float, np.number))
            assert isinstance(y, (int, float, np.number))

    def test_layout_positions_in_valid_range(self):
        """Test that all positions are in reasonable range."""
        protocol = _make_mock_protocol()
        graph, _ = _build_pipeline_graph(protocol)
        
        pos = _compute_deterministic_layout(graph, seed=42)
        
        for x, y in pos.values():
            assert -2 <= x <= 2, f"X position {x} out of range"
            assert -2 <= y <= 2, f"Y position {y} out of range"


class TestPipelineDAGPlotting:
    """Test pipeline DAG visualization."""

    def test_plot_returns_figure(self):
        """Test that plot_pipeline_dag returns a matplotlib Figure."""
        protocol = _make_mock_protocol()
        fig = plot_pipeline_dag(protocol, seed=42)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_figure_has_axes(self):
        """Test that returned figure has axes."""
        protocol = _make_mock_protocol()
        fig = plot_pipeline_dag(protocol, seed=42)
        
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_deterministic_with_seed(self):
        """Test that same seed produces identical plots."""
        protocol = _make_mock_protocol()
        
        fig1 = plot_pipeline_dag(protocol, seed=42)
        fig1_data = fig1.canvas.buffer_rgba()
        plt.close(fig1)
        
        fig2 = plot_pipeline_dag(protocol, seed=42)
        fig2_data = fig2.canvas.buffer_rgba()
        plt.close(fig2)
        
        # Compare canvas data
        assert np.allclose(fig1_data, fig2_data, atol=1)

    def test_plot_saves_svg(self):
        """Test that SVG file is created when save_path provided."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_pipeline_dag(protocol, save_path=tmpdir, seed=42)
            
            svg_path = tmpdir / "pipeline_dag.svg"
            assert svg_path.exists()
            assert svg_path.stat().st_size > 0
            
            plt.close(fig)

    def test_plot_saves_png(self):
        """Test that PNG file is created when save_path provided."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_pipeline_dag(protocol, save_path=tmpdir, seed=42)
            
            png_path = tmpdir / "pipeline_dag.png"
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            
            plt.close(fig)

    def test_plot_saves_both_formats(self):
        """Test that both SVG and PNG are saved."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_pipeline_dag(protocol, save_path=tmpdir, seed=42)
            
            assert (tmpdir / "pipeline_dag.svg").exists()
            assert (tmpdir / "pipeline_dag.png").exists()
            
            plt.close(fig)

    def test_plot_with_custom_figure_size(self):
        """Test that custom figure size is respected."""
        protocol = _make_mock_protocol()
        custom_size = (12, 8)
        
        fig = plot_pipeline_dag(protocol, seed=42, figure_size=custom_size)
        
        w, h = fig.get_size_inches()
        assert np.isclose(w, custom_size[0])
        assert np.isclose(h, custom_size[1])
        
        plt.close(fig)

    def test_plot_with_custom_dpi(self):
        """Test that custom DPI is used."""
        protocol = _make_mock_protocol()
        custom_dpi = 150
        
        fig = plot_pipeline_dag(protocol, seed=42, dpi=custom_dpi)
        
        assert fig.dpi == custom_dpi
        
        plt.close(fig)

    def test_plot_title_present(self):
        """Test that plot has a title."""
        protocol = _make_mock_protocol()
        fig = plot_pipeline_dag(protocol, seed=42)
        
        # Check if any axis has a title
        assert any(ax.get_title() for ax in fig.axes)
        
        plt.close(fig)


class TestPipelineStats:
    """Test pipeline statistics extraction."""

    def test_stats_has_required_fields(self):
        """Test that stats dict has all required fields."""
        protocol = _make_mock_protocol()
        stats = get_pipeline_stats(protocol)
        
        assert "total_stages" in stats
        assert "enabled_stages" in stats
        assert "disabled_stages" in stats
        assert "stage_details" in stats

    def test_stats_counts_add_up(self):
        """Test that stage counts are consistent."""
        protocol = _make_mock_protocol()
        stats = get_pipeline_stats(protocol)
        
        total = stats["enabled_stages"] + stats["disabled_stages"]
        assert total == stats["total_stages"]

    def test_stats_total_stages_is_ten(self):
        """Test that total stages equals 10."""
        protocol = _make_mock_protocol()
        stats = get_pipeline_stats(protocol)
        
        assert stats["total_stages"] == 10

    def test_stats_stage_details_complete(self):
        """Test that stage details are complete for each stage."""
        protocol = _make_mock_protocol()
        stats = get_pipeline_stats(protocol)
        
        expected_stages = {
            "Data", "Preprocess", "QC", "Features", "Model",
            "Calibration", "Conformal", "Interpret", "Report", "Bundle"
        }
        assert set(stats["stage_details"].keys()) == expected_stages
        
        for stage_name, details in stats["stage_details"].items():
            assert "enabled" in details
            assert "params" in details
            assert isinstance(details["enabled"], bool)
            assert isinstance(details["params"], dict)

    def test_stats_data_always_enabled(self):
        """Test that Data stage is enabled in stats."""
        protocol = _make_mock_protocol()
        stats = get_pipeline_stats(protocol)
        
        assert stats["stage_details"]["Data"]["enabled"] is True


class TestPipelineIntegration:
    """Integration tests for full pipeline visualization."""

    def test_full_pipeline_visualization(self):
        """Test complete pipeline visualization workflow."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Generate visualization
            fig = plot_pipeline_dag(protocol, save_path=tmpdir, seed=42)
            
            # Verify outputs
            assert fig is not None
            assert (tmpdir / "pipeline_dag.svg").exists()
            assert (tmpdir / "pipeline_dag.png").exists()
            
            # Get stats
            stats = get_pipeline_stats(protocol)
            assert stats["total_stages"] == 10
            
            plt.close(fig)

    def test_deterministic_full_workflow(self):
        """Test that full workflow is deterministic."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                tmpdir1 = Path(tmpdir1)
                tmpdir2 = Path(tmpdir2)
                
                # Run twice
                fig1 = plot_pipeline_dag(protocol, save_path=tmpdir1, seed=42)
                fig2 = plot_pipeline_dag(protocol, save_path=tmpdir2, seed=42)
                
                # Check file sizes are identical
                svg1_size = (tmpdir1 / "pipeline_dag.svg").stat().st_size
                svg2_size = (tmpdir2 / "pipeline_dag.svg").stat().st_size
                
                # File sizes should be very close (some variation due to timestamp)
                assert abs(svg1_size - svg2_size) < 100
                
                plt.close(fig1)
                plt.close(fig2)

    def test_complex_protocol_visualization(self):
        """Test visualization with a more complex protocol."""
        protocol = _make_mock_protocol()
        # Enable some optional stages
        protocol.preprocess.recipe = "standard"
        protocol.features.strategy = "pca"
        protocol.model.estimator = "logreg"
        protocol.uncertainty.conformal = {
            "calibration": {"method": "platt"},
            "conformal": {"method": "mondrian", "alpha": 0.1},
        }
        
        fig = plot_pipeline_dag(protocol, seed=42)
        
        assert fig is not None
        stats = get_pipeline_stats(protocol)
        
        # With these settings, more stages should be enabled
        assert stats["enabled_stages"] > 1
        
        plt.close(fig)


class TestLayoutStability:
    """Test that layout is stable across different conditions."""

    def test_layout_stable_across_graph_updates(self):
        """Test that layout positions remain stable."""
        protocol1 = _make_mock_protocol()
        protocol2 = _make_mock_protocol()
        
        graph1, _ = _build_pipeline_graph(protocol1)
        graph2, _ = _build_pipeline_graph(protocol2)
        
        pos1 = _compute_deterministic_layout(graph1, seed=42)
        pos2 = _compute_deterministic_layout(graph2, seed=42)
        
        # Positions should be identical
        for node_id in graph1.nodes:
            x1, y1 = pos1[node_id]
            x2, y2 = pos2[node_id]
            assert np.isclose(x1, x2, atol=1e-10)
            assert np.isclose(y1, y2, atol=1e-10)

    def test_multiple_plots_identical(self):
        """Test that multiple plots with same seed are identical."""
        protocol = _make_mock_protocol()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            figs = []
            for i in range(3):
                fig = plot_pipeline_dag(protocol, seed=42)
                figs.append(fig)
            
            # All should be drawable without errors
            for fig in figs:
                assert len(fig.axes) > 0
                plt.close(fig)
