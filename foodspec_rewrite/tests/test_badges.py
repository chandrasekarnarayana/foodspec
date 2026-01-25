"""Tests for reproducibility badge generator."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytest

from foodspec.viz.badges import (
    get_reproducibility_status,
    plot_reproducibility_badge,
    _extract_reproducibility_info,
    _determine_badge_level,
)


def _make_mock_manifest_full():
    """Create fully reproducible manifest."""
    manifest = MagicMock()
    manifest.seed = 42
    manifest.protocol_hash = "abc123def456"
    manifest.data_hash = "xyz789uvw012"
    manifest.env_hash = "env123hash456"
    return manifest


def _make_mock_manifest_partial():
    """Create partially reproducible manifest (missing env)."""
    manifest = MagicMock()
    manifest.seed = 42
    manifest.protocol_hash = "abc123def456"
    manifest.data_hash = "xyz789uvw012"
    manifest.env_hash = None
    return manifest


def _make_mock_manifest_none():
    """Create non-reproducible manifest (missing critical items)."""
    manifest = MagicMock()
    manifest.seed = None
    manifest.protocol_hash = "abc123def456"
    manifest.data_hash = None
    manifest.env_hash = None
    return manifest


class TestReproducibilityExtraction:
    """Test extraction of reproducibility information."""

    def test_extract_all_present(self):
        """Test extraction when all items present."""
        manifest = _make_mock_manifest_full()
        info = _extract_reproducibility_info(manifest)
        
        assert info["seed"] == 42
        assert info["protocol_hash"] == "abc123def456"
        assert info["data_hash"] == "xyz789uvw012"
        assert info["env_hash"] == "env123hash456"

    def test_extract_partial(self):
        """Test extraction with some items missing."""
        manifest = _make_mock_manifest_partial()
        info = _extract_reproducibility_info(manifest)
        
        assert info["seed"] == 42
        assert info["protocol_hash"] is not None
        assert info["data_hash"] is not None
        assert info["env_hash"] is None

    def test_extract_nested_attributes(self):
        """Test extraction from nested attributes."""
        manifest = MagicMock()
        # Remove direct attributes to force fallback to nested
        del manifest.seed
        del manifest.protocol_hash
        del manifest.data_hash
        del manifest.env_hash
        
        # Set nested attributes
        manifest.config = MagicMock()
        manifest.config.seed = 123
        manifest.hashes = MagicMock()
        manifest.hashes.protocol = "proto_hash"
        manifest.hashes.data = "data_hash"
        manifest.environment = MagicMock()
        manifest.environment.hash = "env_hash"
        
        info = _extract_reproducibility_info(manifest)
        
        assert info["seed"] == 123
        assert info["protocol_hash"] == "proto_hash"
        assert info["data_hash"] == "data_hash"
        assert info["env_hash"] == "env_hash"

    def test_extract_all_missing(self):
        """Test extraction when all items missing."""
        manifest = MagicMock()
        manifest.seed = None
        manifest.protocol_hash = None
        manifest.data_hash = None
        manifest.env_hash = None
        
        info = _extract_reproducibility_info(manifest)
        
        assert all(v is None for v in info.values())


class TestBadgeLevelDetermination:
    """Test badge level and color determination."""

    def test_green_level_all_present(self):
        """Test green level when all components present."""
        info = {
            "seed": 42,
            "protocol_hash": "abc123",
            "data_hash": "xyz789",
            "env_hash": "env123",
        }
        level, color, status = _determine_badge_level(info)
        
        assert level == "green"
        assert color == "#4CAF50"
        assert "Fully" in status

    def test_yellow_level_missing_env(self):
        """Test yellow level when only env missing."""
        info = {
            "seed": 42,
            "protocol_hash": "abc123",
            "data_hash": "xyz789",
            "env_hash": None,
        }
        level, color, status = _determine_badge_level(info)
        
        assert level == "yellow"
        assert color == "#FFC107"
        assert "Partial" in status

    def test_red_level_missing_seed(self):
        """Test red level when seed missing."""
        info = {
            "seed": None,
            "protocol_hash": "abc123",
            "data_hash": "xyz789",
            "env_hash": "env123",
        }
        level, color, status = _determine_badge_level(info)
        
        assert level == "red"
        assert color == "#F44336"
        assert "Not" in status

    def test_red_level_missing_protocol(self):
        """Test red level when protocol hash missing."""
        info = {
            "seed": 42,
            "protocol_hash": None,
            "data_hash": "xyz789",
            "env_hash": "env123",
        }
        level, color, status = _determine_badge_level(info)
        
        assert level == "red"

    def test_red_level_missing_data(self):
        """Test red level when data hash missing."""
        info = {
            "seed": 42,
            "protocol_hash": "abc123",
            "data_hash": None,
            "env_hash": "env123",
        }
        level, color, status = _determine_badge_level(info)
        
        assert level == "red"

    def test_red_level_all_missing(self):
        """Test red level when all missing."""
        info = {
            "seed": None,
            "protocol_hash": None,
            "data_hash": None,
            "env_hash": None,
        }
        level, color, status = _determine_badge_level(info)
        
        assert level == "red"


class TestBadgePlotting:
    """Test badge visualization."""

    def test_plot_returns_figure(self):
        """Test that plotting returns matplotlib Figure."""
        manifest = _make_mock_manifest_full()
        fig = plot_reproducibility_badge(manifest)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_has_axes(self):
        """Test that figure has axes."""
        manifest = _make_mock_manifest_full()
        fig = plot_reproducibility_badge(manifest)
        
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_green_badge(self):
        """Test plotting fully reproducible badge."""
        manifest = _make_mock_manifest_full()
        fig = plot_reproducibility_badge(manifest)
        
        assert fig is not None
        plt.close(fig)

    def test_plot_yellow_badge(self):
        """Test plotting partially reproducible badge."""
        manifest = _make_mock_manifest_partial()
        fig = plot_reproducibility_badge(manifest)
        
        assert fig is not None
        plt.close(fig)

    def test_plot_red_badge(self):
        """Test plotting non-reproducible badge."""
        manifest = _make_mock_manifest_none()
        fig = plot_reproducibility_badge(manifest)
        
        assert fig is not None
        plt.close(fig)

    def test_plot_saves_png(self):
        """Test PNG file creation."""
        manifest = _make_mock_manifest_full()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_reproducibility_badge(manifest, save_path=tmpdir)
            
            png_path = tmpdir / "reproducibility_badge.png"
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            
            plt.close(fig)

    def test_plot_with_custom_size(self):
        """Test custom figure size."""
        manifest = _make_mock_manifest_full()
        custom_size = (3, 1.5)
        
        fig = plot_reproducibility_badge(manifest, figure_size=custom_size)
        
        w, h = fig.get_size_inches()
        assert abs(w - custom_size[0]) < 0.1
        assert abs(h - custom_size[1]) < 0.1
        
        plt.close(fig)

    def test_plot_with_custom_dpi(self):
        """Test custom DPI."""
        manifest = _make_mock_manifest_full()
        custom_dpi = 100
        
        fig = plot_reproducibility_badge(manifest, dpi=custom_dpi)
        
        assert fig.dpi == custom_dpi
        
        plt.close(fig)


class TestReproducibilityStatus:
    """Test reproducibility status extraction."""

    def test_status_has_required_fields(self):
        """Test that status has all required fields."""
        manifest = _make_mock_manifest_full()
        status = get_reproducibility_status(manifest)
        
        assert "level" in status
        assert "status" in status
        assert "color" in status
        assert "components" in status
        assert "components_present" in status
        assert "total_components" in status
        assert "is_fully_reproducible" in status
        assert "is_partially_reproducible" in status
        assert "missing_components" in status

    def test_status_green_full_reproducible(self):
        """Test status for fully reproducible manifest."""
        manifest = _make_mock_manifest_full()
        status = get_reproducibility_status(manifest)
        
        assert status["level"] == "green"
        assert status["is_fully_reproducible"] is True
        assert status["is_partially_reproducible"] is False
        assert status["components_present"] == 4
        assert len(status["missing_components"]) == 0

    def test_status_yellow_partial_reproducible(self):
        """Test status for partially reproducible manifest."""
        manifest = _make_mock_manifest_partial()
        status = get_reproducibility_status(manifest)
        
        assert status["level"] == "yellow"
        assert status["is_fully_reproducible"] is False
        assert status["is_partially_reproducible"] is True
        assert status["components_present"] == 3
        assert "env_hash" in status["missing_components"]

    def test_status_red_not_reproducible(self):
        """Test status for non-reproducible manifest."""
        manifest = _make_mock_manifest_none()
        status = get_reproducibility_status(manifest)
        
        assert status["level"] == "red"
        assert status["is_fully_reproducible"] is False
        assert status["is_partially_reproducible"] is False
        assert status["components_present"] < 4
        assert len(status["missing_components"]) > 0

    def test_status_total_components_is_four(self):
        """Test that total components is always 4."""
        manifest = _make_mock_manifest_full()
        status = get_reproducibility_status(manifest)
        
        assert status["total_components"] == 4

    def test_status_components_dict_complete(self):
        """Test that components dict has all keys."""
        manifest = _make_mock_manifest_full()
        status = get_reproducibility_status(manifest)
        
        components = status["components"]
        assert "seed" in components
        assert "protocol_hash" in components
        assert "data_hash" in components
        assert "env_hash" in components


class TestBadgeIntegration:
    """Integration tests for badge generation."""

    def test_full_workflow_green(self):
        """Test complete workflow for green badge."""
        manifest = _make_mock_manifest_full()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            status = get_reproducibility_status(manifest)
            fig = plot_reproducibility_badge(manifest, save_path=tmpdir)
            
            assert status["level"] == "green"
            assert (tmpdir / "reproducibility_badge.png").exists()
            
            plt.close(fig)

    def test_full_workflow_yellow(self):
        """Test complete workflow for yellow badge."""
        manifest = _make_mock_manifest_partial()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            status = get_reproducibility_status(manifest)
            fig = plot_reproducibility_badge(manifest, save_path=tmpdir)
            
            assert status["level"] == "yellow"
            assert (tmpdir / "reproducibility_badge.png").exists()
            
            plt.close(fig)

    def test_full_workflow_red(self):
        """Test complete workflow for red badge."""
        manifest = _make_mock_manifest_none()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            status = get_reproducibility_status(manifest)
            fig = plot_reproducibility_badge(manifest, save_path=tmpdir)
            
            assert status["level"] == "red"
            assert (tmpdir / "reproducibility_badge.png").exists()
            
            plt.close(fig)

    def test_multiple_badges_in_sequence(self):
        """Test generating multiple badges in sequence."""
        manifests = [
            _make_mock_manifest_full(),
            _make_mock_manifest_partial(),
            _make_mock_manifest_none(),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            for i, manifest in enumerate(manifests):
                output_dir = tmpdir / f"badge_{i}"
                fig = plot_reproducibility_badge(manifest, save_path=output_dir)
                
                assert (output_dir / "reproducibility_badge.png").exists()
                plt.close(fig)
