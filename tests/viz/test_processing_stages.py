"""
Tests for processing stages visualization module.

Tests cover multi-stage overlay, preprocessing names, zoom windows,
comparison plots, and statistics extraction.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from foodspec.viz.processing_stages import (
    _extract_zoom_regions,
    _get_stage_colors,
    _validate_spectral_stages,
    _validate_wavenumbers,
    get_processing_statistics,
    plot_preprocessing_comparison,
    plot_processing_stages,
)


class TestValidateWavenumbers:
    """Test wavenumber validation."""

    def test_valid_array(self):
        """Test with valid 1D array."""
        wavenumbers = np.linspace(400, 4000, 1000)
        result = _validate_wavenumbers(wavenumbers)
        assert result == 1000

    def test_empty_array(self):
        """Test with empty array."""
        wavenumbers = np.array([])
        with pytest.raises(ValueError, match="empty"):
            _validate_wavenumbers(wavenumbers)

    def test_wrong_dimensions(self):
        """Test with 2D array."""
        wavenumbers = np.ones((100, 100))
        with pytest.raises(ValueError, match="1D"):
            _validate_wavenumbers(wavenumbers)

    def test_non_numeric(self):
        """Test with non-numeric array."""
        wavenumbers = np.array(["a", "b", "c"])
        with pytest.raises(ValueError, match="numeric"):
            _validate_wavenumbers(wavenumbers)


class TestValidateSpectralStages:
    """Test spectral stages validation."""

    def test_valid_stages(self):
        """Test with valid stage data."""
        stages = {
            "raw": np.random.rand(100),
            "processed": np.random.rand(100),
        }
        result = _validate_spectral_stages(stages, 100)
        assert result == 2

    def test_empty_stages(self):
        """Test with empty stages dictionary."""
        with pytest.raises(ValueError, match="empty"):
            _validate_spectral_stages({}, 100)

    def test_length_mismatch(self):
        """Test with mismatched spectrum length."""
        stages = {
            "raw": np.random.rand(100),
            "processed": np.random.rand(50),
        }
        with pytest.raises(ValueError, match="length"):
            _validate_spectral_stages(stages, 100)

    def test_wrong_dimensions(self):
        """Test with 2D spectrum."""
        stages = {
            "raw": np.random.rand(100, 100),
        }
        with pytest.raises(ValueError, match="1D"):
            _validate_spectral_stages(stages, 100)


class TestGetStageColors:
    """Test stage color generation."""

    def test_single_stage(self):
        """Test color generation for single stage."""
        colors = _get_stage_colors(1, "viridis")
        assert len(colors) == 1
        assert all(0 <= c <= 1 for c in colors[0])

    def test_multiple_stages(self):
        """Test color generation for multiple stages."""
        colors = _get_stage_colors(5, "viridis")
        assert len(colors) == 5
        # Check that colors are different
        assert len(set(colors)) == 5

    def test_different_colormaps(self):
        """Test with different colormaps."""
        for cmap in ["viridis", "plasma", "coolwarm"]:
            colors = _get_stage_colors(3, cmap)
            assert len(colors) == 3


class TestExtractZoomRegions:
    """Test zoom region extraction."""

    def test_no_zoom_regions(self):
        """Test with no zoom regions."""
        wavenumbers = np.linspace(400, 4000, 1000)
        result = _extract_zoom_regions(wavenumbers, None)
        assert result == []

    def test_single_zoom_region(self):
        """Test with single zoom region."""
        wavenumbers = np.linspace(400, 4000, 1000)
        regions = [(1000, 1200)]
        result = _extract_zoom_regions(wavenumbers, regions)
        assert len(result) == 1
        assert all(isinstance(idx, (int, np.integer)) for idx in result[0])

    def test_multiple_zoom_regions(self):
        """Test with multiple zoom regions."""
        wavenumbers = np.linspace(400, 4000, 1000)
        regions = [(1000, 1200), (2500, 2800), (3200, 3500)]
        result = _extract_zoom_regions(wavenumbers, regions)
        assert len(result) == 3

    def test_too_many_zoom_regions(self):
        """Test with more than 3 zoom regions."""
        wavenumbers = np.linspace(400, 4000, 1000)
        regions = [(500, 600), (1000, 1100), (2000, 2100), (3000, 3100)]
        with pytest.raises(ValueError, match="Maximum 3"):
            _extract_zoom_regions(wavenumbers, regions)

    def test_invalid_zoom_region(self):
        """Test with invalid zoom region (min >= max)."""
        wavenumbers = np.linspace(400, 4000, 1000)
        regions = [(1200, 1000)]
        with pytest.raises(ValueError, match="Invalid zoom region"):
            _extract_zoom_regions(wavenumbers, regions)


class TestPlotProcessingStagesBasics:
    """Test basic plot_processing_stages functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectral data."""
        wavenumbers = np.linspace(400, 4000, 500)
        raw = np.sin(wavenumbers / 300) + np.random.normal(0, 0.05, 500)
        baseline = raw - np.mean(raw)
        normalized = baseline / np.std(baseline)

        return {
            "wavenumbers": wavenumbers,
            "stages": {"raw": raw, "baseline": baseline, "normalized": normalized},
            "names": ["Raw", "Baseline Corrected", "Normalized"],
        }

    def test_returns_figure(self, sample_data):
        """Test that function returns Figure."""
        import matplotlib.pyplot as plt

        fig = plot_processing_stages(
            sample_data["wavenumbers"], sample_data["stages"]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_stage_names(self, sample_data):
        """Test with custom stage names."""
        import matplotlib.pyplot as plt

        fig = plot_processing_stages(
            sample_data["wavenumbers"],
            sample_data["stages"],
            stage_names=sample_data["names"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_zoom_regions(self, sample_data):
        """Test with zoom regions."""
        import matplotlib.pyplot as plt

        fig = plot_processing_stages(
            sample_data["wavenumbers"],
            sample_data["stages"],
            zoom_regions=[(1000, 1200), (2800, 3000)],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_zoom_region(self, sample_data):
        """Test with single zoom region."""
        import matplotlib.pyplot as plt

        fig = plot_processing_stages(
            sample_data["wavenumbers"],
            sample_data["stages"],
            zoom_regions=[(1500, 1800)],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_three_zoom_regions(self, sample_data):
        """Test with three zoom regions."""
        import matplotlib.pyplot as plt

        fig = plot_processing_stages(
            sample_data["wavenumbers"],
            sample_data["stages"],
            zoom_regions=[(500, 800), (1500, 1800), (3000, 3300)],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_colors(self, sample_data):
        """Test with custom colors."""
        import matplotlib.pyplot as plt

        colors = ["red", "green", "blue"]
        fig = plot_processing_stages(
            sample_data["wavenumbers"],
            sample_data["stages"],
            stage_colors=colors,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_colormaps(self, sample_data):
        """Test with different colormaps."""
        import matplotlib.pyplot as plt

        for cmap in ["viridis", "plasma", "coolwarm"]:
            fig = plot_processing_stages(
                sample_data["wavenumbers"],
                sample_data["stages"],
                colormap=cmap,
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_custom_title(self, sample_data):
        """Test with custom title."""
        import matplotlib.pyplot as plt

        title = "My Preprocessing Steps"
        fig = plot_processing_stages(
            sample_data["wavenumbers"],
            sample_data["stages"],
            title=title,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_file_save(self, sample_data):
        """Test saving to file."""
        import matplotlib.pyplot as plt

        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.png"
            fig = plot_processing_stages(
                sample_data["wavenumbers"],
                sample_data["stages"],
                save_path=save_path,
            )
            assert save_path.exists()
            assert save_path.stat().st_size > 0
            plt.close(fig)

    def test_custom_figure_size(self, sample_data):
        """Test with custom figure size."""
        import matplotlib.pyplot as plt

        fig = plot_processing_stages(
            sample_data["wavenumbers"],
            sample_data["stages"],
            figure_size=(12, 6),
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_grid(self, sample_data):
        """Test without grid."""
        import matplotlib.pyplot as plt

        fig = plot_processing_stages(
            sample_data["wavenumbers"],
            sample_data["stages"],
            show_grid=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotProcessingStagesValidation:
    """Test validation in plot_processing_stages."""

    def test_mismatched_stage_names(self):
        """Test with mismatched stage names."""
        wavenumbers = np.linspace(400, 4000, 100)
        stages = {"raw": np.random.rand(100), "processed": np.random.rand(100)}
        with pytest.raises(ValueError, match="stage_names length"):
            plot_processing_stages(
                wavenumbers, stages, stage_names=["Raw"]
            )

    def test_mismatched_colors(self):
        """Test with mismatched color count."""
        wavenumbers = np.linspace(400, 4000, 100)
        stages = {"raw": np.random.rand(100), "processed": np.random.rand(100)}
        with pytest.raises(ValueError, match="stage_colors length"):
            plot_processing_stages(
                wavenumbers, stages, stage_colors=["red"]
            )


class TestPlotPreprocessingComparison:
    """Test plot_preprocessing_comparison function."""

    @pytest.fixture
    def comparison_data(self):
        """Create before/after spectral data."""
        wavenumbers = np.linspace(400, 4000, 500)
        before = np.sin(wavenumbers / 300) + np.random.normal(0, 0.1, 500)
        after = before - np.mean(before)
        return wavenumbers, before, after

    def test_returns_figure(self, comparison_data):
        """Test that function returns Figure."""
        import matplotlib.pyplot as plt

        wavenumbers, before, after = comparison_data
        fig = plot_preprocessing_comparison(wavenumbers, before, after)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_difference(self, comparison_data):
        """Test with difference plot."""
        import matplotlib.pyplot as plt

        wavenumbers, before, after = comparison_data
        fig = plot_preprocessing_comparison(
            wavenumbers, before, after, show_difference=True
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_difference(self, comparison_data):
        """Test without difference plot."""
        import matplotlib.pyplot as plt

        wavenumbers, before, after = comparison_data
        fig = plot_preprocessing_comparison(
            wavenumbers, before, after, show_difference=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_preprocessing_name(self, comparison_data):
        """Test with custom preprocessing name."""
        import matplotlib.pyplot as plt

        wavenumbers, before, after = comparison_data
        fig = plot_preprocessing_comparison(
            wavenumbers,
            before,
            after,
            preprocessing_name="Baseline Correction",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_colors(self, comparison_data):
        """Test with custom colors."""
        import matplotlib.pyplot as plt

        wavenumbers, before, after = comparison_data
        fig = plot_preprocessing_comparison(
            wavenumbers,
            before,
            after,
            color_before="steelblue",
            color_after="coral",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_file_save(self, comparison_data):
        """Test saving to file."""
        import matplotlib.pyplot as plt

        wavenumbers, before, after = comparison_data
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "comparison.png"
            fig = plot_preprocessing_comparison(
                wavenumbers, before, after, save_path=save_path
            )
            assert save_path.exists()
            assert save_path.stat().st_size > 0
            plt.close(fig)

    def test_mismatched_before_length(self):
        """Test with mismatched before spectrum length."""
        wavenumbers = np.linspace(400, 4000, 100)
        before = np.random.rand(50)
        after = np.random.rand(100)
        with pytest.raises(ValueError, match="before_spectrum length"):
            plot_preprocessing_comparison(wavenumbers, before, after)

    def test_mismatched_after_length(self):
        """Test with mismatched after spectrum length."""
        wavenumbers = np.linspace(400, 4000, 100)
        before = np.random.rand(100)
        after = np.random.rand(50)
        with pytest.raises(ValueError, match="after_spectrum length"):
            plot_preprocessing_comparison(wavenumbers, before, after)


class TestGetProcessingStatistics:
    """Test statistics extraction."""

    def test_basic_statistics(self):
        """Test basic statistics extraction."""
        stages = {
            "raw": np.array([1, 2, 3, 4, 5]),
            "processed": np.array([2, 3, 4, 5, 6]),
        }
        stats = get_processing_statistics(stages)

        assert "raw" in stats
        assert "processed" in stats
        assert stats["raw"]["mean"] == 3.0
        assert stats["processed"]["mean"] == 4.0

    def test_statistics_keys(self):
        """Test that all expected keys are present."""
        stages = {"stage": np.random.rand(100)}
        stats = get_processing_statistics(stages)

        expected_keys = ["mean", "std", "min", "max", "median", "q25", "q75", "range"]
        for key in expected_keys:
            assert key in stats["stage"]

    def test_multiple_stages(self):
        """Test with multiple stages."""
        stages = {
            "stage1": np.random.rand(100),
            "stage2": np.random.rand(100),
            "stage3": np.random.rand(100),
        }
        stats = get_processing_statistics(stages)

        assert len(stats) == 3
        assert all(
            isinstance(stats[stage], dict) for stage in stages
        )


class TestProcessingIntegration:
    """Integration tests for processing stages module."""

    def test_full_workflow(self):
        """Test complete workflow with all features."""
        import matplotlib.pyplot as plt

        # Create data
        wavenumbers = np.linspace(400, 4000, 1000)
        raw = np.sin(wavenumbers / 250) + np.random.normal(0, 0.05, 1000)
        baseline = raw - np.mean(raw)
        normalized = baseline / np.std(baseline)

        stages_data = {
            "raw": raw,
            "baseline": baseline,
            "normalized": normalized,
        }

        # Get statistics
        stats = get_processing_statistics(stages_data)
        assert len(stats) == 3

        # Plot multi-stage
        fig1 = plot_processing_stages(
            wavenumbers,
            stages_data,
            stage_names=["Raw", "Baseline Corrected", "Normalized"],
            zoom_regions=[(1000, 1500), (2500, 3000)],
        )
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Plot comparison
        fig2 = plot_preprocessing_comparison(
            wavenumbers, raw, baseline, preprocessing_name="Baseline Correction"
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)


class TestEdgeCases:
    """Test edge cases."""

    def test_single_stage(self):
        """Test with single stage."""
        import matplotlib.pyplot as plt

        wavenumbers = np.linspace(400, 4000, 100)
        stages = {"raw": np.random.rand(100)}
        fig = plot_processing_stages(wavenumbers, stages)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_identical_spectra(self):
        """Test with identical spectra in multiple stages."""
        import matplotlib.pyplot as plt

        wavenumbers = np.linspace(400, 4000, 100)
        spectrum = np.ones(100)
        stages = {"stage1": spectrum, "stage2": spectrum}
        fig = plot_processing_stages(wavenumbers, stages)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_very_small_zoom_region(self):
        """Test with very small zoom region."""
        import matplotlib.pyplot as plt

        wavenumbers = np.linspace(400, 4000, 1000)
        stages = {"raw": np.random.rand(1000)}
        fig = plot_processing_stages(
            wavenumbers, stages, zoom_regions=[(1000, 1001)]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_spectral_array(self):
        """Test with large spectral array."""
        import matplotlib.pyplot as plt

        wavenumbers = np.linspace(400, 4000, 10000)
        stages = {"stage": np.random.rand(10000)}
        fig = plot_processing_stages(wavenumbers, stages)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
