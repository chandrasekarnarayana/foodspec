"""
Tests for coefficient heatmap visualization module.

Covers normalization, visualization, validation, and integration workflows.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from foodspec.viz.coefficients import (
    _format_coefficient_annotation,
    _normalize_coefficients,
    _sort_features_by_magnitude,
    get_coefficient_statistics,
    plot_coefficients_heatmap,
)


class TestNormalizeCoefficients:
    """Test coefficient normalization."""

    def test_standard_normalization(self):
        """Test z-score normalization."""
        coefs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = _normalize_coefficients(coefs, method="standard")
        assert normalized.shape == coefs.shape
        # Check means are close to 0
        assert np.allclose(np.mean(normalized, axis=1), 0, atol=1e-10)

    def test_minmax_normalization(self):
        """Test min-max scaling."""
        coefs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = _normalize_coefficients(coefs, method="minmax")
        assert normalized.shape == coefs.shape
        # Check range is [-1, 1]
        assert np.all(normalized >= -1) and np.all(normalized <= 1)

    def test_no_normalization(self):
        """Test no normalization."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _normalize_coefficients(coefs, method="none")
        assert np.array_equal(result, coefs)

    def test_identical_values(self):
        """Test normalization with identical values."""
        coefs = np.array([[5.0, 5.0, 5.0]])
        normalized = _normalize_coefficients(coefs, method="standard")
        # Should not raise, should handle gracefully
        assert normalized.shape == coefs.shape

    def test_empty_array(self):
        """Test empty array."""
        coefs = np.array([]).reshape(0, 0)
        result = _normalize_coefficients(coefs, method="standard")
        assert result.size == 0


class TestSortFeaturesByMagnitude:
    """Test feature sorting."""

    def test_sort_by_mean(self):
        """Test sorting by mean magnitude."""
        coefs = np.array([[1.0, 1.0], [10.0, 10.0], [5.0, 5.0]])
        indices = _sort_features_by_magnitude(coefs, method="mean")
        assert np.array_equal(indices, [0, 2, 1])  # Ascending

    def test_sort_by_max(self):
        """Test sorting by max magnitude."""
        coefs = np.array([[1.0, 2.0], [10.0, 20.0], [5.0, 8.0]])
        indices = _sort_features_by_magnitude(coefs, method="max")
        # Should sort by max: [2, 8, 20]
        assert len(indices) == 3

    def test_sort_by_norm(self):
        """Test sorting by L2 norm."""
        coefs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        indices = _sort_features_by_magnitude(coefs, method="norm")
        assert len(indices) == 3


class TestFormatCoefficientAnnotation:
    """Test annotation formatting."""

    def test_positive_value(self):
        """Test formatting positive value."""
        result = _format_coefficient_annotation(0.5, decimals=2)
        assert result == "+0.50"

    def test_negative_value(self):
        """Test formatting negative value."""
        result = _format_coefficient_annotation(-0.5, decimals=2)
        assert result == "-0.50"

    def test_zero_value(self):
        """Test formatting zero."""
        result = _format_coefficient_annotation(0.0, decimals=2)
        assert result == "+0.00"

    def test_different_decimals(self):
        """Test different decimal places."""
        result = _format_coefficient_annotation(1.23456, decimals=1)
        assert result == "+1.2"


class TestPlotCoefficientsHeatmapBasics:
    """Test basic heatmap functionality."""

    def test_returns_figure(self):
        """Test that function returns a Figure."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        fig = plot_coefficients_heatmap(coefs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_class_names(self):
        """Test with provided class names."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        class_names = ["Class A", "Class B"]
        fig = plot_coefficients_heatmap(coefs, class_names=class_names)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_feature_names(self):
        """Test with provided feature names."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        feature_names = ["Feature X", "Feature Y"]
        fig = plot_coefficients_heatmap(coefs, feature_names=feature_names)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_normalization(self):
        """Test with normalization."""
        coefs = np.array([[1.0, 10.0], [2.0, 20.0]])
        fig = plot_coefficients_heatmap(coefs, normalize="standard")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_sorting(self):
        """Test with feature sorting."""
        coefs = np.array([[1.0, 1.0], [10.0, 10.0]])
        fig = plot_coefficients_heatmap(coefs, sort_features="mean")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_colormaps(self):
        """Test different colormaps."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        for cmap in ["RdBu_r", "coolwarm", "RdYlBu_r"]:
            fig = plot_coefficients_heatmap(coefs, colormap=cmap)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_without_annotations(self):
        """Test without value annotations."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        fig = plot_coefficients_heatmap(coefs, show_values=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self):
        """Test with custom title."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        title = "Custom Title"
        fig = plot_coefficients_heatmap(coefs, title=title)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figure_size(self):
        """Test with custom figure size."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        fig = plot_coefficients_heatmap(coefs, figure_size=(8, 6))
        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 6
        plt.close(fig)


class TestPlotCoefficientsHeatmapValidation:
    """Test input validation."""

    def test_empty_coefficients(self):
        """Test with empty coefficients."""
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_coefficients_heatmap(np.array([]))

    def test_wrong_dimensions(self):
        """Test with wrong dimensions."""
        with pytest.raises(ValueError, match="2D array"):
            plot_coefficients_heatmap(np.array([1, 2, 3]))

    def test_mismatched_class_names(self):
        """Test mismatched class names."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        class_names = ["A"]  # Only 1, need 2
        with pytest.raises(ValueError, match="class_names length"):
            plot_coefficients_heatmap(coefs, class_names=class_names)

    def test_mismatched_feature_names(self):
        """Test mismatched feature names."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        feature_names = ["A"]  # Only 1, need 2
        with pytest.raises(ValueError, match="feature_names length"):
            plot_coefficients_heatmap(coefs, feature_names=feature_names)

    def test_file_save(self):
        """Test saving to file."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.png"
            fig = plot_coefficients_heatmap(coefs, save_path=save_path)
            assert save_path.exists()
            plt.close(fig)


class TestGetCoefficientStatistics:
    """Test statistics extraction."""

    def test_basic_statistics(self):
        """Test basic statistics computation."""
        coefs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stats = get_coefficient_statistics(coefs)
        assert "per_feature" in stats
        assert "per_class" in stats
        assert "global" in stats
        assert "rankings" in stats

    def test_global_stats(self):
        """Test global statistics."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = get_coefficient_statistics(coefs)
        assert stats["global"]["mean"] == 2.5
        assert stats["global"]["min"] == 1.0
        assert stats["global"]["max"] == 4.0

    def test_per_feature_stats(self):
        """Test per-feature statistics."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = get_coefficient_statistics(coefs)
        assert len(stats["per_feature"]) == 2

    def test_per_class_stats(self):
        """Test per-class statistics."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = get_coefficient_statistics(coefs)
        assert len(stats["per_class"]) == 2

    def test_rankings(self):
        """Test feature rankings."""
        coefs = np.array([[1.0, 1.0], [10.0, 10.0]])
        stats = get_coefficient_statistics(coefs)
        rankings = stats["rankings"]["by_mean_magnitude"]
        assert len(rankings) == 2
        # Second feature should be ranked first (higher magnitude)
        assert rankings[0]["rank"] == 1


class TestPlotCoefficientsIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test complete workflow."""
        # Create synthetic coefficients
        coefs = np.random.randn(10, 3) * np.array([1, 10, 5])

        class_names = ["Normal", "Class 1", "Class 2"]
        feature_names = [f"Feature {i}" for i in range(10)]

        # Plot with all options
        fig = plot_coefficients_heatmap(
            coefs,
            class_names=class_names,
            feature_names=feature_names,
            normalize="standard",
            sort_features="mean",
            colormap="RdBu_r",
            show_values=True,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_normalization_methods(self):
        """Test all normalization methods."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        for method in ["standard", "minmax"]:
            fig = plot_coefficients_heatmap(coefs, normalize=method)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_all_sorting_methods(self):
        """Test all sorting methods."""
        coefs = np.array([[1.0, 2.0], [3.0, 4.0]])
        for method in ["mean", "max", "norm"]:
            fig = plot_coefficients_heatmap(coefs, sort_features=method)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)


class TestNegativeCoefficients:
    """Test handling of negative coefficients."""

    def test_mixed_signs(self):
        """Test with mixed positive/negative coefficients."""
        coefs = np.array([[-1.0, 2.0], [3.0, -4.0]])
        fig = plot_coefficients_heatmap(coefs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_negative(self):
        """Test with all negative coefficients."""
        coefs = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        fig = plot_coefficients_heatmap(coefs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestLargeCoefficients:
    """Test with larger datasets."""

    def test_many_features(self):
        """Test with many features."""
        coefs = np.random.randn(50, 3)
        fig = plot_coefficients_heatmap(coefs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_many_classes(self):
        """Test with many classes."""
        coefs = np.random.randn(10, 20)
        fig = plot_coefficients_heatmap(coefs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
