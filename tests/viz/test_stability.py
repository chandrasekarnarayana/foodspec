"""
Tests for feature stability visualization module.

Covers stability heatmaps, bar summaries, clustering, and validation.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from foodspec.viz.stability import (
    _compute_feature_frequency,
    _normalize_stability,
    _sort_by_stability,
    _validate_stability_matrix,
    get_stability_statistics,
    plot_feature_stability,
)


class TestValidateStabilityMatrix:
    """Test matrix validation."""

    def test_valid_matrix(self):
        """Test valid matrix."""
        matrix = np.array([[1, 0, 1], [0, 1, 0]])
        n_features, n_folds = _validate_stability_matrix(matrix)
        assert n_features == 2
        assert n_folds == 3

    def test_empty_matrix(self):
        """Test empty matrix."""
        matrix = np.array([]).reshape(0, 0)
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_stability_matrix(matrix)

    def test_wrong_dimensions(self):
        """Test wrong dimensions."""
        matrix = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="2D"):
            _validate_stability_matrix(matrix)


class TestNormalizeStability:
    """Test stability normalization."""

    def test_frequency_normalization(self):
        """Test frequency normalization."""
        matrix = np.array([[1, 2, 3], [2, 4, 1]], dtype=float)
        normalized = _normalize_stability(matrix, method="frequency")
        assert np.all(normalized >= 0) and np.all(normalized <= 1)

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        matrix = np.array([[1, 2, 3], [2, 4, 1]], dtype=float)
        normalized = _normalize_stability(matrix, method="minmax")
        assert np.all(normalized >= 0) and np.all(normalized <= 1)

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        matrix = np.array([[1, 2, 3], [2, 4, 1]], dtype=float)
        normalized = _normalize_stability(matrix, method="zscore")
        assert np.all(normalized >= 0) and np.all(normalized <= 1)

    def test_empty_array(self):
        """Test empty array."""
        matrix = np.array([]).reshape(0, 0)
        result = _normalize_stability(matrix, method="frequency")
        assert result.size == 0


class TestComputeFeatureFrequency:
    """Test frequency computation."""

    def test_basic_frequency(self):
        """Test frequency computation."""
        matrix = np.array([[1, 1, 0], [0, 1, 1]])
        frequencies = _compute_feature_frequency(matrix)
        assert np.allclose(frequencies, [2/3, 2/3])

    def test_all_selected(self):
        """Test all features selected."""
        matrix = np.array([[1, 1, 1], [1, 1, 1]])
        frequencies = _compute_feature_frequency(matrix)
        assert np.allclose(frequencies, [1.0, 1.0])

    def test_none_selected(self):
        """Test no features selected."""
        matrix = np.array([[0, 0, 0], [0, 0, 0]])
        frequencies = _compute_feature_frequency(matrix)
        assert np.allclose(frequencies, [0.0, 0.0])


class TestSortByStability:
    """Test stability sorting."""

    def test_sort_by_frequency(self):
        """Test sorting by frequency."""
        matrix = np.array([[1, 0, 0], [1, 1, 1], [1, 1, 0]])
        indices = _sort_by_stability(matrix, method="frequency")
        # Frequencies: [1/3, 1, 2/3]
        # Should be sorted ascending: [0, 2, 1]
        assert indices[0] == 0  # Lowest frequency first

    def test_sort_by_std(self):
        """Test sorting by std."""
        matrix = np.array([[1, 0, 0], [1, 1, 1], [1, 1, 0]])
        indices = _sort_by_stability(matrix, method="std")
        assert len(indices) == 3

    def test_no_sorting(self):
        """Test no sorting."""
        matrix = np.array([[1, 0, 0], [0, 1, 1]])
        indices = _sort_by_stability(matrix, method="none")
        assert np.array_equal(indices, [0, 1])


class TestPlotFeatureStabilityBasics:
    """Test basic stability plot functionality."""

    def test_returns_figure(self):
        """Test returns figure."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_fold_names(self):
        """Test with fold names."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fold_names = ["Train", "Test", "Val"]
        fig = plot_feature_stability(matrix, fold_names=fold_names)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_feature_names(self):
        """Test with feature names."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        feature_names = ["Feat A", "Feat B"]
        fig = plot_feature_stability(matrix, feature_names=feature_names)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_bar_summary(self):
        """Test with bar summary."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fig = plot_feature_stability(matrix, show_bar_summary=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_bar_summary(self):
        """Test without bar summary."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fig = plot_feature_stability(matrix, show_bar_summary=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar_position_right(self):
        """Test bar on right."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fig = plot_feature_stability(matrix, bar_position="right")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar_position_bottom(self):
        """Test bar on bottom."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fig = plot_feature_stability(matrix, bar_position="bottom")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_normalization(self):
        """Test with normalization."""
        matrix = np.array([[1, 2, 3], [2, 4, 1]])
        fig = plot_feature_stability(matrix, normalize="frequency")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_sorting(self):
        """Test with feature sorting."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fig = plot_feature_stability(matrix, sort_features="frequency")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_colormaps(self):
        """Test different colormaps."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        for cmap in ["RdYlGn", "YlOrRd", "viridis"]:
            fig = plot_feature_stability(matrix, colormap=cmap)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_with_values_shown(self):
        """Test showing values."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fig = plot_feature_stability(matrix, show_values=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self):
        """Test custom title."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        title = "My Stability Plot"
        fig = plot_feature_stability(matrix, title=title)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figure_size(self):
        """Test custom figure size."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fig = plot_feature_stability(matrix, figure_size=(10, 6))
        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close(fig)


class TestPlotFeatureStabilityValidation:
    """Test input validation."""

    def test_empty_matrix(self):
        """Test empty matrix."""
        matrix = np.array([]).reshape(0, 0)
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_feature_stability(matrix)

    def test_wrong_dimensions(self):
        """Test wrong dimensions."""
        matrix = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="2D"):
            plot_feature_stability(matrix)

    def test_mismatched_fold_names(self):
        """Test mismatched fold names."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        fold_names = ["A", "B"]  # Need 3
        with pytest.raises(ValueError, match="fold_names length"):
            plot_feature_stability(matrix, fold_names=fold_names)

    def test_mismatched_feature_names(self):
        """Test mismatched feature names."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        feature_names = ["A"]  # Need 2
        with pytest.raises(ValueError, match="feature_names length"):
            plot_feature_stability(matrix, feature_names=feature_names)

    def test_file_save(self):
        """Test file saving."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "stability.png"
            fig = plot_feature_stability(matrix, save_path=save_path)
            assert save_path.exists()
            plt.close(fig)


class TestGetStabilityStatistics:
    """Test statistics extraction."""

    def test_basic_statistics(self):
        """Test basic statistics."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        stats = get_stability_statistics(matrix)
        assert "per_feature" in stats
        assert "global" in stats
        assert "rankings" in stats
        assert "consistency_metrics" in stats

    def test_global_stats(self):
        """Test global statistics."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        stats = get_stability_statistics(matrix)
        assert "mean_frequency" in stats["global"]
        assert "min_frequency" in stats["global"]
        assert "max_frequency" in stats["global"]

    def test_per_feature_stats(self):
        """Test per-feature statistics."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        stats = get_stability_statistics(matrix)
        assert len(stats["per_feature"]) == 2
        for feat_key in stats["per_feature"]:
            assert "frequency" in stats["per_feature"][feat_key]
            assert "appearances" in stats["per_feature"][feat_key]

    def test_rankings(self):
        """Test feature rankings."""
        matrix = np.array([[1, 1, 1], [0, 1, 0]])
        stats = get_stability_statistics(matrix)
        rankings = stats["rankings"]["by_frequency"]
        assert len(rankings) == 2
        # First feature appears 3 times, should be ranked higher
        assert rankings[0]["frequency"] >= rankings[1]["frequency"]

    def test_consistency_metrics(self):
        """Test consistency metrics."""
        matrix = np.array([[1, 1, 1], [0, 1, 0]])
        stats = get_stability_statistics(matrix)
        assert "stable_features" in stats["consistency_metrics"]
        assert "unstable_features" in stats["consistency_metrics"]


class TestPlotFeatureStabilityIntegration:
    """Integration tests."""

    def test_full_workflow_5fold(self):
        """Test complete workflow with 5-fold CV."""
        # Simulate 5-fold CV with 10 features
        np.random.seed(42)
        stability_matrix = np.random.randint(0, 2, (10, 5))

        feature_names = [f"Feature {i}" for i in range(10)]
        fold_names = [f"Fold {i}" for i in range(5)]

        fig = plot_feature_stability(
            stability_matrix,
            fold_names=fold_names,
            feature_names=feature_names,
            normalize="frequency",
            sort_features="frequency",
            show_bar_summary=True,
            colormap="RdYlGn",
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_normalization_methods(self):
        """Test all normalization methods."""
        matrix = np.array([[1, 2, 3], [2, 4, 1]])
        for method in ["frequency", "minmax", "zscore"]:
            fig = plot_feature_stability(matrix, normalize=method)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_all_sorting_methods(self):
        """Test all sorting methods."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        for method in ["frequency", "std", "consistency"]:
            fig = plot_feature_stability(matrix, sort_features=method)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_with_clustering(self):
        """Test with hierarchical clustering."""
        matrix = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]])
        fig = plot_feature_stability(matrix, cluster_features=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestStabilityMatrixTypes:
    """Test different input types."""

    def test_binary_matrix(self):
        """Test binary matrix."""
        matrix = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int32)
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_count_matrix(self):
        """Test count matrix."""
        matrix = np.array([[3, 0, 2], [0, 5, 1]], dtype=np.int32)
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_float_matrix(self):
        """Test float matrix (frequencies)."""
        matrix = np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.5]], dtype=float)
        fig = plot_feature_stability(matrix, normalize="none")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestLargeStabilityMatrices:
    """Test with larger datasets."""

    def test_many_features(self):
        """Test with many features."""
        matrix = np.random.randint(0, 2, (50, 5))
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_many_folds(self):
        """Test with many folds."""
        matrix = np.random.randint(0, 2, (10, 20))
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_matrix(self):
        """Test with large matrix."""
        matrix = np.random.randint(0, 2, (100, 10))
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestStabilityEdgeCases:
    """Test edge cases."""

    def test_single_feature(self):
        """Test with single feature."""
        matrix = np.array([[1, 0, 1]])
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_fold(self):
        """Test with single fold."""
        matrix = np.array([[1], [0], [1]])
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_constant_stability(self):
        """Test with constant selection."""
        matrix = np.ones((5, 3))
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_zero_stability(self):
        """Test with zero selection."""
        matrix = np.zeros((5, 3))
        fig = plot_feature_stability(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
