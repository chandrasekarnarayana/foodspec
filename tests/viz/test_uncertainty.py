"""
Tests for uncertainty quantification visualization module.

Covers confidence maps, set size distributions, coverage-efficiency trade-offs,
and abstention patterns.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from foodspec.viz.uncertainty import (
    _get_confidence_class,
    _normalize_confidences,
    _sort_by_confidence,
    _validate_confidence_array,
    get_uncertainty_statistics,
    plot_abstention_distribution,
    plot_confidence_map,
    plot_coverage_efficiency,
    plot_set_size_distribution,
)


class TestValidateConfidenceArray:
    """Test confidence array validation."""

    def test_valid_array(self):
        """Test valid array."""
        confidences = np.array([0.1, 0.5, 0.9])
        length = _validate_confidence_array(confidences)
        assert length == 3

    def test_empty_array(self):
        """Test empty array."""
        confidences = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_confidence_array(confidences)

    def test_wrong_dimensions(self):
        """Test wrong dimensions."""
        confidences = np.array([[0.1, 0.5], [0.9, 0.2]])
        with pytest.raises(ValueError, match="1D array"):
            _validate_confidence_array(confidences)


class TestNormalizeConfidences:
    """Test confidence normalization."""

    def test_valid_range(self):
        """Test confidences already in [0, 1]."""
        confidences = np.array([0.1, 0.5, 0.9])
        result = _normalize_confidences(confidences)
        assert np.allclose(result, confidences)

    def test_clip_exceeding(self):
        """Test clipping values outside [0, 1]."""
        confidences = np.array([-0.1, 0.5, 1.5])
        result = _normalize_confidences(confidences)
        assert np.allclose(result, [0.0, 0.5, 1.0])

    def test_data_type(self):
        """Test output is float."""
        confidences = np.array([1, 0, 1], dtype=int)
        result = _normalize_confidences(confidences)
        assert result.dtype == float


class TestSortByConfidence:
    """Test confidence sorting."""

    def test_descending(self):
        """Test descending sort."""
        confidences = np.array([0.1, 0.9, 0.5])
        indices = _sort_by_confidence(confidences, descending=True)
        assert np.array_equal(indices, [1, 2, 0])  # 0.9, 0.5, 0.1

    def test_ascending(self):
        """Test ascending sort."""
        confidences = np.array([0.1, 0.9, 0.5])
        indices = _sort_by_confidence(confidences, descending=False)
        assert np.array_equal(indices, [0, 2, 1])  # 0.1, 0.5, 0.9


class TestGetConfidenceClass:
    """Test confidence classification."""

    def test_basic_thresholds(self):
        """Test classification with default thresholds."""
        thresholds = [0.5, 0.7, 0.9]
        assert _get_confidence_class(0.3, thresholds) == "[0]"
        assert _get_confidence_class(0.6, thresholds) == "[1]"
        assert _get_confidence_class(0.8, thresholds) == "[2]"
        assert _get_confidence_class(0.95, thresholds) == "[3]"


class TestPlotConfidenceMapBasics:
    """Test basic confidence map functionality."""

    def test_returns_figure(self):
        """Test returns figure."""
        confidences = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        fig = plot_confidence_map(confidences)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_class_predictions(self):
        """Test with class predictions."""
        confidences = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        class_predictions = np.array([0, 1, 0, 1, 2])
        fig = plot_confidence_map(confidences, class_predictions=class_predictions)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_sample_labels(self):
        """Test with sample labels."""
        confidences = np.array([0.5, 0.6, 0.7])
        labels = ["Sample A", "Sample B", "Sample C"]
        fig = plot_confidence_map(confidences, sample_labels=labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_sorting(self):
        """Test without sorting."""
        confidences = np.array([0.5, 0.6, 0.7])
        fig = plot_confidence_map(confidences, sort_by_confidence=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_thresholds(self):
        """Test custom thresholds."""
        confidences = np.array([0.5, 0.6, 0.7])
        fig = plot_confidence_map(confidences, confidence_thresholds=[0.6, 0.8])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_colormaps(self):
        """Test different colormaps."""
        confidences = np.array([0.5, 0.6, 0.7])
        for cmap in ["RdYlGn", "viridis", "plasma"]:
            fig = plot_confidence_map(confidences, colormap=cmap)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_without_values(self):
        """Test without value annotations."""
        confidences = np.array([0.5, 0.6, 0.7])
        fig = plot_confidence_map(confidences, show_values=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self):
        """Test custom title."""
        confidences = np.array([0.5, 0.6, 0.7])
        fig = plot_confidence_map(confidences, title="My Confidence")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_file_save(self):
        """Test file saving."""
        confidences = np.array([0.5, 0.6, 0.7])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "confidence.png"
            fig = plot_confidence_map(confidences, save_path=save_path)
            assert save_path.exists()
            plt.close(fig)


class TestPlotConfidenceMapValidation:
    """Test confidence map validation."""

    def test_mismatched_class_predictions(self):
        """Test mismatched class predictions."""
        confidences = np.array([0.5, 0.6, 0.7])
        class_predictions = np.array([0, 1])  # Only 2, need 3
        with pytest.raises(ValueError, match="class_predictions"):
            plot_confidence_map(confidences, class_predictions=class_predictions)

    def test_mismatched_sample_labels(self):
        """Test mismatched sample labels."""
        confidences = np.array([0.5, 0.6, 0.7])
        labels = ["A", "B"]  # Only 2, need 3
        with pytest.raises(ValueError, match="sample_labels"):
            plot_confidence_map(confidences, sample_labels=labels)


class TestPlotSetSizeDistributionBasics:
    """Test set size distribution functionality."""

    def test_returns_figure(self):
        """Test returns figure."""
        set_sizes = np.array([1, 1, 2, 2, 3, 3])
        fig = plot_set_size_distribution(set_sizes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_batch_labels(self):
        """Test with batch labels."""
        set_sizes = np.array([1, 1, 2, 2, 3, 3])
        batch_labels = np.array([0, 0, 1, 1, 0, 1])
        fig = plot_set_size_distribution(set_sizes, batch_labels=batch_labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_stage_labels(self):
        """Test with stage labels."""
        set_sizes = np.array([1, 1, 2, 2, 3, 3])
        stage_labels = np.array(["A", "A", "B", "B", "A", "B"])
        fig = plot_set_size_distribution(set_sizes, stage_labels=stage_labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_violin(self):
        """Test without violin plot."""
        set_sizes = np.array([1, 1, 2, 2, 3, 3])
        batch_labels = np.array([0, 0, 1, 1, 0, 1])
        fig = plot_set_size_distribution(
            set_sizes, batch_labels=batch_labels, show_violin=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_box(self):
        """Test without box plot."""
        set_sizes = np.array([1, 1, 2, 2, 3, 3])
        batch_labels = np.array([0, 0, 1, 1, 0, 1])
        fig = plot_set_size_distribution(
            set_sizes, batch_labels=batch_labels, show_box=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_file_save(self):
        """Test file saving."""
        set_sizes = np.array([1, 1, 2, 2, 3, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "setsize.png"
            fig = plot_set_size_distribution(set_sizes, save_path=save_path)
            assert save_path.exists()
            plt.close(fig)


class TestPlotSetSizeDistributionValidation:
    """Test set size distribution validation."""

    def test_empty_array(self):
        """Test empty array."""
        set_sizes = np.array([], dtype=int)
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_set_size_distribution(set_sizes)

    def test_wrong_dimensions(self):
        """Test wrong dimensions."""
        set_sizes = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1D"):
            plot_set_size_distribution(set_sizes)

    def test_mismatched_batch_labels(self):
        """Test mismatched batch labels."""
        set_sizes = np.array([1, 1, 2, 2])
        batch_labels = np.array([0, 0, 1])  # Only 3, need 4
        with pytest.raises(ValueError, match="batch_labels"):
            plot_set_size_distribution(set_sizes, batch_labels=batch_labels)


class TestPlotCoverageEfficiencyBasics:
    """Test coverage-efficiency plot functionality."""

    def test_returns_figure(self):
        """Test returns figure."""
        alphas = np.array([0.05, 0.1, 0.2, 0.3])
        coverages = np.array([0.95, 0.90, 0.85, 0.80])
        avg_sizes = np.array([1.5, 1.2, 1.0, 0.8])
        fig = plot_coverage_efficiency(alphas, coverages, avg_sizes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_target_coverage(self):
        """Test custom target coverage."""
        alphas = np.array([0.05, 0.1, 0.2])
        coverages = np.array([0.95, 0.90, 0.85])
        avg_sizes = np.array([1.5, 1.2, 1.0])
        fig = plot_coverage_efficiency(
            alphas, coverages, avg_sizes, target_coverage=0.95
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_colormaps(self):
        """Test different colormaps."""
        alphas = np.array([0.05, 0.1, 0.2])
        coverages = np.array([0.95, 0.90, 0.85])
        avg_sizes = np.array([1.5, 1.2, 1.0])
        for cmap in ["viridis", "plasma", "coolwarm"]:
            fig = plot_coverage_efficiency(
                alphas, coverages, avg_sizes, colormap=cmap
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_file_save(self):
        """Test file saving."""
        alphas = np.array([0.05, 0.1, 0.2])
        coverages = np.array([0.95, 0.90, 0.85])
        avg_sizes = np.array([1.5, 1.2, 1.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "coverage.png"
            fig = plot_coverage_efficiency(
                alphas, coverages, avg_sizes, save_path=save_path
            )
            assert save_path.exists()
            plt.close(fig)


class TestPlotCoverageEfficiencyValidation:
    """Test coverage-efficiency validation."""

    def test_empty_arrays(self):
        """Test empty arrays."""
        alphas = np.array([])
        coverages = np.array([])
        avg_sizes = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_coverage_efficiency(alphas, coverages, avg_sizes)

    def test_mismatched_lengths(self):
        """Test mismatched lengths."""
        alphas = np.array([0.05, 0.1])
        coverages = np.array([0.95, 0.90, 0.85])
        avg_sizes = np.array([1.5, 1.2])
        with pytest.raises(ValueError, match="matching lengths"):
            plot_coverage_efficiency(alphas, coverages, avg_sizes)


class TestPlotAbstentionDistributionBasics:
    """Test abstention distribution functionality."""

    def test_returns_figure_overall(self):
        """Test returns figure for overall distribution."""
        abstain_flags = np.array([0, 0, 1, 0, 1, 1])
        fig = plot_abstention_distribution(abstain_flags)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_class_labels(self):
        """Test with class labels."""
        abstain_flags = np.array([0, 0, 1, 0, 1, 1])
        class_labels = np.array([0, 1, 0, 1, 0, 1])
        fig = plot_abstention_distribution(
            abstain_flags, class_labels=class_labels
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_batch_labels(self):
        """Test with batch labels."""
        abstain_flags = np.array([0, 0, 1, 0, 1, 1])
        batch_labels = np.array(["A", "A", "B", "B", "A", "B"])
        fig = plot_abstention_distribution(
            abstain_flags, batch_labels=batch_labels
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_table(self):
        """Test without summary table."""
        abstain_flags = np.array([0, 0, 1, 0, 1, 1])
        class_labels = np.array([0, 1, 0, 1, 0, 1])
        fig = plot_abstention_distribution(
            abstain_flags, class_labels=class_labels, show_table=False
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_file_save(self):
        """Test file saving."""
        abstain_flags = np.array([0, 0, 1, 0, 1, 1])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "abstention.png"
            fig = plot_abstention_distribution(
                abstain_flags, save_path=save_path
            )
            assert save_path.exists()
            plt.close(fig)


class TestPlotAbstentionDistributionValidation:
    """Test abstention distribution validation."""

    def test_empty_array(self):
        """Test empty array."""
        abstain_flags = np.array([], dtype=int)
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_abstention_distribution(abstain_flags)

    def test_wrong_dimensions(self):
        """Test wrong dimensions."""
        abstain_flags = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="1D"):
            plot_abstention_distribution(abstain_flags)

    def test_mismatched_class_labels(self):
        """Test mismatched class labels."""
        abstain_flags = np.array([0, 0, 1, 0, 1, 1])
        class_labels = np.array([0, 1, 0])  # Only 3, need 6
        with pytest.raises(ValueError, match="class_labels"):
            plot_abstention_distribution(
                abstain_flags, class_labels=class_labels
            )


class TestGetUncertaintyStatistics:
    """Test statistics extraction."""

    def test_confidence_statistics(self):
        """Test confidence statistics."""
        confidences = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        stats = get_uncertainty_statistics(confidences)
        assert "confidence" in stats
        assert stats["confidence"]["mean"] == 0.7
        assert stats["confidence"]["min"] == 0.5
        assert stats["confidence"]["max"] == 0.9

    def test_with_set_sizes(self):
        """Test with set sizes."""
        confidences = np.array([0.5, 0.6, 0.7])
        set_sizes = np.array([1, 2, 3])
        stats = get_uncertainty_statistics(confidences, set_sizes=set_sizes)
        assert "set_size" in stats
        assert stats["set_size"]["mean"] == 2.0

    def test_with_abstention(self):
        """Test with abstention flags."""
        confidences = np.array([0.5, 0.6, 0.7])
        abstain_flags = np.array([0, 1, 0])
        stats = get_uncertainty_statistics(
            confidences, abstain_flags=abstain_flags
        )
        assert "abstention" in stats
        assert stats["abstention"]["rate"] == 1/3
        assert stats["abstention"]["count"] == 1


class TestUncertaintyIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test complete workflow with all visualizations."""
        np.random.seed(42)
        n_samples = 100

        # Generate synthetic data
        confidences = np.random.beta(5, 2, n_samples)
        class_predictions = np.random.randint(0, 3, n_samples)
        set_sizes = np.random.poisson(2, n_samples) + 1
        abstain_flags = np.random.binomial(1, 0.3, n_samples)
        batch_labels = np.random.randint(0, 3, n_samples)

        # Test all visualizations
        plot_confidence_map(confidences, class_predictions=class_predictions)
        plot_set_size_distribution(set_sizes, batch_labels=batch_labels)

        alphas = np.linspace(0.05, 0.3, 5)
        coverages = 1 - alphas
        avg_sizes = alphas * 5 + 1
        plot_coverage_efficiency(alphas, coverages, avg_sizes)

        plot_abstention_distribution(
            abstain_flags, class_labels=class_predictions
        )

        # Get statistics
        stats = get_uncertainty_statistics(
            confidences, set_sizes=set_sizes, abstain_flags=abstain_flags
        )

        assert len(stats) == 3
        plt.close("all")


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_single_sample(self):
        """Test with single sample."""
        confidences = np.array([0.7])
        fig = plot_confidence_map(confidences)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_same_confidence(self):
        """Test with all same confidences."""
        confidences = np.ones(10) * 0.5
        fig = plot_confidence_map(confidences)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_extreme_set_sizes(self):
        """Test with extreme set sizes."""
        set_sizes = np.array([1, 1, 1, 100, 100, 100])
        fig = plot_set_size_distribution(set_sizes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_abstain(self):
        """Test when all samples abstain."""
        abstain_flags = np.ones(10, dtype=int)
        fig = plot_abstention_distribution(abstain_flags)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_abstain(self):
        """Test when no samples abstain."""
        abstain_flags = np.zeros(10, dtype=int)
        fig = plot_abstention_distribution(abstain_flags)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
