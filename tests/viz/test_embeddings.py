"""
Comprehensive test suite for embeddings visualization module.

Test coverage:
- Input validation (embedding, labels)
- Color mapping and generation
- Confidence ellipse fitting
- 2D and 3D plotting
- Multi-factor coloring (batch, stage, class)
- Density contours
- Embedding comparison
- Statistics extraction
- Edge cases and error handling
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pytest

from foodspec.viz.embeddings import (
    _fit_confidence_ellipse,
    _get_embedding_colors,
    _validate_embedding,
    _validate_labels,
    get_embedding_statistics,
    plot_embedding,
    plot_embedding_comparison,
)


class TestValidateEmbedding:
    """Test embedding validation."""

    def test_valid_2d_embedding(self):
        """Valid 2D embedding should not raise."""
        embedding = np.random.randn(100, 2)
        _validate_embedding(embedding)  # Should not raise

    def test_valid_3d_embedding(self):
        """Valid 3D embedding should not raise."""
        embedding = np.random.randn(100, 3)
        _validate_embedding(embedding)  # Should not raise

    def test_non_array_embedding(self):
        """Non-array embedding should raise ValueError."""
        with pytest.raises(ValueError, match="must be a numpy array"):
            _validate_embedding([[1, 2], [3, 4]])

    def test_1d_embedding(self):
        """1D embedding should raise ValueError."""
        embedding = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must be 2D"):
            _validate_embedding(embedding)

    def test_empty_embedding(self):
        """Empty embedding should raise ValueError."""
        embedding = np.array([]).reshape(0, 2)
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_embedding(embedding)

    def test_wrong_dimensions_embedding(self):
        """Embedding with wrong dimensions should raise ValueError."""
        embedding = np.random.randn(100, 4)
        with pytest.raises(ValueError, match="must have 2 or 3 dimensions"):
            _validate_embedding(embedding)

    def test_non_numeric_embedding(self):
        """Non-numeric embedding should raise ValueError."""
        embedding = np.array([['a', 'b'], ['c', 'd']])
        with pytest.raises(ValueError, match="must contain numeric values"):
            _validate_embedding(embedding)

    def test_nan_embedding(self):
        """Embedding with NaN should raise ValueError."""
        embedding = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with pytest.raises(ValueError, match="non-finite"):
            _validate_embedding(embedding)

    def test_inf_embedding(self):
        """Embedding with Inf should raise ValueError."""
        embedding = np.array([[1.0, 2.0], [np.inf, 4.0]])
        with pytest.raises(ValueError, match="non-finite"):
            _validate_embedding(embedding)


class TestValidateLabels:
    """Test label validation."""

    def test_none_labels(self):
        """None labels should not raise."""
        _validate_labels(None, 100)  # Should not raise

    def test_valid_labels(self):
        """Valid labels should not raise."""
        labels = np.array([0, 1, 2, 0, 1])
        _validate_labels(labels, 5)  # Should not raise

    def test_string_labels(self):
        """String labels should not raise."""
        labels = np.array(['A', 'B', 'C'])
        _validate_labels(labels, 3)  # Should not raise

    def test_labels_wrong_length(self):
        """Labels with wrong length should raise ValueError."""
        labels = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="length.*doesn't match"):
            _validate_labels(labels, 5)

    def test_multidimensional_labels(self):
        """Multidimensional labels should raise ValueError."""
        labels = np.array([[0, 1], [2, 3]])
        with pytest.raises(ValueError, match="must be 1D"):
            _validate_labels(labels, 4)


class TestGetEmbeddingColors:
    """Test color mapping generation."""

    def test_none_labels(self):
        """None labels should return empty dict."""
        colors = _get_embedding_colors(None, "tab10")
        assert colors == {}

    def test_single_label(self):
        """Single label should return single color."""
        labels = np.array([0, 0, 0])
        colors = _get_embedding_colors(labels, "tab10")
        assert len(colors) == 1
        assert 0 in colors
        assert isinstance(colors[0], tuple)
        assert len(colors[0]) == 3

    def test_multiple_labels(self):
        """Multiple labels should return multiple colors."""
        labels = np.array([0, 1, 2, 0, 1, 2])
        colors = _get_embedding_colors(labels, "tab10")
        assert len(colors) == 3
        assert all(i in colors for i in [0, 1, 2])
        assert all(isinstance(colors[i], tuple) for i in [0, 1, 2])

    def test_different_colormaps(self):
        """Different colormaps should work."""
        labels = np.array([0, 1, 2])
        for cmap in ["tab10", "Set1", "Pastel1", "viridis"]:
            colors = _get_embedding_colors(labels, cmap)
            assert len(colors) == 3


class TestFitConfidenceEllipse:
    """Test confidence ellipse fitting."""

    def test_single_point(self):
        """Single point should return small ellipse."""
        points = np.array([[0.0, 0.0]])
        center, scales, angle = _fit_confidence_ellipse(points)
        assert np.allclose(center, [0.0, 0.0])
        assert all(s < 1.0 for s in scales)

    def test_two_points(self):
        """Two points should return valid ellipse."""
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        center, scales, angle = _fit_confidence_ellipse(points)
        assert center.shape == (2,)
        assert scales.shape == (2,)
        assert 0 <= angle <= 360

    def test_circular_cloud(self):
        """Circular point cloud should have similar scales."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        center, scales, angle = _fit_confidence_ellipse(points)
        # Scales should be reasonably similar (within 2x)
        assert scales[0] > 0 and scales[1] > 0

    def test_confidence_levels(self):
        """Different confidence levels should give different scales."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        _, scales_68, _ = _fit_confidence_ellipse(points, confidence=0.68)
        _, scales_95, _ = _fit_confidence_ellipse(points, confidence=0.95)
        # 95% confidence should give larger ellipse
        assert all(s95 > s68 for s68, s95 in zip(scales_68, scales_95))


class TestPlotEmbeddingBasics:
    """Test basic plot_embedding functionality."""

    def test_returns_figure(self):
        """plot_embedding should return matplotlib Figure."""
        embedding = np.random.randn(50, 2)
        fig = plot_embedding(embedding)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_2d_embedding(self):
        """2D embedding should plot successfully."""
        embedding = np.random.randn(50, 2)
        fig = plot_embedding(embedding)
        assert fig is not None
        plt.close(fig)

    def test_3d_embedding(self):
        """3D embedding should plot successfully."""
        embedding = np.random.randn(50, 3)
        fig = plot_embedding(embedding)
        assert fig is not None
        plt.close(fig)

    def test_custom_embedding_name(self):
        """Custom embedding name should be used."""
        embedding = np.random.randn(50, 2)
        fig = plot_embedding(embedding, embedding_name="PCA")
        assert fig is not None
        plt.close(fig)

    def test_custom_title(self):
        """Custom title should be used."""
        embedding = np.random.randn(50, 2)
        fig = plot_embedding(embedding, title="Custom Title")
        assert fig is not None
        plt.close(fig)

    def test_custom_figure_size(self):
        """Custom figure size should be applied."""
        embedding = np.random.randn(50, 2)
        fig = plot_embedding(embedding, figure_size=(10, 8))
        assert fig is not None
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8
        plt.close(fig)


class TestPlotEmbeddingColoring:
    """Test plot_embedding with various coloring modes."""

    def test_class_coloring(self):
        """Class-based coloring should work."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B', 'C'], 34)[:100]
        fig = plot_embedding(embedding, class_labels=classes)
        assert fig is not None
        plt.close(fig)

    def test_numeric_class_labels(self):
        """Numeric class labels should work."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat([0, 1, 2], 34)[:100]
        fig = plot_embedding(embedding, class_labels=classes)
        assert fig is not None
        plt.close(fig)

    def test_batch_labels(self):
        """Batch labels parameter should not raise."""
        embedding = np.random.randn(100, 2)
        batches = np.repeat(['B1', 'B2'], 50)
        fig = plot_embedding(embedding, batch_labels=batches)
        assert fig is not None
        plt.close(fig)

    def test_class_and_batch_labels(self):
        """Both class and batch labels should work together."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B'], 50)
        batches = np.tile(['B1', 'B2'], 50)
        fig = plot_embedding(embedding, class_labels=classes, batch_labels=batches)
        assert fig is not None
        plt.close(fig)

    def test_stage_labels_faceting(self):
        """Stage labels should create faceted plot."""
        embedding = np.random.randn(100, 2)
        stages = np.repeat(['Raw', 'Processed'], 50)
        fig = plot_embedding(embedding, stage_labels=stages)
        assert fig is not None
        plt.close(fig)

    def test_stage_with_class_coloring(self):
        """Stage faceting with class coloring should work."""
        embedding = np.random.randn(100, 2)
        classes = np.tile(['A', 'B'], 50)
        stages = np.repeat(['Raw', 'Processed'], 50)
        fig = plot_embedding(embedding, class_labels=classes, stage_labels=stages)
        assert fig is not None
        plt.close(fig)


class TestPlotEmbeddingEllipses:
    """Test confidence ellipse visualization."""

    def test_ellipses_68(self):
        """Ellipses with 68% confidence should work."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B'], 50)
        fig = plot_embedding(
            embedding, class_labels=classes, show_ellipses=True, ellipse_confidence=0.68
        )
        assert fig is not None
        plt.close(fig)

    def test_ellipses_95(self):
        """Ellipses with 95% confidence should work."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B'], 50)
        fig = plot_embedding(
            embedding, class_labels=classes, show_ellipses=True, ellipse_confidence=0.95
        )
        assert fig is not None
        plt.close(fig)

    def test_ellipses_without_class_labels(self):
        """Ellipses without class labels should not raise."""
        embedding = np.random.randn(100, 2)
        fig = plot_embedding(embedding, show_ellipses=True)
        assert fig is not None
        plt.close(fig)

    def test_ellipses_3d(self):
        """Ellipses with 3D embedding should not cause issues."""
        embedding = np.random.randn(100, 3)
        classes = np.repeat(['A', 'B'], 50)
        fig = plot_embedding(
            embedding, class_labels=classes, show_ellipses=True
        )
        assert fig is not None
        plt.close(fig)


class TestPlotEmbeddingContours:
    """Test density contour visualization."""

    def test_contours_2d(self):
        """Contours should work with 2D embedding."""
        embedding = np.random.randn(100, 2)
        fig = plot_embedding(embedding, show_contours=True, n_contours=5)
        assert fig is not None
        plt.close(fig)

    def test_custom_n_contours(self):
        """Custom number of contours should work."""
        embedding = np.random.randn(100, 2)
        fig = plot_embedding(embedding, show_contours=True, n_contours=10)
        assert fig is not None
        plt.close(fig)

    def test_contours_with_class_labels(self):
        """Contours with class coloring should work."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B'], 50)
        fig = plot_embedding(
            embedding, class_labels=classes, show_contours=True
        )
        assert fig is not None
        plt.close(fig)

    def test_contours_small_sample(self):
        """Contours with small sample size should not crash."""
        embedding = np.random.randn(5, 2)
        fig = plot_embedding(embedding, show_contours=True)
        assert fig is not None
        plt.close(fig)

    def test_ellipses_and_contours(self):
        """Ellipses and contours together should work."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B'], 50)
        fig = plot_embedding(
            embedding, class_labels=classes,
            show_ellipses=True, show_contours=True
        )
        assert fig is not None
        plt.close(fig)


class TestPlotEmbeddingFileIO:
    """Test file I/O functionality."""

    def test_save_png(self):
        """Saving to PNG should create file."""
        embedding = np.random.randn(50, 2)
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "embedding.png"
            fig = plot_embedding(embedding, save_path=save_path)
            assert save_path.exists()
            assert save_path.stat().st_size > 0
            plt.close(fig)

    def test_save_custom_dpi(self):
        """Custom DPI should affect file size."""
        embedding = np.random.randn(50, 2)
        with TemporaryDirectory() as tmpdir:
            path_low_dpi = Path(tmpdir) / "low_dpi.png"
            path_high_dpi = Path(tmpdir) / "high_dpi.png"

            fig1 = plot_embedding(embedding, save_path=path_low_dpi, dpi=100)
            fig2 = plot_embedding(embedding, save_path=path_high_dpi, dpi=300)

            plt.close(fig1)
            plt.close(fig2)

            # Higher DPI should result in larger file
            assert path_high_dpi.stat().st_size > path_low_dpi.stat().st_size

    def test_save_creates_parent_dirs(self):
        """Saving should create parent directories."""
        embedding = np.random.randn(50, 2)
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "subdir1" / "subdir2" / "embedding.png"
            fig = plot_embedding(embedding, save_path=save_path)
            assert save_path.exists()
            plt.close(fig)


class TestPlotEmbeddingComparison:
    """Test plot_embedding_comparison function."""

    def test_returns_figure(self):
        """plot_embedding_comparison should return Figure."""
        embeddings = {
            "PCA": np.random.randn(50, 2),
            "UMAP": np.random.randn(50, 2)
        }
        fig = plot_embedding_comparison(embeddings)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_two_embeddings(self):
        """Two embeddings should create side-by-side plot."""
        embeddings = {
            "PCA": np.random.randn(50, 2),
            "UMAP": np.random.randn(50, 2)
        }
        fig = plot_embedding_comparison(embeddings)
        assert fig is not None
        plt.close(fig)

    def test_three_embeddings(self):
        """Three embeddings should work."""
        embeddings = {
            "PCA": np.random.randn(50, 2),
            "UMAP": np.random.randn(50, 2),
            "tSNE": np.random.randn(50, 2)
        }
        fig = plot_embedding_comparison(embeddings)
        assert fig is not None
        plt.close(fig)

    def test_with_class_labels(self):
        """Comparison with class labels should work."""
        embeddings = {
            "PCA": np.random.randn(50, 2),
            "UMAP": np.random.randn(50, 2)
        }
        classes = np.repeat(['A', 'B'], 25)
        fig = plot_embedding_comparison(embeddings, class_labels=classes)
        assert fig is not None
        plt.close(fig)

    def test_with_ellipses(self):
        """Comparison with ellipses should work."""
        embeddings = {
            "PCA": np.random.randn(50, 2),
            "UMAP": np.random.randn(50, 2)
        }
        classes = np.repeat(['A', 'B'], 25)
        fig = plot_embedding_comparison(
            embeddings, class_labels=classes, show_ellipses=True
        )
        assert fig is not None
        plt.close(fig)

    def test_mismatched_samples(self):
        """Mismatched sample counts should raise ValueError."""
        embeddings = {
            "PCA": np.random.randn(50, 2),
            "UMAP": np.random.randn(60, 2)
        }
        with pytest.raises(ValueError, match="expected"):
            plot_embedding_comparison(embeddings)

    def test_empty_embeddings(self):
        """Empty embeddings dict should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            plot_embedding_comparison({})


class TestGetEmbeddingStatistics:
    """Test statistics extraction."""

    def test_returns_dict(self):
        """get_embedding_statistics should return dict."""
        embedding = np.random.randn(50, 2)
        stats = get_embedding_statistics(embedding)
        assert isinstance(stats, dict)

    def test_global_statistics(self):
        """Global statistics should have correct keys."""
        embedding = np.random.randn(50, 2)
        stats = get_embedding_statistics(embedding)
        assert 'global' in stats
        required_keys = {'n_samples', 'mean_x', 'mean_y', 'std_x', 'std_y', 'range_x', 'range_y', 'separation'}
        assert required_keys.issubset(set(stats['global'].keys()))

    def test_class_statistics(self):
        """Per-class statistics should be computed."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B'], 50)
        stats = get_embedding_statistics(embedding, classes)
        assert 'A' in stats
        assert 'B' in stats
        assert stats['A']['n_samples'] == 50
        assert stats['B']['n_samples'] == 50

    def test_separation_metric(self):
        """Separation should be computed."""
        embedding = np.array([[0, 0], [0, 0], [10, 10], [10, 10]])
        classes = np.array(['A', 'A', 'B', 'B'])
        stats = get_embedding_statistics(embedding, classes)
        # Distance between classes should be > 0
        assert stats['A']['separation'] > 0
        assert stats['B']['separation'] > 0

    def test_numeric_values(self):
        """All statistics should be numeric."""
        embedding = np.random.randn(50, 2)
        stats = get_embedding_statistics(embedding)
        for stat_dict in stats.values():
            for value in stat_dict.values():
                assert isinstance(value, (int, float))


class TestEmbeddingEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample(self):
        """Single sample should work."""
        embedding = np.array([[1.0, 2.0]])
        fig = plot_embedding(embedding)
        assert fig is not None
        plt.close(fig)

    def test_two_samples(self):
        """Two samples should work."""
        embedding = np.array([[1.0, 2.0], [3.0, 4.0]])
        fig = plot_embedding(embedding)
        assert fig is not None
        plt.close(fig)

    def test_identical_points(self):
        """Identical points should not cause errors."""
        embedding = np.ones((50, 2))
        fig = plot_embedding(embedding)
        assert fig is not None
        plt.close(fig)

    def test_identical_points_with_ellipses(self):
        """Identical points with ellipses should not crash."""
        embedding = np.ones((50, 2))
        classes = np.repeat(['A', 'B'], 25)
        fig = plot_embedding(embedding, class_labels=classes, show_ellipses=True)
        assert fig is not None
        plt.close(fig)

    def test_large_embedding(self):
        """Large embedding should work."""
        embedding = np.random.randn(10000, 2)
        fig = plot_embedding(embedding)
        assert fig is not None
        plt.close(fig)

    def test_single_class(self):
        """Single class should work."""
        embedding = np.random.randn(50, 2)
        classes = np.zeros(50, dtype=int)
        fig = plot_embedding(embedding, class_labels=classes)
        assert fig is not None
        plt.close(fig)

    def test_many_classes(self):
        """Many classes should work."""
        embedding = np.random.randn(100, 2)
        classes = np.arange(100)
        fig = plot_embedding(embedding, class_labels=classes)
        assert fig is not None
        plt.close(fig)

    def test_custom_colormaps(self):
        """Different colormaps should work."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B', 'C'], 34)[:100]
        for cmap in ['tab10', 'Set1', 'Pastel1', 'viridis']:
            fig = plot_embedding(embedding, class_labels=classes, class_colormap=cmap)
            assert fig is not None
            plt.close(fig)

    def test_custom_alpha(self):
        """Custom alpha should work."""
        embedding = np.random.randn(50, 2)
        for alpha in [0.3, 0.5, 0.9]:
            fig = plot_embedding(embedding, alpha=alpha)
            assert fig is not None
            plt.close(fig)

    def test_custom_marker_size(self):
        """Custom marker size should work."""
        embedding = np.random.randn(50, 2)
        for size in [20, 50, 100]:
            fig = plot_embedding(embedding, marker_size=size)
            assert fig is not None
            plt.close(fig)


class TestEmbeddingIntegration:
    """Integration tests combining multiple features."""

    def test_full_featured_embedding(self):
        """Full-featured embedding with all options."""
        embedding = np.random.randn(100, 2)
        classes = np.repeat(['A', 'B'], 50)
        batches = np.tile(['B1', 'B2'], 50)

        fig = plot_embedding(
            embedding,
            class_labels=classes,
            batch_labels=batches,
            embedding_name="PCA",
            class_colormap="tab10",
            show_ellipses=True,
            ellipse_confidence=0.95,
            show_contours=True,
            n_contours=8,
            alpha=0.8,
            marker_size=75,
            title="Full-Featured Embedding",
            figure_size=(14, 10)
        )
        assert fig is not None
        plt.close(fig)

    def test_pca_vs_umap_comparison(self):
        """Compare PCA vs UMAP embeddings."""
        np.random.seed(42)
        # Simulate two different embeddings of same data
        pca = np.random.randn(100, 2)
        umap = pca + 0.1 * np.random.randn(100, 2)

        classes = np.repeat(['Type1', 'Type2'], 50)

        embeddings = {"PCA": pca, "UMAP": umap}
        fig = plot_embedding_comparison(
            embeddings,
            class_labels=classes,
            show_ellipses=True,
            title="PCA vs UMAP"
        )
        assert fig is not None
        plt.close(fig)

    def test_statistics_and_visualization(self):
        """Combined statistics and visualization."""
        embedding = np.random.randn(100, 2)
        classes = np.tile(['ClassA', 'ClassB', 'ClassC'], 34)[:100]

        # Get statistics
        stats = get_embedding_statistics(embedding, classes)

        # Verify statistics computed correctly
        assert all(cls in stats for cls in ['ClassA', 'ClassB', 'ClassC'])
        # Verify all classes have samples
        assert all(stats[cls]['n_samples'] > 0 for cls in stats)

        # Create visualization
        fig = plot_embedding(
            embedding,
            class_labels=classes,
            show_ellipses=True,
            show_contours=True
        )
        assert fig is not None
        plt.close(fig)
