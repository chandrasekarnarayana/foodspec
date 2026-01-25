"""Tests for batch drift and stage difference visualizations."""

import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pytest

from foodspec.viz.drift import (
    get_batch_statistics,
    get_stage_statistics,
    plot_batch_drift,
    plot_stage_differences,
    plot_replicate_similarity,
    plot_temporal_drift,
    _compute_batch_statistics,
    _compute_difference_from_reference,
    _compute_stage_statistics,
    _compute_pairwise_differences,
    _auto_select_baseline_stage,
    _compute_similarity_matrix,
    _perform_hierarchical_clustering,
    _parse_timestamps,
    _compute_rolling_average,
)


def _create_synthetic_batches(n_batches=3, samples_per_batch=30, n_features=100):
    """Create synthetic spectral data with batch effects."""
    np.random.seed(42)
    
    spectra_list = []
    batch_labels = []
    
    for batch_id in range(n_batches):
        # Add batch-specific offset and noise
        offset = batch_id * 0.5
        batch_spectra = np.random.randn(samples_per_batch, n_features) + offset
        
        spectra_list.append(batch_spectra)
        batch_labels.extend([f"Batch{batch_id}"] * samples_per_batch)
    
    spectra = np.vstack(spectra_list)
    return spectra, np.array(batch_labels)


def _create_synthetic_stages(n_stages=3, samples_per_stage=50, n_features=100):
    """Create synthetic spectral data for different processing stages."""
    np.random.seed(42)
    
    stages = ["raw", "baseline_corrected", "normalized", "smoothed"][:n_stages]
    spectra_by_stage = {}
    
    for idx, stage in enumerate(stages):
        # Each stage has different characteristics
        scale = 1.0 + idx * 0.3
        offset = idx * 0.2
        stage_spectra = np.random.randn(samples_per_stage, n_features) * scale + offset
        spectra_by_stage[stage] = stage_spectra
    
    return spectra_by_stage


class TestBatchStatistics:
    """Test batch statistics computation."""
    
    def test_compute_batch_statistics_basic(self):
        """Test basic batch statistics computation."""
        spectra, batch_labels = _create_synthetic_batches(n_batches=2)
        stats = _compute_batch_statistics(spectra, batch_labels)
        
        assert "Batch0" in stats
        assert "Batch1" in stats
        assert "mean" in stats["Batch0"]
        assert "std" in stats["Batch0"]
        assert "ci_lower" in stats["Batch0"]
        assert "ci_upper" in stats["Batch0"]
    
    def test_batch_statistics_sample_counts(self):
        """Test that sample counts are correct."""
        spectra, batch_labels = _create_synthetic_batches(
            n_batches=3, samples_per_batch=25
        )
        stats = _compute_batch_statistics(spectra, batch_labels)
        
        for batch in ["Batch0", "Batch1", "Batch2"]:
            assert stats[batch]["n_samples"] == 25
    
    def test_confidence_intervals_contain_mean(self):
        """Test that confidence intervals contain mean."""
        spectra, batch_labels = _create_synthetic_batches()
        stats = _compute_batch_statistics(spectra, batch_labels)
        
        for batch, batch_stats in stats.items():
            assert np.all(batch_stats["ci_lower"] <= batch_stats["mean"])
            assert np.all(batch_stats["ci_upper"] >= batch_stats["mean"])
    
    def test_difference_from_reference(self):
        """Test difference computation."""
        spectra, batch_labels = _create_synthetic_batches(n_batches=3)
        stats = _compute_batch_statistics(spectra, batch_labels)
        differences = _compute_difference_from_reference(stats, "Batch0")
        
        assert "Batch1" in differences
        assert "Batch2" in differences
        assert "Batch0" not in differences


class TestBatchDriftPlotting:
    """Test batch drift plotting."""
    
    def test_plot_returns_figure(self):
        """Test that plotting returns matplotlib Figure."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        fig = plot_batch_drift(spectra, meta)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_has_two_axes(self):
        """Test that figure has two subplots."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        fig = plot_batch_drift(spectra, meta)
        
        assert len(fig.axes) == 2
        plt.close(fig)
    
    def test_plot_with_wavenumbers(self):
        """Test plotting with custom wavenumbers."""
        spectra, batch_labels = _create_synthetic_batches(n_features=50)
        meta = {"batch": batch_labels}
        wavenumbers = np.linspace(400, 4000, 50)
        
        fig = plot_batch_drift(spectra, meta, wavenumbers=wavenumbers)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_reference_batch(self):
        """Test plotting with specified reference batch."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        fig = plot_batch_drift(spectra, meta, reference_batch="Batch1")
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_saves_file(self):
        """Test that plot saves PNG file."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_batch_drift(spectra, meta, save_path=tmpdir)
            
            png_path = tmpdir / "batch_drift.png"
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            
            plt.close(fig)
    
    def test_plot_with_custom_size(self):
        """Test plotting with custom figure size."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        custom_size = (10, 8)
        
        fig = plot_batch_drift(spectra, meta, figure_size=custom_size)
        
        w, h = fig.get_size_inches()
        assert abs(w - custom_size[0]) < 0.1
        assert abs(h - custom_size[1]) < 0.1
        
        plt.close(fig)
    
    def test_plot_with_custom_confidence(self):
        """Test plotting with custom confidence level."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        fig = plot_batch_drift(spectra, meta, confidence=0.99)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_missing_batch_key_raises(self):
        """Test that missing batch key raises KeyError."""
        spectra, _ = _create_synthetic_batches()
        meta = {"wrong_key": np.array(["A"] * len(spectra))}
        
        with pytest.raises(KeyError):
            plot_batch_drift(spectra, meta, batch_key="batch")
    
    def test_plot_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels[:10]}  # Wrong length
        
        with pytest.raises(ValueError):
            plot_batch_drift(spectra, meta)


class TestStageStatistics:
    """Test stage statistics computation."""
    
    def test_compute_stage_statistics_basic(self):
        """Test basic stage statistics computation."""
        spectra_by_stage = _create_synthetic_stages(n_stages=3)
        stats = _compute_stage_statistics(spectra_by_stage)
        
        assert "raw" in stats
        assert "baseline_corrected" in stats
        assert "normalized" in stats
        
        for stage_stats in stats.values():
            assert "mean" in stage_stats
            assert "std" in stage_stats
            assert "n_samples" in stage_stats
    
    def test_pairwise_differences(self):
        """Test pairwise difference computation."""
        spectra_by_stage = _create_synthetic_stages(n_stages=3)
        stats = _compute_stage_statistics(spectra_by_stage)
        differences = _compute_pairwise_differences(stats, "raw")
        
        assert "baseline_corrected" in differences
        assert "normalized" in differences
        assert "raw" not in differences
    
    def test_auto_select_baseline_with_raw(self):
        """Test baseline selection prefers 'raw' stage."""
        spectra_by_stage = {
            "processed": np.random.randn(50, 100),
            "raw": np.random.randn(30, 100),
            "normalized": np.random.randn(40, 100),
        }
        
        baseline = _auto_select_baseline_stage(spectra_by_stage)
        
        assert baseline == "raw"
    
    def test_auto_select_baseline_with_order(self):
        """Test baseline selection uses stage_order."""
        spectra_by_stage = {
            "stage_a": np.random.randn(50, 100),
            "stage_b": np.random.randn(30, 100),
            "stage_c": np.random.randn(40, 100),
        }
        stage_order = ["stage_b", "stage_a", "stage_c"]
        
        baseline = _auto_select_baseline_stage(spectra_by_stage, stage_order)
        
        assert baseline == "stage_b"
    
    def test_auto_select_baseline_most_samples(self):
        """Test baseline selection uses most samples as fallback."""
        spectra_by_stage = {
            "stage_a": np.random.randn(30, 100),
            "stage_b": np.random.randn(50, 100),  # Most samples
            "stage_c": np.random.randn(20, 100),
        }
        
        baseline = _auto_select_baseline_stage(spectra_by_stage)
        
        assert baseline == "stage_b"


class TestStageDifferencePlotting:
    """Test stage difference plotting."""
    
    def test_plot_returns_figure(self):
        """Test that plotting returns matplotlib Figure."""
        spectra_by_stage = _create_synthetic_stages()
        
        fig = plot_stage_differences(spectra_by_stage)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_has_two_axes(self):
        """Test that figure has two subplots."""
        spectra_by_stage = _create_synthetic_stages()
        
        fig = plot_stage_differences(spectra_by_stage)
        
        assert len(fig.axes) == 2
        plt.close(fig)
    
    def test_plot_with_wavenumbers(self):
        """Test plotting with custom wavenumbers."""
        spectra_by_stage = _create_synthetic_stages(n_features=50)
        wavenumbers = np.linspace(400, 4000, 50)
        
        fig = plot_stage_differences(spectra_by_stage, wavenumbers=wavenumbers)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_baseline_stage(self):
        """Test plotting with specified baseline stage."""
        spectra_by_stage = _create_synthetic_stages()
        
        fig = plot_stage_differences(spectra_by_stage, baseline_stage="normalized")
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_stage_order(self):
        """Test plotting with custom stage order."""
        spectra_by_stage = _create_synthetic_stages()
        stage_order = ["normalized", "baseline_corrected", "raw"]
        
        fig = plot_stage_differences(spectra_by_stage, stage_order=stage_order)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_saves_file(self):
        """Test that plot saves PNG file."""
        spectra_by_stage = _create_synthetic_stages()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_stage_differences(spectra_by_stage, save_path=tmpdir)
            
            png_path = tmpdir / "stage_differences.png"
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            
            plt.close(fig)
    
    def test_plot_with_custom_size(self):
        """Test plotting with custom figure size."""
        spectra_by_stage = _create_synthetic_stages()
        custom_size = (12, 8)
        
        fig = plot_stage_differences(spectra_by_stage, figure_size=custom_size)
        
        w, h = fig.get_size_inches()
        assert abs(w - custom_size[0]) < 0.1
        assert abs(h - custom_size[1]) < 0.1
        
        plt.close(fig)
    
    def test_plot_empty_dict_raises(self):
        """Test that empty dict raises ValueError."""
        with pytest.raises(ValueError):
            plot_stage_differences({})
    
    def test_plot_mismatched_features_raises(self):
        """Test that mismatched features raise ValueError."""
        spectra_by_stage = {
            "stage_a": np.random.randn(50, 100),
            "stage_b": np.random.randn(50, 150),  # Wrong number of features
        }
        
        with pytest.raises(ValueError):
            plot_stage_differences(spectra_by_stage)


class TestGetBatchStatistics:
    """Test batch statistics extraction."""
    
    def test_get_batch_statistics_structure(self):
        """Test that returned dict has correct structure."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        result = get_batch_statistics(spectra, meta)
        
        assert "batch_stats" in result
        assert "summary" in result
    
    def test_summary_has_required_fields(self):
        """Test that summary has all required fields."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        result = get_batch_statistics(spectra, meta)
        summary = result["summary"]
        
        assert "total_batches" in summary
        assert "total_samples" in summary
        assert "batch_names" in summary
        assert "samples_per_batch" in summary
        assert "max_pairwise_difference" in summary
        assert "max_difference_pair" in summary
    
    def test_batch_counts_correct(self):
        """Test that batch counts are correct."""
        spectra, batch_labels = _create_synthetic_batches(
            n_batches=4, samples_per_batch=20
        )
        meta = {"batch": batch_labels}
        
        result = get_batch_statistics(spectra, meta)
        summary = result["summary"]
        
        assert summary["total_batches"] == 4
        assert summary["total_samples"] == 80
    
    def test_max_difference_computed(self):
        """Test that max pairwise difference is computed."""
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        result = get_batch_statistics(spectra, meta)
        summary = result["summary"]
        
        assert summary["max_pairwise_difference"] > 0
        assert summary["max_difference_pair"] is not None


class TestGetStageStatistics:
    """Test stage statistics extraction."""
    
    def test_get_stage_statistics_structure(self):
        """Test that returned dict has correct structure."""
        spectra_by_stage = _create_synthetic_stages()
        
        result = get_stage_statistics(spectra_by_stage)
        
        assert "stage_stats" in result
        assert "differences" in result
        assert "summary" in result
    
    def test_summary_has_required_fields(self):
        """Test that summary has all required fields."""
        spectra_by_stage = _create_synthetic_stages()
        
        result = get_stage_statistics(spectra_by_stage)
        summary = result["summary"]
        
        assert "total_stages" in summary
        assert "total_samples" in summary
        assert "stage_names" in summary
        assert "baseline_stage" in summary
        assert "samples_per_stage" in summary
        assert "max_difference_from_baseline" in summary
        assert "max_difference_stage" in summary
    
    def test_stage_counts_correct(self):
        """Test that stage counts are correct."""
        spectra_by_stage = _create_synthetic_stages(
            n_stages=3, samples_per_stage=40
        )
        
        result = get_stage_statistics(spectra_by_stage)
        summary = result["summary"]
        
        assert summary["total_stages"] == 3
        assert summary["total_samples"] == 120
    
    def test_baseline_stage_identified(self):
        """Test that baseline stage is correctly identified."""
        spectra_by_stage = _create_synthetic_stages()
        
        result = get_stage_statistics(spectra_by_stage)
        summary = result["summary"]
        
        assert summary["baseline_stage"] == "raw"
    
    def test_max_difference_computed(self):
        """Test that max difference from baseline is computed."""
        spectra_by_stage = _create_synthetic_stages()
        
        result = get_stage_statistics(spectra_by_stage)
        summary = result["summary"]
        
        assert summary["max_difference_from_baseline"] > 0
        assert summary["max_difference_stage"] is not None


class TestIntegration:
    """Integration tests for drift visualizations."""
    
    def test_batch_drift_full_workflow(self):
        """Test complete batch drift workflow."""
        spectra, batch_labels = _create_synthetic_batches(n_batches=4)
        meta = {"batch": batch_labels}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Get statistics
            stats = get_batch_statistics(spectra, meta)
            
            # Generate plot
            fig = plot_batch_drift(spectra, meta, save_path=tmpdir)
            
            assert stats["summary"]["total_batches"] == 4
            assert (tmpdir / "batch_drift.png").exists()
            
            plt.close(fig)
    
    def test_stage_differences_full_workflow(self):
        """Test complete stage differences workflow."""
        spectra_by_stage = _create_synthetic_stages(n_stages=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Get statistics
            stats = get_stage_statistics(spectra_by_stage)
            
            # Generate plot
            fig = plot_stage_differences(spectra_by_stage, save_path=tmpdir)
            
            assert len(stats["stage_stats"]) == 4
            assert (tmpdir / "stage_differences.png").exists()
            
            plt.close(fig)
    
    def test_both_visualizations_together(self):
        """Test generating both visualizations together."""
        # Batch drift
        spectra, batch_labels = _create_synthetic_batches()
        meta = {"batch": batch_labels}
        
        # Stage differences
        spectra_by_stage = _create_synthetic_stages()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            fig1 = plot_batch_drift(spectra, meta, save_path=tmpdir / "batch")
            fig2 = plot_stage_differences(
                spectra_by_stage, save_path=tmpdir / "stage"
            )
            
            assert (tmpdir / "batch" / "batch_drift.png").exists()
            assert (tmpdir / "stage" / "stage_differences.png").exists()
            
            plt.close(fig1)
            plt.close(fig2)


class TestSimilarityMatrix:
    """Test similarity matrix computation."""
    
    def test_compute_cosine_similarity(self):
        """Test cosine similarity computation."""
        np.random.seed(42)
        spectra = np.random.randn(10, 50)
        
        similarity = _compute_similarity_matrix(spectra, metric="cosine")
        
        assert similarity.shape == (10, 10)
        assert np.allclose(np.diag(similarity), 1.0)  # Self-similarity = 1
        assert np.all(similarity >= -1) and np.all(similarity <= 1)
    
    def test_compute_correlation_similarity(self):
        """Test correlation similarity computation."""
        np.random.seed(42)
        spectra = np.random.randn(10, 50)
        
        similarity = _compute_similarity_matrix(spectra, metric="correlation")
        
        assert similarity.shape == (10, 10)
        assert np.allclose(np.diag(similarity), 1.0)
    
    def test_similarity_is_symmetric(self):
        """Test that similarity matrix is symmetric."""
        np.random.seed(42)
        spectra = np.random.randn(10, 50)
        
        similarity = _compute_similarity_matrix(spectra)
        
        assert np.allclose(similarity, similarity.T)
    
    def test_invalid_metric_raises(self):
        """Test that invalid metric raises ValueError."""
        spectra = np.random.randn(10, 50)
        
        with pytest.raises(ValueError):
            _compute_similarity_matrix(spectra, metric="invalid")


class TestHierarchicalClustering:
    """Test hierarchical clustering."""
    
    def test_perform_clustering(self):
        """Test clustering computation."""
        np.random.seed(42)
        similarity = np.random.rand(10, 10)
        similarity = (similarity + similarity.T) / 2  # Make symmetric
        np.fill_diagonal(similarity, 1.0)
        
        row_linkage, col_linkage = _perform_hierarchical_clustering(similarity)
        
        assert row_linkage.shape[0] == 9  # n-1 merges
        assert row_linkage.shape[1] == 4  # linkage format
    
    def test_clustering_with_identical_samples(self):
        """Test clustering with identical samples."""
        similarity = np.ones((5, 5))
        
        row_linkage, col_linkage = _perform_hierarchical_clustering(similarity)
        
        assert row_linkage.shape[0] == 4


class TestReplicateSimilarityPlotting:
    """Test replicate similarity plotting."""
    
    def test_plot_returns_figure(self):
        """Test that plotting returns matplotlib Figure."""
        np.random.seed(42)
        spectra = np.random.randn(20, 50)
        
        fig = plot_replicate_similarity(spectra)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_labels(self):
        """Test plotting with custom labels."""
        np.random.seed(42)
        spectra = np.random.randn(15, 50)
        labels = [f"Rep_{i}" for i in range(15)]
        
        fig = plot_replicate_similarity(spectra, labels=labels)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_correlation_metric(self):
        """Test plotting with correlation metric."""
        np.random.seed(42)
        spectra = np.random.randn(12, 50)
        
        fig = plot_replicate_similarity(spectra, metric="correlation")
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_without_clustering(self):
        """Test plotting without clustering."""
        np.random.seed(42)
        spectra = np.random.randn(10, 50)
        
        fig = plot_replicate_similarity(spectra, cluster=False)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_saves_file(self):
        """Test that plot saves PNG file."""
        np.random.seed(42)
        spectra = np.random.randn(15, 50)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_replicate_similarity(spectra, save_path=tmpdir)
            
            png_path = tmpdir / "replicate_similarity.png"
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            
            plt.close(fig)
    
    def test_plot_with_custom_size(self):
        """Test plotting with custom figure size."""
        np.random.seed(42)
        spectra = np.random.randn(10, 50)
        custom_size = (10, 8)
        
        fig = plot_replicate_similarity(spectra, figure_size=custom_size)
        
        w, h = fig.get_size_inches()
        assert abs(w - custom_size[0]) < 0.1
        assert abs(h - custom_size[1]) < 0.1
        
        plt.close(fig)


class TestTimestampParsing:
    """Test timestamp parsing."""
    
    def test_parse_numeric_timestamps(self):
        """Test parsing numeric timestamps."""
        time_values = np.array([1.0, 2.0, 3.0, 4.0])
        
        parsed = _parse_timestamps(time_values)
        
        assert np.allclose(parsed, time_values)
    
    def test_parse_datetime_objects(self):
        """Test parsing datetime objects."""
        base = datetime(2025, 1, 1)
        time_values = np.array([base + timedelta(days=i) for i in range(5)])
        
        parsed = _parse_timestamps(time_values)
        
        assert len(parsed) == 5
        assert isinstance(parsed[0], (float, np.floating))
    
    def test_parse_iso_strings(self):
        """Test parsing ISO format strings."""
        time_values = np.array([
            "2025-01-01T00:00:00",
            "2025-01-02T00:00:00",
            "2025-01-03T00:00:00",
        ])
        
        parsed = _parse_timestamps(time_values)
        
        assert len(parsed) == 3
        assert parsed[1] > parsed[0]
    
    def test_parse_with_custom_format(self):
        """Test parsing with custom format string."""
        time_values = np.array(["2025-01-01", "2025-01-02", "2025-01-03"])
        time_format = "%Y-%m-%d"
        
        parsed = _parse_timestamps(time_values, time_format)
        
        assert len(parsed) == 3
    
    def test_parse_invalid_fallback_to_indices(self):
        """Test that invalid formats fall back to indices."""
        time_values = np.array(["invalid1", "invalid2", "invalid3"])
        
        parsed = _parse_timestamps(time_values)
        
        assert np.allclose(parsed, [0, 1, 2])


class TestRollingAverage:
    """Test rolling average computation."""
    
    def test_rolling_average_basic(self):
        """Test basic rolling average."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window = 3
        
        smoothed = _compute_rolling_average(values, window)
        
        assert len(smoothed) == len(values)
        assert smoothed[2] == pytest.approx(3.0, abs=0.1)
    
    def test_rolling_average_window_one(self):
        """Test that window=1 returns original."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        
        smoothed = _compute_rolling_average(values, 1)
        
        assert np.allclose(smoothed, values)
    
    def test_rolling_average_reduces_variance(self):
        """Test that rolling average reduces variance."""
        np.random.seed(42)
        values = np.random.randn(100)
        
        smoothed = _compute_rolling_average(values, 5)
        
        assert np.var(smoothed) < np.var(values)


class TestTemporalDriftPlotting:
    """Test temporal drift plotting."""
    
    def test_plot_returns_figure(self):
        """Test that plotting returns matplotlib Figure."""
        np.random.seed(42)
        spectra = np.random.randn(50, 100)
        meta = {"timestamp": np.arange(50)}
        
        fig = plot_temporal_drift(spectra, meta, band_indices=[10, 30, 50])
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_wavenumbers(self):
        """Test plotting with wavenumber ranges."""
        np.random.seed(42)
        spectra = np.random.randn(40, 200)
        wavenumbers = np.linspace(400, 4000, 200)
        meta = {"timestamp": np.arange(40)}
        band_ranges = [(800, 1000), (1400, 1600), (2800, 3000)]
        
        fig = plot_temporal_drift(
            spectra,
            meta,
            wavenumbers=wavenumbers,
            band_ranges=band_ranges,
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_rolling_window(self):
        """Test plotting with rolling average."""
        np.random.seed(42)
        spectra = np.random.randn(60, 100)
        meta = {"timestamp": np.arange(60)}
        
        fig = plot_temporal_drift(
            spectra,
            meta,
            band_indices=[20, 50, 80],
            rolling_window=5,
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_datetime(self):
        """Test plotting with datetime timestamps."""
        np.random.seed(42)
        spectra = np.random.randn(30, 100)
        base = datetime(2025, 1, 1)
        timestamps = [base + timedelta(hours=i) for i in range(30)]
        meta = {"timestamp": timestamps}
        
        fig = plot_temporal_drift(spectra, meta, band_indices=[25, 50, 75])
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_saves_file(self):
        """Test that plot saves PNG file."""
        np.random.seed(42)
        spectra = np.random.randn(40, 100)
        meta = {"timestamp": np.arange(40)}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fig = plot_temporal_drift(
                spectra,
                meta,
                band_indices=[20, 60],
                save_path=tmpdir,
            )
            
            png_path = tmpdir / "temporal_drift.png"
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            
            plt.close(fig)
    
    def test_plot_auto_bands(self):
        """Test plotting with auto-selected bands."""
        np.random.seed(42)
        spectra = np.random.randn(35, 100)
        meta = {"timestamp": np.arange(35)}
        
        fig = plot_temporal_drift(spectra, meta)  # No bands specified
        
        assert fig is not None
        assert len(fig.axes) == 5  # Auto-selects 5 bands
        plt.close(fig)
    
    def test_plot_missing_time_key_raises(self):
        """Test that missing time key raises KeyError."""
        spectra = np.random.randn(30, 100)
        meta = {"wrong_key": np.arange(30)}
        
        with pytest.raises(KeyError):
            plot_temporal_drift(spectra, meta, time_key="timestamp")
    
    def test_plot_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        spectra = np.random.randn(50, 100)
        meta = {"timestamp": np.arange(30)}  # Wrong length
        
        with pytest.raises(ValueError):
            plot_temporal_drift(spectra, meta)
    
    def test_plot_with_custom_size(self):
        """Test plotting with custom figure size."""
        np.random.seed(42)
        spectra = np.random.randn(40, 100)
        meta = {"timestamp": np.arange(40)}
        custom_size = (12, 6)
        
        fig = plot_temporal_drift(
            spectra,
            meta,
            band_indices=[30, 70],
            figure_size=custom_size,
        )
        
        w, h = fig.get_size_inches()
        assert abs(w - custom_size[0]) < 0.1
        assert abs(h - custom_size[1]) < 0.1
        
        plt.close(fig)


class TestReplicateSimilarityIntegration:
    """Integration tests for replicate similarity."""
    
    def test_replicate_similarity_full_workflow(self):
        """Test complete replicate similarity workflow."""
        np.random.seed(42)
        
        # Create replicates with some similarity
        n_replicates = 20
        n_features = 100
        
        # Generate base spectrum
        base = np.random.randn(n_features)
        
        # Add noise to create replicates
        spectra = []
        labels = []
        for i in range(n_replicates):
            replicate = base + np.random.randn(n_features) * 0.5
            spectra.append(replicate)
            labels.append(f"Rep_{i+1}")
        
        spectra = np.array(spectra)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Test both metrics
            fig1 = plot_replicate_similarity(
                spectra,
                labels=labels,
                metric="cosine",
                save_path=tmpdir / "cosine",
            )
            
            fig2 = plot_replicate_similarity(
                spectra,
                labels=labels,
                metric="correlation",
                cluster=False,
                save_path=tmpdir / "correlation",
            )
            
            assert (tmpdir / "cosine" / "replicate_similarity.png").exists()
            assert (tmpdir / "correlation" / "replicate_similarity.png").exists()
            
            plt.close(fig1)
            plt.close(fig2)


class TestTemporalDriftIntegration:
    """Integration tests for temporal drift."""
    
    def test_temporal_drift_full_workflow(self):
        """Test complete temporal drift workflow."""
        np.random.seed(42)
        
        # Create time series with drift
        n_timepoints = 100
        n_features = 200
        
        # Base spectrum with temporal drift
        wavenumbers = np.linspace(400, 4000, n_features)
        base = np.sin(wavenumbers / 500)
        
        spectra = []
        timestamps = []
        base_time = datetime(2025, 1, 1)
        
        for t in range(n_timepoints):
            # Add drift: linear increase over time
            drift = t * 0.01
            spectrum = base + drift + np.random.randn(n_features) * 0.1
            spectra.append(spectrum)
            timestamps.append(base_time + timedelta(hours=t))
        
        spectra = np.array(spectra)
        meta = {"timestamp": timestamps}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Test with different configurations
            fig1 = plot_temporal_drift(
                spectra,
                meta,
                band_indices=[50, 100, 150],
                rolling_window=1,
                save_path=tmpdir / "no_smooth",
            )
            
            fig2 = plot_temporal_drift(
                spectra,
                meta,
                wavenumbers=wavenumbers,
                band_ranges=[(800, 1000), (1800, 2000), (3000, 3200)],
                rolling_window=10,
                save_path=tmpdir / "smoothed",
            )
            
            assert (tmpdir / "no_smooth" / "temporal_drift.png").exists()
            assert (tmpdir / "smoothed" / "temporal_drift.png").exists()
            
            plt.close(fig1)
            plt.close(fig2)
