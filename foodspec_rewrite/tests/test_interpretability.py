"""
Tests for interpretability visualizations module.

Tests cover:
- Importance normalization and scaling
- Peak selection and ranking
- Label formatting
- Importance overlay visualization
- Marker bands visualization
- Band statistics extraction
- Input validation and error handling
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import pytest

from foodspec.viz.interpretability import (
    plot_importance_overlay,
    plot_marker_bands,
    get_band_statistics,
    _normalize_importance,
    _select_prominent_peaks,
    _format_band_label,
)


class TestNormalizeImportance:
    """Test importance normalization."""
    
    def test_normalize_basic(self):
        """Test basic normalization to [0, 1]."""
        importance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = _normalize_importance(importance)
        
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
        assert len(normalized) == len(importance)
    
    def test_normalize_negative(self):
        """Test normalization with negative values."""
        importance = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        normalized = _normalize_importance(importance)
        
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
    
    def test_normalize_identical(self):
        """Test normalization with identical values."""
        importance = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = _normalize_importance(importance)
        
        # All should be 0.5
        assert np.allclose(normalized, 0.5)
    
    def test_normalize_empty(self):
        """Test normalization of empty array."""
        importance = np.array([])
        normalized = _normalize_importance(importance)
        
        assert len(normalized) == 0
    
    def test_normalize_single(self):
        """Test normalization with single value."""
        importance = np.array([3.14])
        normalized = _normalize_importance(importance)
        
        assert normalized[0] == pytest.approx(0.5)


class TestSelectProminentPeaks:
    """Test peak selection from importance."""
    
    def test_select_peaks_basic(self):
        """Test basic peak selection."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        importance = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        
        peaks = _select_prominent_peaks(spectrum, importance, n_peaks=3)
        
        assert len(peaks) == 3
        assert all(isinstance(i, int) for i in peaks)
        assert all(0 <= i < len(importance) for i in peaks)
        # Peaks should be sorted by index
        assert peaks == sorted(peaks)
    
    def test_select_peaks_ordered(self):
        """Test that peaks are returned in feature order."""
        spectrum = np.random.randn(100)
        importance = np.random.rand(100)
        
        peaks = _select_prominent_peaks(spectrum, importance, n_peaks=10)
        
        # Should be sorted by index
        assert peaks == sorted(peaks)
    
    def test_select_peaks_more_than_available(self):
        """Test requesting more peaks than available."""
        spectrum = np.array([1.0, 2.0, 3.0])
        importance = np.array([0.1, 0.5, 0.9])
        
        peaks = _select_prominent_peaks(spectrum, importance, n_peaks=10)
        
        # Should return all available
        assert len(peaks) == 3


class TestFormatBandLabel:
    """Test band label formatting."""
    
    def test_format_with_name_only(self):
        """Test formatting with band name."""
        label = _format_band_label(50, name="C-H stretch")
        assert "C-H stretch" in label
    
    def test_format_with_wavenumber(self):
        """Test formatting with wavenumber."""
        label = _format_band_label(50, wavenumber=1234.5)
        assert "1234.5" in label
    
    def test_format_with_importance(self):
        """Test formatting with importance."""
        label = _format_band_label(50, importance=0.85)
        assert "0.85" in label
    
    def test_format_complete(self):
        """Test formatting with all parameters."""
        label = _format_band_label(
            50,
            wavenumber=1234.5,
            importance=0.85,
            name="C-H"
        )
        assert "C-H" in label
        assert "1234.5" in label
        assert "0.85" in label
    
    def test_format_fallback_index(self):
        """Test formatting falls back to index."""
        label = _format_band_label(50)
        assert "50" in label or "Band" in label


class TestImportanceOverlayBasics:
    """Basic tests for importance overlay visualization."""
    
    def test_plot_returns_figure(self):
        """Test that function returns matplotlib Figure."""
        spectrum = np.random.randn(100)
        importance = np.abs(np.random.randn(100))
        
        fig = plot_importance_overlay(spectrum, importance)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_wavenumbers(self):
        """Test plot with wavenumber array."""
        spectrum = np.random.randn(100)
        importance = np.abs(np.random.randn(100))
        wavenumbers = np.linspace(1000, 3000, 100)
        
        fig = plot_importance_overlay(
            spectrum, importance, wavenumbers=wavenumbers
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_style_overlay(self):
        """Test overlay style."""
        spectrum = np.random.randn(50)
        importance = np.abs(np.random.randn(50))
        
        fig = plot_importance_overlay(spectrum, importance, style="overlay")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_style_bar(self):
        """Test bar style."""
        spectrum = np.random.randn(50)
        importance = np.abs(np.random.randn(50))
        
        fig = plot_importance_overlay(spectrum, importance, style="bar")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_style_heat(self):
        """Test heat style."""
        spectrum = np.random.randn(50)
        importance = np.abs(np.random.randn(50))
        
        fig = plot_importance_overlay(spectrum, importance, style="heat")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_threshold(self):
        """Test with importance threshold."""
        spectrum = np.random.randn(50)
        importance = np.abs(np.random.randn(50))
        
        fig = plot_importance_overlay(spectrum, importance, threshold=0.5)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_no_peak_labels(self):
        """Test without peak highlighting."""
        spectrum = np.random.randn(50)
        importance = np.abs(np.random.randn(50))
        
        fig = plot_importance_overlay(
            spectrum, importance, highlight_peaks=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_band_names(self):
        """Test with custom band names."""
        spectrum = np.random.randn(50)
        importance = np.abs(np.random.randn(50))
        band_names = {0: "C-H", 10: "O-H", 20: "C=O"}
        
        fig = plot_importance_overlay(
            spectrum, importance, band_names=band_names
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_saves_file(self):
        """Test saving to file."""
        spectrum = np.random.randn(50)
        importance = np.abs(np.random.randn(50))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_importance.png"
            fig = plot_importance_overlay(spectrum, importance, save_path=save_path)
            
            assert save_path.exists()
            plt.close(fig)
    
    def test_plot_custom_figure_size(self):
        """Test custom figure size."""
        spectrum = np.random.randn(50)
        importance = np.abs(np.random.randn(50))
        
        fig = plot_importance_overlay(
            spectrum, importance, figure_size=(16, 8)
        )
        
        assert fig.get_figwidth() == 16
        assert fig.get_figheight() == 8
        plt.close(fig)


class TestImportanceOverlayValidation:
    """Validation and error handling for importance overlay."""
    
    def test_mismatched_lengths(self):
        """Test error on mismatched spectrum and importance."""
        spectrum = np.random.randn(100)
        importance = np.random.randn(50)
        
        with pytest.raises(ValueError):
            plot_importance_overlay(spectrum, importance)
    
    def test_empty_spectrum(self):
        """Test error on empty spectrum."""
        with pytest.raises(ValueError):
            plot_importance_overlay(np.array([]), np.array([]))
    
    def test_mismatched_wavenumbers(self):
        """Test error on mismatched wavenumbers."""
        spectrum = np.random.randn(100)
        importance = np.random.randn(100)
        wavenumbers = np.linspace(1000, 3000, 50)
        
        with pytest.raises(ValueError):
            plot_importance_overlay(
                spectrum, importance, wavenumbers=wavenumbers
            )


class TestMarkerBandsBasics:
    """Basic tests for marker bands visualization."""
    
    def test_plot_returns_figure(self):
        """Test that function returns matplotlib Figure."""
        spectrum = np.random.randn(100)
        marker_bands = {10: "Band A", 50: "Band B", 80: "Band C"}
        
        fig = plot_marker_bands(spectrum, marker_bands)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_wavenumbers(self):
        """Test plot with wavenumber array."""
        spectrum = np.random.randn(100)
        marker_bands = {10: "Band A", 50: "Band B"}
        wavenumbers = np.linspace(1000, 3000, 100)
        
        fig = plot_marker_bands(
            spectrum, marker_bands, wavenumbers=wavenumbers
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_importance(self):
        """Test with band importance scores."""
        spectrum = np.random.randn(100)
        marker_bands = {10: "Band A", 50: "Band B"}
        band_importance = np.array([0.5, 0.8])
        
        fig = plot_marker_bands(
            spectrum, marker_bands,
            band_importance=band_importance
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_custom_colors(self):
        """Test with custom colors."""
        spectrum = np.random.randn(100)
        marker_bands = {10: "Band A", 50: "Band B"}
        colors = {10: "red", 50: "blue"}
        
        fig = plot_marker_bands(
            spectrum, marker_bands, colors=colors
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_no_peak_heights(self):
        """Test without peak height display."""
        spectrum = np.random.randn(100)
        marker_bands = {10: "Band A", 50: "Band B"}
        
        fig = plot_marker_bands(
            spectrum, marker_bands, show_peak_heights=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_saves_file(self):
        """Test saving to file."""
        spectrum = np.random.randn(100)
        marker_bands = {10: "Band A", 50: "Band B"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_markers.png"
            fig = plot_marker_bands(spectrum, marker_bands, save_path=save_path)
            
            assert save_path.exists()
            plt.close(fig)
    
    def test_plot_custom_figure_size(self):
        """Test custom figure size."""
        spectrum = np.random.randn(100)
        marker_bands = {10: "Band A", 50: "Band B"}
        
        fig = plot_marker_bands(
            spectrum, marker_bands, figure_size=(16, 8)
        )
        
        assert fig.get_figwidth() == 16
        assert fig.get_figheight() == 8
        plt.close(fig)


class TestMarkerBandsValidation:
    """Validation and error handling for marker bands."""
    
    def test_empty_spectrum(self):
        """Test error on empty spectrum."""
        marker_bands = {0: "Band A"}
        with pytest.raises(ValueError):
            plot_marker_bands(np.array([]), marker_bands)
    
    def test_empty_marker_bands(self):
        """Test error on empty marker bands."""
        spectrum = np.random.randn(100)
        with pytest.raises(ValueError):
            plot_marker_bands(spectrum, {})
    
    def test_invalid_band_index(self):
        """Test error on invalid band index."""
        spectrum = np.random.randn(50)
        marker_bands = {100: "Out of range"}
        
        with pytest.raises(ValueError):
            plot_marker_bands(spectrum, marker_bands)
    
    def test_mismatched_wavenumbers(self):
        """Test error on mismatched wavenumbers."""
        spectrum = np.random.randn(100)
        marker_bands = {10: "Band A"}
        wavenumbers = np.linspace(1000, 3000, 50)
        
        with pytest.raises(ValueError):
            plot_marker_bands(
                spectrum, marker_bands, wavenumbers=wavenumbers
            )


class TestBandStatistics:
    """Test band statistics extraction."""
    
    def test_stats_basic(self):
        """Test basic statistics extraction."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = get_band_statistics(spectrum)
        
        assert len(stats) == 5
        assert "intensity" in stats["band_0"]
        assert stats["band_0"]["intensity"] == pytest.approx(1.0)
    
    def test_stats_with_importance(self):
        """Test statistics with importance."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        importance = np.array([0.5, 0.8, 0.3, 0.9, 0.1])
        
        stats = get_band_statistics(spectrum, importance=importance)
        
        assert "importance" in stats["band_0"]
        assert "importance_rank" in stats["band_0"]
        assert stats["band_3"]["importance_rank"] == 1  # Highest importance
    
    def test_stats_with_wavenumbers(self):
        """Test statistics with wavenumbers."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wavenumbers = np.array([1000, 1500, 2000, 2500, 3000])
        
        stats = get_band_statistics(spectrum, wavenumbers=wavenumbers)
        
        assert "wavenumber" in stats["band_0"]
        assert stats["band_0"]["wavenumber"] == pytest.approx(1000)
    
    def test_stats_subset_bands(self):
        """Test statistics for subset of bands."""
        spectrum = np.random.randn(100)
        bands_of_interest = [10, 25, 50, 75]
        
        stats = get_band_statistics(spectrum, bands_of_interest=bands_of_interest)
        
        assert len(stats) == 4
        assert "band_10" in stats
        assert "band_25" in stats


class TestImportanceIntegration:
    """Integration tests for importance overlay."""
    
    def test_full_workflow(self):
        """Test complete importance overlay workflow."""
        np.random.seed(42)
        spectrum = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.randn(200) * 0.1
        importance = np.abs(np.random.randn(200))
        wavenumbers = np.linspace(1000, 3000, 200)
        band_names = {50: "C-H", 100: "O-H", 150: "C=O"}
        
        fig = plot_importance_overlay(
            spectrum, importance,
            wavenumbers=wavenumbers,
            style="overlay",
            highlight_peaks=True,
            n_peaks=5,
            band_names=band_names
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_workflow_all_styles(self):
        """Test workflow with all visualization styles."""
        spectrum = np.random.randn(100)
        importance = np.abs(np.random.randn(100))
        
        for style in ["overlay", "bar", "heat"]:
            fig = plot_importance_overlay(spectrum, importance, style=style)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)


class TestMarkerBandsIntegration:
    """Integration tests for marker bands."""
    
    def test_full_workflow(self):
        """Test complete marker bands workflow."""
        np.random.seed(42)
        spectrum = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.randn(200) * 0.1
        wavenumbers = np.linspace(1000, 3000, 200)
        marker_bands = {
            50: "C-H stretch",
            100: "O-H stretch",
            150: "C=O stretch"
        }
        band_importance = np.array([0.7, 0.9, 0.6])
        
        fig = plot_marker_bands(
            spectrum, marker_bands,
            wavenumbers=wavenumbers,
            band_importance=band_importance,
            show_peak_heights=True
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_many_marker_bands(self):
        """Test with many marker bands."""
        spectrum = np.random.randn(500)
        marker_bands = {i*20: f"Band {i}" for i in range(20)}
        
        fig = plot_marker_bands(spectrum, marker_bands)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
