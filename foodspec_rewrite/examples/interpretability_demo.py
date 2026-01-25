#!/usr/bin/env python
"""
Interpretability Visualizations Demo

Demonstrates the use of plot_importance_overlay() and plot_marker_bands()
for understanding model predictions and identifying chemically relevant bands.

Features:
- Feature importance overlay on spectra
- Multiple visualization styles (overlay, bar, heat)
- Marker band highlighting with custom colors
- Integration with importance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from foodspec.viz.interpretability import (
    plot_importance_overlay,
    plot_marker_bands,
    get_band_statistics,
)


def generate_synthetic_spectrum():
    """Generate synthetic spectral data with known features."""
    np.random.seed(42)
    n_features = 300
    
    # Create base spectrum with known peaks
    wavenumbers = np.linspace(1000, 3000, n_features)
    spectrum = np.zeros(n_features)
    
    # Add Lorentzian peaks at known positions
    peak_positions = [1000, 1500, 2000, 2500, 2850]
    peak_names = ["C-C stretch", "O-H stretch", "C=O stretch", "C-H bend", "C-H stretch"]
    peak_widths = [100, 80, 120, 90, 110]
    peak_heights = [0.8, 0.6, 0.9, 0.5, 0.7]
    
    markers = {}
    
    for pos, width, height, name in zip(peak_positions, peak_widths, peak_heights, peak_names):
        # Find closest wavenumber index
        idx = np.argmin(np.abs(wavenumbers - pos))
        
        # Add peak to spectrum
        x = np.arange(n_features)
        lorentzian = height / (1 + ((x - idx) / (width / 10)) ** 2)
        spectrum += lorentzian
        
        markers[idx] = name
    
    # Add noise
    spectrum += np.random.randn(n_features) * 0.05
    spectrum = np.maximum(spectrum, 0)  # No negative intensities
    
    return spectrum, wavenumbers, markers


def generate_importance_scores(n_features, markers=None):
    """Generate importance scores emphasizing marker bands."""
    importance = np.random.rand(n_features) * 0.3
    
    # Boost importance for marker bands
    if markers:
        for idx in markers.keys():
            # Create Gaussian around marker position
            x = np.arange(n_features)
            gaussian = np.exp(-((x - idx) ** 2) / (30 ** 2))
            importance += gaussian * 0.5
    
    return importance


def demo_importance_overlay_basic():
    """Demo 1: Basic importance overlay with default settings."""
    print("\n" + "="*60)
    print("Demo 1: Importance Overlay - Basic")
    print("="*60)
    
    spectrum, wavenumbers, markers = generate_synthetic_spectrum()
    importance = generate_importance_scores(len(spectrum), markers)
    
    output_dir = Path("outputs/interpretability_demo/importance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating overlay style with cosine colormap...")
    fig = plot_importance_overlay(
        spectrum,
        importance,
        wavenumbers=wavenumbers,
        style="overlay",
        colormap="RdYlGn",
        highlight_peaks=True,
        n_peaks=5,
        save_path=output_dir / "overlay_basic.png",
    )
    print(f"✓ Saved: {output_dir / 'overlay_basic.png'}")
    plt.close(fig)


def demo_importance_overlay_styles():
    """Demo 2: Compare different visualization styles."""
    print("\n" + "="*60)
    print("Demo 2: Importance Overlay - Different Styles")
    print("="*60)
    
    spectrum, wavenumbers, markers = generate_synthetic_spectrum()
    importance = generate_importance_scores(len(spectrum), markers)
    
    output_dir = Path("outputs/interpretability_demo/importance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    styles = ["overlay", "bar", "heat"]
    colormaps = ["RdYlGn", "viridis", "plasma"]
    
    for style, cmap in zip(styles, colormaps):
        print(f"\nGenerating {style} style with {cmap} colormap...")
        fig = plot_importance_overlay(
            spectrum,
            importance,
            wavenumbers=wavenumbers,
            style=style,
            colormap=cmap,
            highlight_peaks=True,
            n_peaks=5,
            save_path=output_dir / f"overlay_{style}.png",
        )
        print(f"✓ Saved: {output_dir / f'overlay_{style}.png'}")
        plt.close(fig)


def demo_importance_with_band_names():
    """Demo 3: Importance overlay with chemical band names."""
    print("\n" + "="*60)
    print("Demo 3: Importance Overlay - With Band Names")
    print("="*60)
    
    spectrum, wavenumbers, markers = generate_synthetic_spectrum()
    importance = generate_importance_scores(len(spectrum), markers)
    
    output_dir = Path("outputs/interpretability_demo/importance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating overlay with chemical band names...")
    fig = plot_importance_overlay(
        spectrum,
        importance,
        wavenumbers=wavenumbers,
        style="overlay",
        colormap="coolwarm",
        highlight_peaks=True,
        n_peaks=5,
        band_names=markers,
        save_path=output_dir / "overlay_named.png",
    )
    print(f"✓ Saved: {output_dir / 'overlay_named.png'}")
    plt.close(fig)


def demo_marker_bands_basic():
    """Demo 4: Basic marker bands visualization."""
    print("\n" + "="*60)
    print("Demo 4: Marker Bands - Basic Highlighting")
    print("="*60)
    
    spectrum, wavenumbers, markers = generate_synthetic_spectrum()
    
    output_dir = Path("outputs/interpretability_demo/markers")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating marker bands visualization...")
    fig = plot_marker_bands(
        spectrum,
        markers,
        wavenumbers=wavenumbers,
        show_peak_heights=True,
        save_path=output_dir / "markers_basic.png",
    )
    print(f"✓ Saved: {output_dir / 'markers_basic.png'}")
    plt.close(fig)


def demo_marker_bands_with_importance():
    """Demo 5: Marker bands with importance scores."""
    print("\n" + "="*60)
    print("Demo 5: Marker Bands - With Importance Scores")
    print("="*60)
    
    spectrum, wavenumbers, markers = generate_synthetic_spectrum()
    importance = generate_importance_scores(len(spectrum), markers)
    
    output_dir = Path("outputs/interpretability_demo/markers")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract importance for marker bands
    marker_indices = list(markers.keys())
    band_importance = importance[marker_indices]
    
    print("\nGenerating marker bands with importance scores...")
    fig = plot_marker_bands(
        spectrum,
        markers,
        wavenumbers=wavenumbers,
        band_importance=band_importance,
        show_peak_heights=True,
        colormap="RdYlGn",
        save_path=output_dir / "markers_importance.png",
    )
    print(f"✓ Saved: {output_dir / 'markers_importance.png'}")
    plt.close(fig)


def demo_marker_bands_custom_colors():
    """Demo 6: Marker bands with custom colors."""
    print("\n" + "="*60)
    print("Demo 6: Marker Bands - Custom Colors")
    print("="*60)
    
    spectrum, wavenumbers, markers = generate_synthetic_spectrum()
    
    output_dir = Path("outputs/interpretability_demo/markers")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define custom colors for each marker band
    custom_colors = {
        list(markers.keys())[0]: "red",
        list(markers.keys())[1]: "blue",
        list(markers.keys())[2]: "green",
        list(markers.keys())[3]: "orange",
        list(markers.keys())[4]: "purple",
    }
    
    print("\nGenerating marker bands with custom colors...")
    fig = plot_marker_bands(
        spectrum,
        markers,
        wavenumbers=wavenumbers,
        colors=custom_colors,
        show_peak_heights=True,
        save_path=output_dir / "markers_custom_colors.png",
    )
    print(f"✓ Saved: {output_dir / 'markers_custom_colors.png'}")
    plt.close(fig)


def demo_band_statistics():
    """Demo 7: Extract and display band statistics."""
    print("\n" + "="*60)
    print("Demo 7: Band Statistics Extraction")
    print("="*60)
    
    spectrum, wavenumbers, markers = generate_synthetic_spectrum()
    importance = generate_importance_scores(len(spectrum), markers)
    
    print("\nExtracting statistics for marker bands...")
    stats = get_band_statistics(
        spectrum,
        importance=importance,
        bands_of_interest=list(markers.keys()),
        wavenumbers=wavenumbers,
    )
    
    print("\nBand Statistics Summary:")
    print("-" * 60)
    for band_key in sorted(stats.keys()):
        band_stats = stats[band_key]
        band_idx = int(band_key.split("_")[1])
        band_name = markers.get(band_idx, "Unknown")
        
        print(f"\n{band_name} (Band {band_idx}):")
        print(f"  Intensity:       {band_stats['intensity']:.4f}")
        if 'wavenumber' in band_stats:
            print(f"  Wavenumber:      {band_stats['wavenumber']:.1f} cm⁻¹")
        if 'importance' in band_stats:
            print(f"  Importance:      {band_stats['importance']:.4f}")
            print(f"  Importance Rank: {band_stats['importance_rank']}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("INTERPRETABILITY VISUALIZATIONS DEMO")
    print("="*60)
    print("\nThis demo showcases:")
    print("  • Feature importance overlay on spectra")
    print("  • Multiple visualization styles (overlay, bar, heat)")
    print("  • Marker band highlighting with colors")
    print("  • Integration with importance metrics")
    print("  • Band statistics extraction")
    
    try:
        # Run demos
        demo_importance_overlay_basic()
        demo_importance_overlay_styles()
        demo_importance_with_band_names()
        demo_marker_bands_basic()
        demo_marker_bands_with_importance()
        demo_marker_bands_custom_colors()
        demo_band_statistics()
        
        # Summary
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETE!")
        print("="*60)
        print("\nGenerated visualizations:")
        print("  • outputs/interpretability_demo/importance/")
        print("    - overlay_basic.png")
        print("    - overlay_overlay.png")
        print("    - overlay_bar.png")
        print("    - overlay_heat.png")
        print("    - overlay_named.png")
        print("  • outputs/interpretability_demo/markers/")
        print("    - markers_basic.png")
        print("    - markers_importance.png")
        print("    - markers_custom_colors.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
