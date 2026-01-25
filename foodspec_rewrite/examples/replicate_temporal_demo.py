#!/usr/bin/env python
"""
Replicate Similarity and Temporal Drift Demo
=============================================

Demonstrates the use of plot_replicate_similarity() and plot_temporal_drift()
for quality control and time-series monitoring of spectral data.

Features:
- Replicate similarity heatmaps (cosine and correlation metrics)
- Hierarchical clustering of replicates
- Temporal drift visualization with rolling averages
- Multi-band time series monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

from foodspec.viz.drift import (
    plot_replicate_similarity,
    plot_temporal_drift,
)


def generate_replicate_data():
    """Generate synthetic replicate data with varying similarity."""
    np.random.seed(42)
    n_features = 200
    
    # Create 4 groups of replicates with different characteristics
    group_a = np.random.randn(3, n_features) + np.sin(np.linspace(0, 2*np.pi, n_features))
    group_b = np.random.randn(3, n_features) + np.cos(np.linspace(0, 2*np.pi, n_features))
    group_c = np.random.randn(3, n_features) + np.linspace(0, 1, n_features)
    group_d = np.random.randn(3, n_features) + np.linspace(1, 0, n_features)
    
    spectra = np.vstack([group_a, group_b, group_c, group_d])
    labels = [f"A{i+1}" for i in range(3)] + [f"B{i+1}" for i in range(3)] + \
             [f"C{i+1}" for i in range(3)] + [f"D{i+1}" for i in range(3)]
    
    return spectra, labels


def generate_temporal_data():
    """Generate synthetic temporal data with drift patterns."""
    np.random.seed(42)
    n_samples = 30
    n_features = 200
    
    # Generate time series (30 samples over 30 days)
    timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]
    
    # Create spectra with temporal drift
    base_spectrum = np.sin(np.linspace(0, 4*np.pi, n_features))
    spectra = []
    
    for i in range(n_samples):
        # Add progressive drift
        drift = 0.1 * i / n_samples
        noise = np.random.randn(n_features) * 0.1
        spectrum = base_spectrum + drift + noise
        spectra.append(spectrum)
    
    spectra = np.array(spectra)
    
    # Create metadata
    meta = {
        "timestamp": timestamps,
        "batch": [f"B{i//10 + 1}" for i in range(n_samples)],
    }
    
    # Create wavenumbers
    wavenumbers = np.linspace(1000, 3000, n_features)
    
    return spectra, meta, wavenumbers


def demo_replicate_similarity():
    """Demonstrate replicate similarity visualization."""
    print("\n" + "="*60)
    print("Demo 1: Replicate Similarity Heatmaps")
    print("="*60)
    
    spectra, labels = generate_replicate_data()
    output_dir = Path("outputs/replicate_temporal_demo/similarity")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cosine similarity with clustering
    print("\n1. Cosine similarity with hierarchical clustering")
    fig = plot_replicate_similarity(
        spectra,
        labels=labels,
        metric="cosine",
        cluster=True,
        save_path=output_dir / "cosine_clustered.png",
        figure_size=(12, 10),
    )
    print(f"   ✓ Saved: {output_dir / 'cosine_clustered.png'}")
    plt.close(fig)
    
    # 2. Correlation similarity without clustering
    print("\n2. Correlation similarity without clustering")
    fig = plot_replicate_similarity(
        spectra,
        labels=labels,
        metric="correlation",
        cluster=False,
        save_path=output_dir / "correlation_original.png",
        figure_size=(12, 10),
    )
    print(f"   ✓ Saved: {output_dir / 'correlation_original.png'}")
    plt.close(fig)
    
    print("\n✅ Replicate similarity demo complete!")
    print(f"   Output directory: {output_dir}")


def demo_temporal_drift():
    """Demonstrate temporal drift visualization."""
    print("\n" + "="*60)
    print("Demo 2: Temporal Drift Visualization")
    print("="*60)
    
    spectra, meta, wavenumbers = generate_temporal_data()
    output_dir = Path("outputs/replicate_temporal_demo/temporal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Specific band indices
    print("\n1. Temporal drift for specific bands")
    band_indices = [20, 50, 100, 150, 180]
    fig = plot_temporal_drift(
        spectra,
        meta,
        time_key="timestamp",
        band_indices=band_indices,
        save_path=output_dir / "bands_specific.png",
        figure_size=(14, 8),
    )
    print(f"   ✓ Saved: {output_dir / 'bands_specific.png'}")
    plt.close(fig)
    
    # 2. Wavenumber ranges with rolling average
    print("\n2. Temporal drift with wavenumber ranges (smoothed)")
    band_ranges = [(1200, 1400), (1800, 2000), (2400, 2600)]
    fig = plot_temporal_drift(
        spectra,
        meta,
        time_key="timestamp",
        wavenumbers=wavenumbers,
        band_ranges=band_ranges,
        rolling_window=5,
        save_path=output_dir / "ranges_smoothed.png",
        figure_size=(14, 8),
    )
    print(f"   ✓ Saved: {output_dir / 'ranges_smoothed.png'}")
    plt.close(fig)
    
    # 3. Auto band selection
    print("\n3. Temporal drift with auto-selected bands")
    fig = plot_temporal_drift(
        spectra,
        meta,
        time_key="timestamp",
        save_path=output_dir / "bands_auto.png",
        figure_size=(14, 8),
    )
    print(f"   ✓ Saved: {output_dir / 'bands_auto.png'}")
    plt.close(fig)
    
    print("\n✅ Temporal drift demo complete!")
    print(f"   Output directory: {output_dir}")


def demo_combined_workflow():
    """Demonstrate combined replicate similarity and temporal analysis."""
    print("\n" + "="*60)
    print("Demo 3: Combined Replicate + Temporal Analysis")
    print("="*60)
    
    # Generate data with both replicate structure and temporal drift
    np.random.seed(42)
    n_timepoints = 10
    n_replicates = 3
    n_features = 200
    
    timestamps = [datetime(2024, 1, 1) + timedelta(days=i*3) for i in range(n_timepoints)]
    
    # Create replicate data at each timepoint
    all_spectra = []
    all_labels = []
    all_timestamps = []
    
    for t, timestamp in enumerate(timestamps):
        # Base spectrum changes over time
        base = np.sin(np.linspace(0, 4*np.pi, n_features)) + 0.05 * t
        
        # Add replicates with small variations
        for r in range(n_replicates):
            spectrum = base + np.random.randn(n_features) * 0.1
            all_spectra.append(spectrum)
            all_labels.append(f"T{t+1}R{r+1}")
            all_timestamps.append(timestamp)
    
    all_spectra = np.array(all_spectra)
    meta = {"timestamp": all_timestamps}
    
    output_dir = Path("outputs/replicate_temporal_demo/combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Replicate similarity across all timepoints
    print("\n1. Replicate similarity across time")
    fig = plot_replicate_similarity(
        all_spectra,
        labels=all_labels,
        metric="cosine",
        cluster=True,
        save_path=output_dir / "replicates_across_time.png",
        figure_size=(14, 12),
    )
    print(f"   ✓ Saved: {output_dir / 'replicates_across_time.png'}")
    plt.close(fig)
    
    # 2. Temporal drift with smoothing
    print("\n2. Temporal drift with rolling average")
    fig = plot_temporal_drift(
        all_spectra,
        meta,
        time_key="timestamp",
        band_indices=[50, 100, 150],
        rolling_window=3,
        save_path=output_dir / "temporal_drift_smoothed.png",
        figure_size=(14, 8),
    )
    print(f"   ✓ Saved: {output_dir / 'temporal_drift_smoothed.png'}")
    plt.close(fig)
    
    print("\n✅ Combined workflow demo complete!")
    print(f"   Output directory: {output_dir}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("REPLICATE SIMILARITY AND TEMPORAL DRIFT DEMO")
    print("="*60)
    print("\nThis demo showcases:")
    print("  • Replicate similarity heatmaps (cosine & correlation)")
    print("  • Hierarchical clustering of replicates")
    print("  • Temporal drift visualization")
    print("  • Rolling average smoothing")
    print("  • Combined replicate + temporal analysis")
    
    try:
        # Run demos
        demo_replicate_similarity()
        demo_temporal_drift()
        demo_combined_workflow()
        
        # Summary
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETE!")
        print("="*60)
        print("\nGenerated visualizations:")
        print("  • outputs/replicate_temporal_demo/similarity/")
        print("    - cosine_clustered.png")
        print("    - correlation_original.png")
        print("  • outputs/replicate_temporal_demo/temporal/")
        print("    - bands_specific.png")
        print("    - ranges_smoothed.png")
        print("    - bands_auto.png")
        print("  • outputs/replicate_temporal_demo/combined/")
        print("    - replicates_across_time.png")
        print("    - temporal_drift_smoothed.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
