"""
Demonstration script for processing stages visualization.

Showcases multi-stage spectral preprocessing visualization with:
- Multi-stage overlay with preprocessing names
- Zoom windows highlighting specific spectral regions
- Before/after preprocessing comparisons
- Statistics extraction for preprocessing impact
"""

from pathlib import Path
import sys

# Ensure src is in path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from foodspec.viz.processing_stages import (
    get_processing_statistics,
    plot_preprocessing_comparison,
    plot_processing_stages,
)


def generate_synthetic_spectral_data(n_points=1000, seed=42):
    """Generate synthetic spectral data."""
    np.random.seed(seed)
    wavenumbers = np.linspace(400, 4000, n_points)
    
    # Create realistic spectral features
    raw = (
        2 * np.sin(wavenumbers / 250)
        + 0.5 * np.sin(wavenumbers / 100)
        + np.random.normal(0, 0.15, n_points)
        + 0.01 * wavenumbers  # Baseline drift
    )
    
    return wavenumbers, raw


def demo_basic_multi_stage():
    """Basic multi-stage overlay visualization."""
    print("\n=== Demo 1: Basic Multi-Stage Overlay ===")
    wavenumbers, raw = generate_synthetic_spectral_data(n_points=800)
    
    # Create processing pipeline
    baseline_corrected = raw - np.polyfit(wavenumbers, raw, 2)[2]
    normalized = baseline_corrected / np.std(baseline_corrected)
    
    fig = plot_processing_stages(
        wavenumbers,
        stages_data={
            "raw": raw,
            "baseline_corrected": baseline_corrected,
            "normalized": normalized,
        },
        stage_names=["Raw", "Baseline Corrected", "Normalized"],
        title="Multi-Stage Spectral Processing",
        save_path=Path("outputs/processing_demo") / "01_basic_multistage.png",
    )
    print("✓ Generated: 01_basic_multistage.png")
    return fig


def demo_with_zoom_single():
    """Multi-stage overlay with single zoom window."""
    print("\n=== Demo 2: Multi-Stage with Single Zoom ===")
    wavenumbers, raw = generate_synthetic_spectral_data(n_points=1000)
    
    baseline_corrected = raw - np.polyfit(wavenumbers, raw, 2)[2]
    normalized = baseline_corrected / np.std(baseline_corrected)
    
    fig = plot_processing_stages(
        wavenumbers,
        stages_data={
            "raw": raw,
            "baseline": baseline_corrected,
            "normalized": normalized,
        },
        stage_names=["Raw", "Baseline Corrected", "Normalized"],
        zoom_regions=[(1500, 1700)],
        colormap="viridis",
        title="Processing with Detail Zoom",
        save_path=Path("outputs/processing_demo") / "02_zoom_single.png",
    )
    print("✓ Generated: 02_zoom_single.png")
    return fig


def demo_with_zoom_multiple():
    """Multi-stage overlay with multiple zoom windows."""
    print("\n=== Demo 3: Multi-Stage with Multiple Zooms ===")
    wavenumbers, raw = generate_synthetic_spectral_data(n_points=1200)
    
    baseline_corrected = raw - np.polyfit(wavenumbers, raw, 2)[2]
    normalized = baseline_corrected / np.std(baseline_corrected)
    smoothed = np.convolve(normalized, np.ones(5)/5, mode='same')
    
    fig = plot_processing_stages(
        wavenumbers,
        stages_data={
            "raw": raw,
            "baseline": baseline_corrected,
            "normalized": normalized,
            "smoothed": smoothed,
        },
        stage_names=["Raw", "Baseline Corrected", "Normalized", "Smoothed"],
        zoom_regions=[(900, 1100), (2200, 2400), (3300, 3500)],
        colormap="plasma",
        linewidth=1.5,
        title="Processing Pipeline with Three Detail Views",
        figure_size=(16, 10),
        save_path=Path("outputs/processing_demo") / "03_zoom_multiple.png",
    )
    print("✓ Generated: 03_zoom_multiple.png")
    return fig


def demo_baseline_correction():
    """Before/after baseline correction comparison."""
    print("\n=== Demo 4: Baseline Correction Comparison ===")
    wavenumbers, raw = generate_synthetic_spectral_data(n_points=900)
    
    # Estimate baseline
    baseline = np.polyfit(wavenumbers, raw, 2)
    baseline_curve = np.polyval(baseline, wavenumbers)
    corrected = raw - baseline_curve
    
    fig = plot_preprocessing_comparison(
        wavenumbers,
        before_spectrum=raw,
        after_spectrum=corrected,
        preprocessing_name="Baseline Correction",
        show_difference=True,
        color_before="steelblue",
        color_after="coral",
        title="Impact of Baseline Correction",
        save_path=Path("outputs/processing_demo") / "04_baseline_comparison.png",
    )
    print("✓ Generated: 04_baseline_comparison.png")
    return fig


def demo_normalization():
    """Before/after normalization comparison."""
    print("\n=== Demo 5: Normalization Comparison ===")
    wavenumbers, raw = generate_synthetic_spectral_data(n_points=900)
    
    baseline_corrected = raw - np.polyfit(wavenumbers, raw, 2)[2]
    normalized = baseline_corrected / np.std(baseline_corrected)
    
    fig = plot_preprocessing_comparison(
        wavenumbers,
        before_spectrum=baseline_corrected,
        after_spectrum=normalized,
        preprocessing_name="Vector Normalization",
        show_difference=True,
        color_before="forestgreen",
        color_after="gold",
        title="Impact of Normalization",
        save_path=Path("outputs/processing_demo") / "05_normalization_comparison.png",
    )
    print("✓ Generated: 05_normalization_comparison.png")
    return fig


def demo_smoothing():
    """Before/after smoothing comparison."""
    print("\n=== Demo 6: Smoothing Comparison ===")
    wavenumbers, raw = generate_synthetic_spectral_data(n_points=1000)
    
    baseline_corrected = raw - np.polyfit(wavenumbers, raw, 2)[2]
    
    # Savitzky-Golay style smoothing
    window_size = 11
    smoothed = np.convolve(baseline_corrected, np.ones(window_size)/window_size, mode='same')
    
    fig = plot_preprocessing_comparison(
        wavenumbers,
        before_spectrum=baseline_corrected,
        after_spectrum=smoothed,
        preprocessing_name="Savitzky-Golay Smoothing",
        show_difference=True,
        color_before="purple",
        color_after="orange",
        title="Impact of Spectral Smoothing",
        save_path=Path("outputs/processing_demo") / "06_smoothing_comparison.png",
    )
    print("✓ Generated: 06_smoothing_comparison.png")
    return fig


def demo_statistics_extraction():
    """Statistics extraction for preprocessing impact."""
    print("\n=== Demo 7: Preprocessing Statistics ===")
    wavenumbers, raw = generate_synthetic_spectral_data(n_points=800)
    
    baseline_corrected = raw - np.polyfit(wavenumbers, raw, 2)[2]
    normalized = baseline_corrected / np.std(baseline_corrected)
    
    stages_data = {
        "Raw": raw,
        "Baseline Corrected": baseline_corrected,
        "Normalized": normalized,
    }
    
    stats = get_processing_statistics(stages_data)
    
    print("\nStatistics by Stage:")
    print("-" * 60)
    for stage_name, stage_stats in stats.items():
        print(f"\n{stage_name}:")
        print(f"  Mean:     {stage_stats['mean']:10.4f}")
        print(f"  Std Dev:  {stage_stats['std']:10.4f}")
        print(f"  Min:      {stage_stats['min']:10.4f}")
        print(f"  Max:      {stage_stats['max']:10.4f}")
        print(f"  Median:   {stage_stats['median']:10.4f}")
        print(f"  Range:    {stage_stats['range']:10.4f}")
    
    # Visualize
    fig = plot_processing_stages(
        wavenumbers,
        stages_data=stages_data,
        stage_names=list(stages_data.keys()),
        zoom_regions=[(1200, 1500)],
        title="Processing Pipeline with Statistics",
        save_path=Path("outputs/processing_demo") / "07_statistics.png",
    )
    print("\n✓ Generated: 07_statistics.png")
    return fig


def demo_integrated_workflow():
    """Complete integrated preprocessing workflow."""
    print("\n=== Demo 8: Integrated Workflow ===")
    
    # Generate realistic multi-sample scenario
    n_samples = 3
    n_points = 1000
    wavenumbers = np.linspace(400, 4000, n_points)
    
    for sample_idx in range(n_samples):
        np.random.seed(100 + sample_idx)
        
        # Generate raw spectrum
        raw = (
            2 * np.sin(wavenumbers / 250)
            + 0.3 * np.cos(wavenumbers / 150)
            + np.random.normal(0, 0.1, n_points)
            + 0.008 * wavenumbers
        )
        
        # Apply preprocessing steps
        baseline_corrected = raw - np.polyfit(wavenumbers, raw, 2)[2]
        normalized = baseline_corrected / np.std(baseline_corrected)
        smoothed = np.convolve(normalized, np.ones(7)/7, mode='same')
        
        stages_data = {
            "Raw": raw,
            "Baseline": baseline_corrected,
            "Normalized": normalized,
            "Smoothed": smoothed,
        }
        
        # Get statistics
        stats = get_processing_statistics(stages_data)
        
        # Visualize
        fig = plot_processing_stages(
            wavenumbers,
            stages_data=stages_data,
            stage_names=list(stages_data.keys()),
            zoom_regions=[(1000, 1200), (2800, 3000)],
            colormap="coolwarm",
            title=f"Sample {sample_idx + 1}: Complete Processing Pipeline",
            figure_size=(16, 10),
            save_path=Path("outputs/processing_demo") / f"08_integrated_sample{sample_idx + 1}.png",
        )
        
        print(f"✓ Generated: 08_integrated_sample{sample_idx + 1}.png")
        print(f"  Mean intensity progression: {stats['Raw']['mean']:.3f} → {stats['Smoothed']['mean']:.3f}")


def main():
    """Run all demonstrations."""
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("FoodSpec: Processing Stages Visualization Demo")
    print("=" * 70)

    # Create output directory
    output_dir = Path("outputs/processing_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run demonstrations
        demo_basic_multi_stage()
        demo_with_zoom_single()
        demo_with_zoom_multiple()
        demo_baseline_correction()
        demo_normalization()
        demo_smoothing()
        demo_statistics_extraction()
        demo_integrated_workflow()

        print("\n" + "=" * 70)
        print(f"✓ All demonstrations completed successfully!")
        print(f"✓ Output directory: {output_dir.absolute()}")
        print(f"✓ Generated 11 visualizations")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        raise
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
