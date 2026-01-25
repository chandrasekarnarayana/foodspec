#!/usr/bin/env python3
"""
Batch Drift and Stage Difference Demo
======================================

Demonstrates spectral drift analysis and stage-wise comparisons:
- Batch drift detection with confidence bands
- Stage-wise difference visualization
- Statistical analysis of spectral variations
"""

from pathlib import Path

import numpy as np

from foodspec.viz import (
    plot_batch_drift,
    plot_stage_differences,
    get_batch_statistics,
    get_stage_statistics,
)


def create_demo_output_dir():
    """Create directory for demo outputs."""
    output_dir = Path("outputs/drift_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_synthetic_batch_data():
    """Create synthetic spectral data with batch effects."""
    np.random.seed(42)
    
    # Simulate Raman spectra (wavenumbers 400-4000 cm⁻¹)
    n_features = 200
    wavenumbers = np.linspace(400, 4000, n_features)
    
    # Three batches with different drift patterns
    batches = []
    batch_labels = []
    
    for batch_id in range(3):
        n_samples = 40 + batch_id * 5  # Different sample sizes
        
        # Base spectrum (Gaussian peaks)
        base = np.zeros(n_features)
        for peak_pos in [1000, 1600, 2900]:
            peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
            base += np.exp(-((np.arange(n_features) - peak_idx) ** 2) / 200)
        
        # Add batch-specific drift
        batch_offset = batch_id * 0.15  # Increasing baseline
        batch_scale = 1.0 + batch_id * 0.1  # Increasing intensity
        
        # Generate samples with noise
        batch_spectra = []
        for _ in range(n_samples):
            spectrum = base * batch_scale + batch_offset
            spectrum += np.random.randn(n_features) * 0.05  # Noise
            batch_spectra.append(spectrum)
        
        batches.append(np.array(batch_spectra))
        batch_labels.extend([f"Batch_{batch_id+1}"] * n_samples)
    
    spectra = np.vstack(batches)
    meta = {"batch": np.array(batch_labels)}
    
    return spectra, meta, wavenumbers


def create_synthetic_stage_data():
    """Create synthetic spectral data for processing stages."""
    np.random.seed(42)
    
    n_samples = 60
    n_features = 200
    wavenumbers = np.linspace(400, 4000, n_features)
    
    # Generate base spectrum
    base = np.zeros(n_features)
    for peak_pos in [800, 1400, 2800, 3200]:
        peak_idx = np.argmin(np.abs(wavenumbers - peak_pos))
        base += np.exp(-((np.arange(n_features) - peak_idx) ** 2) / 150)
    
    # Stage 1: Raw with baseline drift
    baseline = 0.3 * np.sin(np.linspace(0, 4 * np.pi, n_features))
    raw = []
    for _ in range(n_samples):
        spectrum = base + baseline + np.random.randn(n_features) * 0.08
        raw.append(spectrum)
    
    # Stage 2: Baseline corrected
    baseline_corrected = []
    for _ in range(n_samples):
        spectrum = base + np.random.randn(n_features) * 0.05
        baseline_corrected.append(spectrum)
    
    # Stage 3: Normalized (unit vector)
    normalized = []
    for _ in range(n_samples):
        spectrum = base + np.random.randn(n_features) * 0.04
        norm = np.linalg.norm(spectrum)
        if norm > 0:
            spectrum = spectrum / norm
        normalized.append(spectrum)
    
    # Stage 4: Smoothed (moving average)
    smoothed = []
    window_size = 5
    for _ in range(n_samples):
        spectrum = base + np.random.randn(n_features) * 0.03
        norm = np.linalg.norm(spectrum)
        if norm > 0:
            spectrum = spectrum / norm
        # Apply smoothing
        smoothed_spectrum = np.convolve(
            spectrum, np.ones(window_size) / window_size, mode="same"
        )
        smoothed.append(smoothed_spectrum)
    
    spectra_by_stage = {
        "raw": np.array(raw),
        "baseline_corrected": np.array(baseline_corrected),
        "normalized": np.array(normalized),
        "smoothed": np.array(smoothed),
    }
    
    return spectra_by_stage, wavenumbers


def demo_batch_drift_basic():
    """Demo 1: Basic batch drift analysis."""
    print("\n" + "=" * 70)
    print("DEMO 1: Batch Drift Analysis")
    print("=" * 70)
    
    spectra, meta, wavenumbers = create_synthetic_batch_data()
    
    print(f"\nData shape: {spectra.shape}")
    print(f"Batches: {np.unique(meta['batch'])}")
    
    # Get statistics
    stats = get_batch_statistics(spectra, meta)
    summary = stats["summary"]
    
    print(f"\nBatch Statistics:")
    print(f"  Total batches: {summary['total_batches']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Samples per batch:")
    for batch, count in summary["samples_per_batch"].items():
        print(f"    {batch}: {count}")
    print(f"  Max pairwise difference: {summary['max_pairwise_difference']:.4f}")
    print(f"  Max difference pair: {summary['max_difference_pair']}")
    
    # Generate plot
    output_dir = create_demo_output_dir()
    fig = plot_batch_drift(
        spectra,
        meta,
        wavenumbers=wavenumbers,
        confidence=0.95,
        save_path=output_dir / "batch_drift_basic",
    )
    
    print(f"\n✓ Batch drift plot saved: {output_dir / 'batch_drift_basic'}")
    
    return fig


def demo_batch_drift_reference():
    """Demo 2: Batch drift with specified reference."""
    print("\n" + "=" * 70)
    print("DEMO 2: Batch Drift with Custom Reference")
    print("=" * 70)
    
    spectra, meta, wavenumbers = create_synthetic_batch_data()
    
    # Use Batch_2 as reference
    reference = "Batch_2"
    print(f"\nUsing {reference} as reference batch")
    
    # Generate plot
    output_dir = create_demo_output_dir()
    fig = plot_batch_drift(
        spectra,
        meta,
        wavenumbers=wavenumbers,
        reference_batch=reference,
        confidence=0.99,  # 99% confidence
        save_path=output_dir / "batch_drift_reference",
    )
    
    print(f"✓ Plot saved with 99% confidence bands")
    print(f"  Path: {output_dir / 'batch_drift_reference'}")
    
    return fig


def demo_stage_differences_basic():
    """Demo 3: Stage-wise difference analysis."""
    print("\n" + "=" * 70)
    print("DEMO 3: Stage Difference Analysis")
    print("=" * 70)
    
    spectra_by_stage, wavenumbers = create_synthetic_stage_data()
    
    print(f"\nStages: {list(spectra_by_stage.keys())}")
    print(f"Samples per stage:")
    for stage, spectra in spectra_by_stage.items():
        print(f"  {stage}: {spectra.shape[0]}")
    
    # Get statistics
    stats = get_stage_statistics(spectra_by_stage)
    summary = stats["summary"]
    
    print(f"\nStage Statistics:")
    print(f"  Total stages: {summary['total_stages']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Baseline stage: {summary['baseline_stage']}")
    print(f"  Max difference from baseline: {summary['max_difference_from_baseline']:.4f}")
    print(f"  Max difference stage: {summary['max_difference_stage']}")
    
    # Generate plot
    output_dir = create_demo_output_dir()
    fig = plot_stage_differences(
        spectra_by_stage,
        wavenumbers=wavenumbers,
        save_path=output_dir / "stage_differences_basic",
    )
    
    print(f"\n✓ Stage differences plot saved: {output_dir / 'stage_differences_basic'}")
    
    return fig


def demo_stage_differences_custom():
    """Demo 4: Stage differences with custom baseline and order."""
    print("\n" + "=" * 70)
    print("DEMO 4: Stage Differences with Custom Settings")
    print("=" * 70)
    
    spectra_by_stage, wavenumbers = create_synthetic_stage_data()
    
    # Custom baseline and stage order
    baseline = "baseline_corrected"
    stage_order = ["baseline_corrected", "normalized", "smoothed", "raw"]
    
    print(f"\nCustom baseline: {baseline}")
    print(f"Custom order: {stage_order}")
    
    # Generate plot
    output_dir = create_demo_output_dir()
    fig = plot_stage_differences(
        spectra_by_stage,
        wavenumbers=wavenumbers,
        baseline_stage=baseline,
        stage_order=stage_order,
        save_path=output_dir / "stage_differences_custom",
    )
    
    print(f"\n✓ Custom stage plot saved: {output_dir / 'stage_differences_custom'}")
    
    return fig


def demo_combined_analysis():
    """Demo 5: Combined batch and stage analysis."""
    print("\n" + "=" * 70)
    print("DEMO 5: Combined Batch and Stage Analysis")
    print("=" * 70)
    
    # Batch analysis
    batch_spectra, batch_meta, batch_wn = create_synthetic_batch_data()
    batch_stats = get_batch_statistics(batch_spectra, batch_meta)
    
    print("\nBatch Analysis Summary:")
    print(f"  Batches detected: {batch_stats['summary']['total_batches']}")
    print(f"  Max drift: {batch_stats['summary']['max_pairwise_difference']:.4f}")
    
    # Stage analysis
    stage_spectra, stage_wn = create_synthetic_stage_data()
    stage_stats = get_stage_statistics(stage_spectra)
    
    print("\nStage Analysis Summary:")
    print(f"  Processing stages: {stage_stats['summary']['total_stages']}")
    print(f"  Baseline: {stage_stats['summary']['baseline_stage']}")
    print(f"  Max change: {stage_stats['summary']['max_difference_from_baseline']:.4f}")
    
    # Generate both plots
    output_dir = create_demo_output_dir()
    
    fig1 = plot_batch_drift(
        batch_spectra,
        batch_meta,
        wavenumbers=batch_wn,
        save_path=output_dir / "combined_batch",
    )
    
    fig2 = plot_stage_differences(
        stage_spectra,
        wavenumbers=stage_wn,
        save_path=output_dir / "combined_stage",
    )
    
    print(f"\n✓ Combined analysis saved:")
    print(f"  Batch: {output_dir / 'combined_batch'}")
    print(f"  Stage: {output_dir / 'combined_stage'}")
    
    return fig1, fig2


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("FoodSpec Drift and Stage Difference Demo")
    print("=" * 70)
    print("\nDemonstrates batch drift and stage difference visualizations")
    print("for spectral quality control and processing analysis.")
    
    # Run demos
    demo_batch_drift_basic()
    demo_batch_drift_reference()
    demo_stage_differences_basic()
    demo_stage_differences_custom()
    demo_combined_analysis()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nGenerated 6 visualization sets:")
    print("  1. outputs/drift_demo/batch_drift_basic/")
    print("  2. outputs/drift_demo/batch_drift_reference/")
    print("  3. outputs/drift_demo/stage_differences_basic/")
    print("  4. outputs/drift_demo/stage_differences_custom/")
    print("  5. outputs/drift_demo/combined_batch/")
    print("  6. outputs/drift_demo/combined_stage/")
    
    print("\nBatch Drift Visualization:")
    print("  • Top panel: Mean spectra per batch with confidence bands")
    print("  • Bottom panel: Differences from reference batch")
    print("  • Auto-selects reference (most samples) or use custom")
    print("  • Configurable confidence levels (95% or 99%)")
    
    print("\nStage Difference Visualization:")
    print("  • Top panel: Mean spectra per processing stage")
    print("  • Bottom panel: Differences from baseline stage")
    print("  • Auto-selects baseline ('raw' or most samples)")
    print("  • Custom stage ordering supported")
    
    print("\nUse Cases:")
    print("  • Batch-to-batch quality control")
    print("  • Instrument drift detection")
    print("  • Processing pipeline validation")
    print("  • Method transfer verification")
    print("  • Preprocessing effect evaluation")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
