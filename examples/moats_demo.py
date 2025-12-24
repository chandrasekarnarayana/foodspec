"""
FoodSpec Moats Demo: Matrix Correction + Heating Trajectory + Calibration Transfer

This example demonstrates all three differentiating capabilities in a realistic workflow.

Scenario:
- Analyze heating degradation of chips vs. pure oil samples over time
- Correct for matrix effects
- Track oxidation indices and estimate shelf life
- Transfer calibration from reference to production instrument

Run:
    python examples/moats_demo.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from foodspec import FoodSpec
from foodspec.core.dataset import FoodSpectrumSet


def generate_synthetic_heating_data(n_samples=40, n_wavenumbers=500, n_timepoints=8):
    """Generate synthetic heating study data with matrix effects."""
    
    # Wavenumbers (Raman shift)
    wavenumbers = np.linspace(800, 1800, n_wavenumbers)
    
    # Baseline spectra (pure oil vs chips)
    baseline_oil = 1.0 + 0.1 * np.sin(2 * np.pi * (wavenumbers - 800) / 1000)
    baseline_chips = 1.5 + 0.3 * np.sin(2 * np.pi * (wavenumbers - 800) / 500)  # Different matrix effect
    
    # Time points (hours)
    time_hours = np.linspace(0, 168, n_timepoints)  # 1 week
    
    # Generate spectra
    spectra = []
    metadata_rows = []
    
    for i, t in enumerate(time_hours):
        # Degradation factor (increases with time)
        degradation = 1.0 + 0.5 * (t / time_hours[-1])  # 50% intensity increase over time
        
        # Oil samples
        for j in range(n_samples // 2):
            # Add peaks at key wavenumbers (degradation-sensitive)
            spectrum = baseline_oil.copy()
            spectrum += 0.2 * degradation * np.exp(-((wavenumbers - 840) ** 2) / 100)  # Peroxide
            spectrum += 0.5 / degradation * np.exp(-((wavenumbers - 1440) ** 2) / 100)  # CH2 (decreases)
            spectrum += 0.3 * degradation * np.exp(-((wavenumbers - 1660) ** 2) / 100)  # C=C (increases)
            spectrum += 0.05 * np.random.randn(n_wavenumbers)  # Noise
            
            spectra.append(spectrum)
            metadata_rows.append({
                "sample_id": f"oil_t{i}_s{j}",
                "matrix_type": "pure_oil",
                "time_hours": t,
                "degradation_stage": _stage_from_time(t),
            })
        
        # Chips samples (stronger matrix effect)
        for j in range(n_samples // 2):
            spectrum = baseline_chips.copy()
            spectrum += 0.2 * degradation * np.exp(-((wavenumbers - 840) ** 2) / 100)
            spectrum += 0.5 / degradation * np.exp(-((wavenumbers - 1440) ** 2) / 100)
            spectrum += 0.3 * degradation * np.exp(-((wavenumbers - 1660) ** 2) / 100)
            spectrum += 0.08 * np.random.randn(n_wavenumbers)  # More noise
            
            spectra.append(spectrum)
            metadata_rows.append({
                "sample_id": f"chips_t{i}_s{j}",
                "matrix_type": "chips",
                "time_hours": t,
                "degradation_stage": _stage_from_time(t),
            })
    
    X = np.array(spectra)
    metadata = pd.DataFrame(metadata_rows)
    
    return FoodSpectrumSet(
        x=X,
        wavenumbers=wavenumbers,
        metadata=metadata,
        modality="raman",
    )


def _stage_from_time(t):
    """Map time to degradation stage."""
    if t < 24:
        return "fresh"
    elif t < 72:
        return "early"
    elif t < 120:
        return "advanced"
    else:
        return "spoiled"


def generate_transfer_standards(base_spectra, n_standards=20):
    """Generate paired source/target standards for calibration transfer."""
    
    # Select random samples as standards
    indices = np.random.choice(len(base_spectra.x), n_standards, replace=False)
    source_std = base_spectra.x[indices]
    
    # Target instrument has systematic bias (baseline shift + intensity scaling)
    target_std = 1.1 * source_std + 0.2 + 0.05 * np.random.randn(*source_std.shape)
    
    return source_std, target_std


def main():
    print("=" * 70)
    print("FoodSpec Moats Demo: Full Workflow")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic heating study data...")
    dataset = generate_synthetic_heating_data(n_samples=40, n_wavenumbers=500, n_timepoints=8)
    print(f"  ✓ Dataset: {len(dataset)} samples × {dataset.x.shape[1]} wavenumbers")
    print(f"  ✓ Matrices: {dataset.metadata['matrix_type'].unique().tolist()}")
    print(f"  ✓ Time range: {dataset.metadata['time_hours'].min():.0f}–{dataset.metadata['time_hours'].max():.0f} hours")
    
    # Step 2: Matrix Correction
    print("\n[2/5] Applying matrix correction...")
    fs = FoodSpec(dataset, output_dir="./moats_demo_output")
    fs.apply_matrix_correction(
        method="adaptive_baseline",
        scaling="median_mad",
        domain_adapt=True,
        matrix_column="matrix_type"
    )
    
    mc_metrics = fs.bundle.metrics.get("matrix_correction_matrix_effect_magnitude", {})
    print(f"  ✓ Correction magnitude: {mc_metrics.get('total_correction_magnitude', 0):.3f}")
    print(f"  ✓ Per-matrix correction: {mc_metrics.get('per_matrix_correction', {})}")
    
    # Step 3: Heating Trajectory Analysis
    print("\n[3/5] Analyzing heating trajectory...")
    traj_results = fs.analyze_heating_trajectory(
        time_column="time_hours",
        indices=["pi", "tfc", "oit_proxy"],
        classify_stages=True,
        stage_column="degradation_stage",
        estimate_shelf_life=True,
        shelf_life_threshold=0.4,  # PI threshold for "early" stage
        shelf_life_index="pi"
    )
    
    print(f"  ✓ Indices extracted: {list(traj_results['indices'].columns)}")
    
    # Print trajectory model quality
    print("  ✓ Trajectory models:")
    for idx, metrics in traj_results["trajectory_models"].items():
        print(f"    - {idx}: R²={metrics['r_squared']:.3f}, trend={metrics['trend_direction']}")
    
    # Stage classification
    if "stage_classification" in traj_results:
        class_metrics = traj_results["stage_classification"]["metrics"]
        print(f"  ✓ Stage classifier: CV accuracy={class_metrics['cv_accuracy']:.3f}")
        print(f"    Feature importance: {class_metrics['feature_importance']}")
    
    # Shelf life
    if "shelf_life" in traj_results:
        sl = traj_results["shelf_life"]
        print(f"  ✓ Shelf life estimate: {sl['shelf_life_estimate']:.1f} hours")
        print(f"    Confidence interval: {sl['confidence_interval']}")
        if sl['extrapolation_warning']:
            print("    ⚠️  Warning: Estimate extrapolates beyond observed time range")
    
    # Step 4: Calibration Transfer Setup
    print("\n[4/5] Setting up calibration transfer...")
    source_std, target_std = generate_transfer_standards(fs.data, n_standards=20)
    print(f"  ✓ Generated {len(source_std)} paired standards")
    
    # Apply transfer
    fs.apply_calibration_transfer(
        source_standards=source_std,
        target_standards=target_std,
        method="pds",
        pds_window_size=11,
        alpha=1.0
    )
    
    ct_metrics = fs.bundle.metrics.get("calibration_transfer_transfer", {})
    print(f"  ✓ Transfer reconstruction RMSE: {ct_metrics.get('reconstruction_rmse', 0):.4f}")
    print(f"  ✓ Standards used: {ct_metrics.get('n_standards', 0)}")
    
    # Step 5: Export Results
    print("\n[5/5] Exporting results...")
    output_dir = Path("./moats_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    fs.export(path=output_dir, formats=["json", "csv"])
    print(f"  ✓ Results saved to {output_dir}/")
    
    # Save artifact
    from foodspec import save_artifact
    artifact_path = output_dir / "moats_demo.foodspec"
    
    # Add a placeholder model to bundle artifacts
    fs.bundle.artifacts["model"] = {"type": "demo", "description": "No trained model in this demo"}
    
    save_artifact(
        bundle=fs.bundle,
        path=artifact_path,
        schema={
            "description": "Moats demo: matrix correction + heating trajectory + calibration transfer",
            "moats_applied": ["matrix_correction", "heating_trajectory", "calibration_transfer"],
            "dataset_summary": {
                "n_samples": len(fs.data),
                "n_wavenumbers": fs.data.x.shape[1],
                "modality": fs.data.modality,
            }
        }
    )
    print(f"  ✓ Artifact saved: {artifact_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Pipeline steps: {', '.join(fs._steps_applied)}")
    print(f"Total metrics recorded: {len(fs.bundle.metrics)}")
    print(f"\nKey findings:")
    print(f"  • Matrix correction reduced bias by {mc_metrics.get('total_correction_magnitude', 0):.3f} units")
    if "shelf_life" in traj_results:
        print(f"  • Estimated shelf life: {traj_results['shelf_life']['shelf_life_estimate']:.1f} hours")
    print(f"  • Calibration transfer RMSE: {ct_metrics.get('reconstruction_rmse', 0):.4f}")
    print("\n✅ All moats successfully applied!")
    print(f"\nView full results in {output_dir}/")


if __name__ == "__main__":
    main()
