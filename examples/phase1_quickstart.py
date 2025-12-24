"""
Phase 1 Quickstart: Unified FoodSpec API

This script demonstrates the Phase 1 unified entry point with chainable API.
Run this to see the complete workflow: load → QC → preprocess → train → export
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from foodspec import FoodSpec


def main():
    """Demonstrate Phase 1 unified FoodSpec API."""
    
    print("=" * 70)
    print("PHASE 1: UNIFIED FOODSPEC API - QUICKSTART")
    print("=" * 70)
    
    # 1. CREATE SYNTHETIC SPECTROSCOPY DATA
    print("\n1. Creating synthetic spectroscopy data...")
    n_samples = 30
    n_wavenumbers = 200
    
    # Simulate Raman spectra (3 oil types)
    np.random.seed(42)
    x = np.random.randn(n_samples, n_wavenumbers) * 0.5 + np.linspace(0, 10, n_wavenumbers)
    wn = np.linspace(500, 2000, n_wavenumbers)
    
    oil_types = ["olive", "sunflower", "canola"] * 10
    metadata = pd.DataFrame({
        "sample_id": [f"oil_{i:03d}" for i in range(n_samples)],
        "oil_type": oil_types,
        "batch": np.random.randint(1, 4, n_samples),
    })
    
    print(f"   - Shape: {n_samples} samples × {n_wavenumbers} wavenumbers")
    print(f"   - Modality: Raman (500-2000 cm⁻¹)")
    print(f"   - Classes: {metadata['oil_type'].unique()}")
    
    # 2. INITIALIZE FOODSPEC
    print("\n2. Initializing FoodSpec with data...")
    fs = FoodSpec(
        x,
        wavenumbers=wn,
        metadata=metadata,
        modality="raman",
        kind="oils",
    )
    print(f"   ✓ FoodSpec initialized: {len(fs.data)} samples, {fs.data.x.shape[1]} wavenumbers")
    
    # 3. DEMONSTRATE CHAINABLE API
    print("\n3. Executing chainable workflow...")
    
    # QC Step
    print("   a) Running QC (outlier detection)...")
    fs.qc(method="isolation_forest", threshold=0.1)
    print(f"      ✓ QC complete: {len(fs._steps_applied)} step(s) recorded")
    
    # Preprocessing Step
    print("   b) Preprocessing...")
    fs.preprocess(preset="standard")
    print(f"      ✓ Preprocessing logged: {len(fs._steps_applied)} step(s) recorded")
    
    # Retrieve workflow summary
    print("\n4. Workflow Summary:")
    summary = fs.summary()
    print("   " + "\n   ".join(summary.split("\n")))
    
    # 5. ADD METRICS AND DIAGNOSTICS
    print("\n5. Adding metrics and diagnostics...")
    
    # Simulate some analysis results
    fs.bundle.add_metrics("n_samples_analyzed", len(fs.data))
    fs.bundle.add_metrics("preprocessing_time", 2.34)
    fs.bundle.add_metrics("quality_score", 0.95)
    
    # Add a diagnostic (e.g., PCA variance explained)
    pca_variance = pd.DataFrame({
        "PC": [1, 2, 3, 4, 5],
        "variance_explained": [0.45, 0.25, 0.15, 0.10, 0.05],
    })
    fs.bundle.add_diagnostic("pca_variance", pca_variance)
    
    print(f"   ✓ Added {len(fs.bundle.metrics)} metrics")
    print(f"   ✓ Added {len(fs.bundle.diagnostics)} diagnostic")
    
    # 6. EXPORT RESULTS
    print("\n6. Exporting results...")
    with TemporaryDirectory() as tmpdir:
        out_dir = fs.export(tmpdir)
        print(f"   ✓ Exported to: {out_dir}")
        
        # List output files
        for item in sorted(out_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(out_dir)
                size = item.stat().st_size
                print(f"      - {rel_path} ({size} bytes)")
    
    # 7. ACCESS OUTPUTS PROGRAMMATICALLY
    print("\n7. Accessing outputs programmatically...")
    print(f"   Metrics:")
    for key, value in fs.bundle.metrics.items():
        print(f"      - {key}: {value}")
    
    print(f"\n   Diagnostics:")
    for key, value in fs.bundle.diagnostics.items():
        print(f"      - {key}: {type(value).__name__}")
    
    # 8. PROVENANCE
    print("\n8. Provenance Tracking:")
    record = fs.bundle.run_record
    print(f"   - Workflow: {record.workflow_name}")
    print(f"   - Config hash: {record.config_hash}")
    print(f"   - Dataset hash: {record.dataset_hash}")
    print(f"   - Steps recorded: {len(record.step_records)}")
    for i, step in enumerate(record.step_records, 1):
        print(f"      {i}. {step['name']} (hash: {step['hash']})")
    
    print("\n" + "=" * 70)
    print("✓ PHASE 1 WORKFLOW COMPLETE")
    print("=" * 70)
    print("\nKey Points:")
    print("  • Single entry point (FoodSpec) for entire workflow")
    print("  • Chainable API (.qc().preprocess().train().export())")
    print("  • Complete provenance tracking (config + dataset + steps)")
    print("  • Unified artifact export (metrics/diagnostics/provenance)")
    print("  • Type-safe, validated data structures")
    print("\nNext: Phase 2 will integrate real preprocessing/training pipelines")
    print("=" * 70)


if __name__ == "__main__":
    main()
