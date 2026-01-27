"""Demo of reproducibility pack and archive export functionality.

This script demonstrates creating reproducibility packs and exporting
shareable archives from analysis runs.
"""

import json
import sys
from pathlib import Path

# Add src to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from foodspec.reporting.export import (
    build_reproducibility_pack,
    export_archive,
    get_archive_file_list,
    verify_archive_integrity,
)


def create_demo_run(run_dir: Path) -> None:
    """Create a demo analysis run with sample artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "run_id": "demo_run_001",
        "timestamp": "2024-01-15T10:30:00",
        "algorithm": "PLS-DA",
        "parameters": {
            "n_components": 5,
            "cv_folds": 5,
            "preprocessing": "SNV + baseline_removal",
        },
        "data_source": "oil_authentication_dataset",
        "samples_processed": 250,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Create protocol snapshot
    protocol = {
        "name": "Oil Authentication Protocol v2.1",
        "version": "2.1",
        "description": "Authentication of extra virgin olive oils",
        "steps": [
            {
                "name": "Data Loading",
                "type": "input",
                "description": "Load spectral data from CSV",
                "parameters": {"format": "csv", "wavelength_range": [900, 1700]},
            },
            {
                "name": "SNV Normalization",
                "type": "preprocessing",
                "description": "Apply Standard Normal Variate normalization",
                "parameters": {"method": "snv"},
            },
            {
                "name": "Baseline Removal",
                "type": "preprocessing",
                "description": "Remove baseline using polynomial fitting",
                "parameters": {"method": "poly", "order": 3},
            },
            {
                "name": "PLS-DA Model",
                "type": "analysis",
                "description": "Partial Least Squares Discriminant Analysis",
                "parameters": {"n_components": 5, "cv_folds": 5},
            },
            {
                "name": "Validation",
                "type": "validation",
                "description": "Cross-validation and performance assessment",
                "parameters": {"metrics": ["accuracy", "precision", "recall"]},
            },
        ],
    }
    with open(run_dir / "protocol_snapshot.json", "w") as f:
        json.dump(protocol, f, indent=2)

    # Create metrics
    metrics = {
        "accuracy": 0.954,
        "precision": 0.951,
        "recall": 0.948,
        "f1_score": 0.950,
        "roc_auc": 0.989,
        "rmse": 0.047,
        "r2_score": 0.923,
        "matthews_cc": 0.907,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Create predictions
    predictions = {
        "sample_001": {"predicted": "EVOO", "confidence": 0.98},
        "sample_002": {"predicted": "EVOO", "confidence": 0.95},
        "sample_003": {"predicted": "non-EVOO", "confidence": 0.92},
        "sample_004": {"predicted": "non-EVOO", "confidence": 0.87},
        "sample_005": {"predicted": "EVOO", "confidence": 0.93},
    }
    with open(run_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    # Create QC results
    qc_results = {
        "data_completeness": 1.0,
        "outliers_detected": 3,
        "preprocessing_success": True,
        "model_convergence": True,
        "cross_validation_stability": 0.94,
        "feature_relevance_mean": 0.87,
    }
    with open(run_dir / "qc_results.json", "w") as f:
        json.dump(qc_results, f, indent=2)

    # Create plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Create sample plot files
    plot_contents = {
        "pls_scores_plot.png": b"PNG fake data for PLS scores",
        "loadings_plot.pdf": b"PDF fake data for loadings",
        "confusion_matrix.png": b"PNG fake data for confusion matrix",
        "roc_curve.png": b"PNG fake data for ROC curve",
        "variable_importance.png": b"PNG fake data for VIP plot",
    }

    for filename, content in plot_contents.items():
        (plots_dir / filename).write_bytes(content)

    # Create dossier directory
    dossier_dir = run_dir / "dossier"
    dossier_dir.mkdir(exist_ok=True)

    methods_md = """# Methods

## Data Acquisition
Extra virgin olive oil samples (n=250) were analyzed using visible-near infrared spectroscopy.

## Preprocessing
Spectra were normalized using Standard Normal Variate (SNV) and baseline-corrected
using polynomial (order 3) fitting.

## Multivariate Analysis
Partial Least Squares Discriminant Analysis (PLS-DA) was applied with:
- 5 latent components
- 5-fold cross-validation
- 70% training / 30% test split
"""
    (dossier_dir / "methods.md").write_text(methods_md)

    results_md = """# Results

## Model Performance
The PLS-DA model achieved excellent discrimination between authentic and counterfeit oils:

| Metric | Value |
|--------|-------|
| Accuracy | 95.4% |
| Precision | 95.1% |
| Recall | 94.8% |
| F1 Score | 95.0% |
| ROC-AUC | 0.989 |

## Cross-Validation
The model showed stable performance across all folds with CV accuracy of 94.2%.

## Key Findings
- 5 spectral regions were identified as discriminant
- Variable importance plots show strong contributions from 12 key wavelengths
- No significant overfitting detected
"""
    (dossier_dir / "results.md").write_text(results_md)

    # Create bundle directory
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    data_file = bundle_dir / "processed_data.json"
    data = {
        "n_samples": 250,
        "n_variables": 500,
        "preprocessing_applied": ["SNV", "baseline_removal"],
        "pca_variance_explained": 0.92,
    }
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    """Run demo."""
    # Setup
    demo_dir = Path("demo_export")
    run_dir = demo_dir / "sample_run"
    pack_dir = demo_dir / "reproducibility_pack"
    output_dir = demo_dir / "exports"

    print("=" * 70)
    print("FoodSpec Reproducibility Export Demo")
    print("=" * 70)

    # Step 1: Create demo run
    print("\n1. Creating demo analysis run...")
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    create_demo_run(run_dir)
    print(f"   ✓ Demo run created at {run_dir}")
    print(f"   ✓ Files: {len(list(run_dir.rglob('*')))} artifacts")

    # Step 2: Build reproducibility pack
    print("\n2. Building reproducibility pack...")
    pack_path = build_reproducibility_pack(run_dir, pack_dir)
    pack_files = list(pack_dir.glob("*"))
    print(f"   ✓ Pack created at {pack_path}")
    print(f"   ✓ Pack contents ({len(pack_files)} items):")
    for item in sorted(pack_files):
        if item.is_file():
            size = item.stat().st_size
            print(f"     - {item.name} ({size} bytes)")
        else:
            sub_files = list(item.glob("*"))
            print(f"     - {item.name}/ ({len(sub_files)} files)")

    # Step 3: Export full archive
    print("\n3. Exporting full archive...")
    output_dir.mkdir(exist_ok=True)
    archive_full = export_archive(
        output_dir / "oil_analysis_full.zip",
        run_dir,
        include=["dossier", "figures", "tables", "bundle"],
    )
    print(f"   ✓ Full archive: {archive_full.name} ({archive_full.stat().st_size} bytes)")

    # Step 4: Export selective archives
    print("\n4. Exporting selective archives...")

    archive_minimal = export_archive(
        output_dir / "oil_analysis_minimal.zip",
        run_dir,
        include=["dossier", "figures"],
    )
    print(
        f"   ✓ Minimal (dossier + figures): {archive_minimal.name} "
        f"({archive_minimal.stat().st_size} bytes)"
    )

    archive_data = export_archive(
        output_dir / "oil_analysis_data.zip",
        run_dir,
        include=["tables", "bundle"],
    )
    print(
        f"   ✓ Data (tables + bundle): {archive_data.name} "
        f"({archive_data.stat().st_size} bytes)"
    )

    # Step 5: Verify archives
    print("\n5. Verifying archives...")
    for archive_path in [archive_full, archive_minimal, archive_data]:
        is_valid = verify_archive_integrity(archive_path)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"   {status}: {archive_path.name}")

    # Step 6: Show archive contents
    print("\n6. Archive contents (deterministic order):")
    full_contents = get_archive_file_list(archive_full)
    print(f"   Full archive has {len(full_contents)} files:")
    for filename in full_contents[:10]:
        print(f"     - {filename}")
    if len(full_contents) > 10:
        print(f"     ... and {len(full_contents) - 10} more files")

    minimal_contents = get_archive_file_list(archive_minimal)
    print(f"\n   Minimal archive has {len(minimal_contents)} files:")
    for filename in minimal_contents:
        print(f"     - {filename}")

    # Step 7: Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nOriginal run: {run_dir} ({len(list(run_dir.rglob('*')))} files)")
    print(f"Reproducibility pack: {pack_dir} ({len(pack_files)} items)")
    print("\nArchives created:")
    print(f"  1. Full archive: {archive_full.name}")
    print(f"     Size: {archive_full.stat().st_size / 1024:.1f} KB")
    print(f"  2. Minimal archive: {archive_minimal.name}")
    print(f"     Size: {archive_minimal.stat().st_size / 1024:.1f} KB")
    print(f"  3. Data archive: {archive_data.name}")
    print(f"     Size: {archive_data.stat().st_size / 1024:.1f} KB")

    print("\n✓ All archives are shareable and reproducible!")
    print(f"✓ Located in: {output_dir}")


if __name__ == "__main__":
    main()
