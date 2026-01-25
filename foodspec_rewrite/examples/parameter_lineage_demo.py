"""Demo of parameter map and data lineage visualizations."""

from pathlib import Path
from unittest.mock import MagicMock

from foodspec.viz.parameters import plot_parameter_map, get_parameter_summary
from foodspec.viz.lineage import plot_data_lineage, get_lineage_summary


def create_demo_protocol():
    """Create a demo protocol with mixed default/non-default parameters."""
    protocol = MagicMock()
    protocol.data = MagicMock()
    protocol.data.input = "/data/raman_samples.csv"
    protocol.data.format = "csv"
    
    protocol.preprocess = MagicMock()
    protocol.preprocess.recipe = "raman_baseline_correction"
    protocol.preprocess.steps = ["baseline", "normalize", "smooth"]
    
    protocol.qc = MagicMock()
    protocol.qc.thresholds = {"snr": 50, "fwhm": 10, "intensity": 100}
    protocol.qc.metrics = ["snr", "fwhm", "rms", "intensity"]
    
    protocol.features = MagicMock()
    protocol.features.modules = ["pca", "plsr"]
    protocol.features.strategy = "hybrid"
    
    protocol.model = MagicMock()
    protocol.model.estimator = "random_forest"
    protocol.model.hyperparameters = {"n_estimators": 200, "max_depth": 15}
    
    protocol.uncertainty = MagicMock()
    protocol.uncertainty.conformal = {
        "method": "mondrian",
        "alpha": 0.1,
        "calibration": {"method": "platt"}
    }
    
    protocol.interpretability = MagicMock()
    protocol.interpretability.methods = ["shap", "lime", "permutation"]
    protocol.interpretability.marker_panel = ["principal_markers", "secondary_markers"]
    
    protocol.reporting = MagicMock()
    protocol.reporting.format = "html"
    protocol.reporting.sections = ["executive_summary", "results", "diagnostics", "appendix"]
    
    protocol.export = MagicMock()
    protocol.export.bundle = True
    
    return protocol


def create_demo_manifest():
    """Create a demo manifest with data lineage."""
    manifest = MagicMock()
    
    manifest.inputs = [
        {
            "path": "/raw_data/raman_samples_batch1.csv",
            "hash": "sha256:3f8a2b5c7d9e1f4a6b8c0d2e4f6a8b0c2d4e6f8a0b2c4d6e8f0a2c4e6f8a0b",
            "timestamp": "2026-01-25T08:00:00"
        },
        {
            "path": "/raw_data/raman_samples_batch2.csv",
            "hash": "sha256:9f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f",
            "timestamp": "2026-01-25T08:15:00"
        },
        {
            "path": "/metadata/sample_metadata.json",
            "hash": "sha256:1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1",
            "timestamp": "2026-01-25T07:00:00"
        }
    ]
    
    manifest.preprocessing = [
        {
            "name": "baseline_correction",
            "method": "polynomial",
            "parameters": {"order": 5},
            "duration": 12.3
        },
        {
            "name": "normalization",
            "method": "vector",
            "parameters": {"norm": "l2"},
            "duration": 3.2
        },
        {
            "name": "smoothing",
            "method": "savitzky_golay",
            "parameters": {"window_length": 11, "polyorder": 3},
            "duration": 2.8
        }
    ]
    
    manifest.processing = [
        {
            "name": "pca",
            "n_components": 10,
            "variance_explained": 0.85,
            "duration": 5.1
        },
        {
            "name": "plsr",
            "n_components": 5,
            "cross_validation_folds": 5,
            "duration": 8.4
        },
        {
            "name": "random_forest_training",
            "n_estimators": 200,
            "max_depth": 15,
            "cross_validation_score": 0.92,
            "duration": 15.7
        }
    ]
    
    manifest.outputs = [
        {
            "path": "/results/predictions.csv",
            "hash": "sha256:7a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a",
            "timestamp": "2026-01-25T08:50:00"
        },
        {
            "path": "/results/confidence_scores.json",
            "hash": "sha256:2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c",
            "timestamp": "2026-01-25T08:51:00"
        },
        {
            "path": "/results/model_report.html",
            "hash": "sha256:5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5",
            "timestamp": "2026-01-25T08:52:00"
        },
        {
            "path": "/results/diagnostics.json",
            "hash": "sha256:8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
            "timestamp": "2026-01-25T08:53:00"
        }
    ]
    
    return manifest


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("outputs/viz_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FoodSpec Visualization Demo: Parameters & Data Lineage")
    print("=" * 70)
    
    # ========================================================================
    # Part A: Parameter Map
    # ========================================================================
    print("\n📋 PARAMETER MAP VISUALIZATION")
    print("-" * 70)
    
    protocol = create_demo_protocol()
    param_summary = get_parameter_summary(protocol)
    
    print(f"\n✓ Protocol loaded with {param_summary['total_parameters']} parameters")
    print(f"  - Non-default parameters: {param_summary['non_default_parameters']}")
    print(f"  - Percentage customized: {param_summary['non_default_percentage']}%")
    
    print("\nNon-default parameters detected:")
    for param, value in param_summary['non_defaults'].items():
        print(f"  • {param}")
    
    print("\nGenerating parameter map visualization...")
    param_fig = plot_parameter_map(
        protocol,
        save_path=output_dir,
        figure_size=(14, 10),
        dpi=300
    )
    
    param_files = [
        output_dir / "parameter_map.png",
        output_dir / "parameter_map.json"
    ]
    
    for fpath in param_files:
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✓ {fpath.name} ({size_kb:.1f} KB)")
    
    # ========================================================================
    # Part B: Data Lineage
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 DATA LINEAGE VISUALIZATION")
    print("-" * 70)
    
    manifest = create_demo_manifest()
    lineage_summary = get_lineage_summary(manifest)
    
    print(f"\n✓ Data lineage loaded")
    print(f"  - Input files: {lineage_summary['input_count']}")
    print(f"  - Preprocessing steps: {lineage_summary['preprocessing_steps']}")
    print(f"  - Processing steps: {lineage_summary['processing_steps']}")
    print(f"  - Output files: {lineage_summary['output_count']}")
    print(f"  - Total items: {lineage_summary['total_items']}")
    
    print("\nInput files:")
    for item in lineage_summary['lineage']['inputs']:
        path = item['path'].split('/')[-1]
        hash_val = item.get('hash', '')[:16] if item.get('hash') else '—'
        print(f"  • {path} [{hash_val}...]")
    
    print("\nProcessing pipeline:")
    for step in lineage_summary['lineage']['processing']:
        print(f"  • {step.get('name', 'unknown')}")
    
    print("\nOutput artifacts:")
    for item in lineage_summary['lineage']['outputs']:
        path = item['path'].split('/')[-1]
        hash_val = item.get('hash', '')[:16] if item.get('hash') else '—'
        print(f"  • {path} [{hash_val}...]")
    
    print("\nGenerating data lineage visualization...")
    lineage_fig = plot_data_lineage(
        manifest,
        save_path=output_dir,
        figure_size=(16, 10),
        dpi=300
    )
    
    lineage_files = [
        output_dir / "data_lineage.png",
        output_dir / "data_lineage.json"
    ]
    
    for fpath in lineage_files:
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✓ {fpath.name} ({size_kb:.1f} KB)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print(f"\nGenerated files in {output_dir}:")
    for fpath in sorted(output_dir.glob("*")):
        size_kb = fpath.stat().st_size / 1024
        print(f"  • {fpath.name} ({size_kb:.1f} KB)")
    
    print("\nVisualization Features:")
    print("  ✓ Parameter Map:")
    print("    - Hierarchical parameter display")
    print("    - Highlights non-default parameters in gold")
    print("    - PNG export for presentations")
    print("    - JSON snapshot for programmatic access")
    print("\n  ✓ Data Lineage:")
    print("    - Shows input → preprocessing → processing → output flow")
    print("    - Includes file hashes and timestamps")
    print("    - Horizontal flow chart layout")
    print("    - Color-coded stages")
    
    print("\n" + "=" * 70)
