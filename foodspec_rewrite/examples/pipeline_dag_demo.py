"""Quick demo of the pipeline DAG visualizer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from foodspec.viz.pipeline import plot_pipeline_dag, get_pipeline_stats


def create_demo_protocol():
    """Create a demo protocol with some stages enabled."""
    protocol = MagicMock()
    protocol.data = MagicMock()
    protocol.data.input = "/data/samples.csv"
    protocol.data.format = "csv"
    
    protocol.preprocess = MagicMock()
    protocol.preprocess.recipe = "raman_baseline"
    protocol.preprocess.steps = ["baseline_correction", "normalization"]
    
    protocol.qc = MagicMock()
    protocol.qc.thresholds = {"snr": 50, "fwhm": 10}
    protocol.qc.metrics = ["snr", "fwhm", "rms"]
    
    protocol.features = MagicMock()
    protocol.features.modules = ["pca", "plsr"]
    protocol.features.strategy = "hybrid"
    
    protocol.model = MagicMock()
    protocol.model.estimator = "ensemble"
    protocol.model.hyperparameters = {"n_estimators": 100}
    
    protocol.uncertainty = MagicMock()
    protocol.uncertainty.conformal = {"method": "mondrian", "alpha": 0.1}
    
    protocol.interpretability = MagicMock()
    protocol.interpretability.methods = ["shap", "lime"]
    protocol.interpretability.marker_panel = ["principal_markers"]
    
    protocol.reporting = MagicMock()
    protocol.reporting.format = "html"
    protocol.reporting.sections = ["summary", "results", "diagnostics"]
    
    protocol.export = MagicMock()
    protocol.export.bundle = {"format": "zip", "include_data": False}
    
    return protocol


if __name__ == "__main__":
    # Create a demo protocol
    protocol = create_demo_protocol()
    
    # Create output directory
    output_dir = Path("outputs/pipeline_dag_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 FoodSpec Pipeline DAG Visualizer Demo")
    print("=" * 50)
    
    # Get statistics
    stats = get_pipeline_stats(protocol)
    print(f"\nPipeline Statistics:")
    print(f"  Total stages: {stats['total_stages']}")
    print(f"  Enabled stages: {stats['enabled_stages']}")
    print(f"  Disabled stages: {stats['disabled_stages']}")
    
    print(f"\nEnabled stages:")
    for stage_name, details in stats['stage_details'].items():
        if details['enabled']:
            print(f"  ✓ {stage_name}")
            if details['params']:
                for key, value in details['params'].items():
                    print(f"      - {key}: {value}")
    
    # Generate visualization
    print(f"\nGenerating pipeline DAG visualization...")
    fig = plot_pipeline_dag(
        protocol,
        save_path=output_dir,
        seed=42,
        figure_size=(16, 10),
        dpi=300
    )
    
    svg_path = output_dir / "pipeline_dag.svg"
    png_path = output_dir / "pipeline_dag.png"
    
    print(f"✓ SVG saved to: {svg_path}")
    print(f"✓ PNG saved to: {png_path}")
    
    print("\n" + "=" * 50)
    print("Demo complete! Check the output directory for visualizations.")
