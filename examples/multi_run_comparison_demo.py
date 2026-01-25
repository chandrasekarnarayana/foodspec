"""Demo of multi-run comparison utilities.

This script demonstrates:
1. Scanning for analysis runs
2. Loading run summaries
3. Creating comparison visualizations
4. Baseline tracking
5. Monitoring metrics over time
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from foodspec.viz.compare import (
    compare_runs,
    load_run_summary,
    scan_runs,
)


def create_demo_runs(root_dir: Path) -> None:
    """Create demo run directories for testing."""
    print("Creating demo run directories...")
    root_dir.mkdir(parents=True, exist_ok=True)

    models = ["RandomForest", "SVM", "XGBoost", "LogisticRegression"]

    for i, model in enumerate(models):
        run_dir = root_dir / f"run_{i+1:03d}"
        run_dir.mkdir(exist_ok=True)

        # Create manifest
        manifest = {
            "run_id": f"run_{i+1:03d}",
            "timestamp": f"2026-01-{15+i:02d}T{10+i:02d}:00:00Z",
            "algorithm": model,
            "validation_scheme": "holdout" if i % 2 == 0 else "5-fold-cv",
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Create metrics (with realistic progression)
        base_f1 = 0.75 + i * 0.05
        metrics = {
            "macro_f1": base_f1,
            "accuracy": base_f1 + 0.05,
            "auroc": 0.85 + i * 0.03,
            "coverage": 0.95 - i * 0.02,
            "precision": base_f1 + 0.02,
            "recall": base_f1 - 0.01,
        }
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Create trust metrics
        trust_metrics = {
            "ece": 0.12 - i * 0.02,
            "abstain_rate": 0.05 + i * 0.01,
            "brier_score": 0.10 - i * 0.01,
            "nll": 0.30 - i * 0.03,
        }
        with open(run_dir / "uncertainty_metrics.json", "w") as f:
            json.dump(trust_metrics, f, indent=2)

        # Create QC results
        qc_results = {
            "passed_snr": True,
            "passed_baseline": i >= 1,
            "passed_drift": i >= 2,
        }
        with open(run_dir / "qc_results.json", "w") as f:
            json.dump(qc_results, f, indent=2)

    print(f"✓ Created {len(models)} demo runs in {root_dir}")


def main() -> None:
    """Run comparison demo."""
    print("=" * 60)
    print("🥗 FoodSpec Multi-Run Comparison Demo")
    print("=" * 60)
    print()

    # Set up directories
    demo_runs_dir = Path("demo_runs")
    output_dir = Path("comparison_output")

    # Create demo runs
    create_demo_runs(demo_runs_dir)
    print()

    # Step 1: Scan for runs
    print("Step 1: Scanning for runs...")
    print("-" * 60)
    run_dirs = scan_runs(demo_runs_dir)
    print(f"Found {len(run_dirs)} runs:")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")
    print()

    # Step 2: Load run summaries
    print("Step 2: Loading run summaries...")
    print("-" * 60)
    summaries = []
    for run_dir in run_dirs:
        summary = load_run_summary(run_dir)
        summaries.append(summary)
        print(f"Loaded {summary.run_id}: {summary.model_name}")
        print(f"  F1 Score: {summary.get_metric('macro_f1'):.4f}")
        print(f"  AUROC: {summary.get_metric('auroc'):.4f}")
        print(f"  ECE: {summary.get_trust_metric('ece'):.4f}")
    print()

    # Step 3: Compare runs
    print("Step 3: Generating comparison outputs...")
    print("-" * 60)
    outputs = compare_runs(
        summaries,
        output_dir=output_dir,
        baseline_id="run_001",  # Use first run as baseline
        top_n=4,  # Include all runs in radar
    )

    print("Generated outputs:")
    for output_type, output_path in outputs.items():
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"  ✓ {output_type}: {output_path.name} ({file_size:.1f} KB)")
    print()

    # Step 4: Display leaderboard
    print("Step 4: Leaderboard Preview")
    print("-" * 60)
    leaderboard_path = outputs["leaderboard"]
    import pandas as pd

    leaderboard = pd.read_csv(leaderboard_path)
    print(leaderboard[["rank", "run_id", "model", "macro_f1", "auroc", "coverage"]])
    print()

    # Step 5: Display baseline deltas
    if "baseline_deltas" in outputs:
        print("Step 5: Baseline Deltas")
        print("-" * 60)
        deltas = pd.read_csv(outputs["baseline_deltas"])
        print(deltas[["run_id", "model", "delta_macro_f1", "delta_auroc"]])
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Scanned {len(run_dirs)} runs")
    print(f"✓ Generated {len(outputs)} output files")
    print(f"✓ Dashboard available: {outputs['dashboard']}")
    print(f"✓ Radar plot: {outputs['radar']}")
    print(f"✓ Monitoring plot: {outputs['monitoring']}")
    print()
    print("Next steps:")
    print("  1. Open comparison_dashboard.html in your browser")
    print("  2. Review radar.png for visual comparison")
    print("  3. Check monitoring.png for trends over time")
    print("  4. Use comparison.csv for further analysis")
    print()


if __name__ == "__main__":
    main()
