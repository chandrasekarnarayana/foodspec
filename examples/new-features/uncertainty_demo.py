"""
Demonstration script for uncertainty quantification visualizations.

Showcases:
- Confidence maps for prediction confidence visualization
- Set size distributions for conformal prediction analysis
- Coverage-efficiency trade-off curves
- Abstention distribution analysis
"""

import sys
from pathlib import Path

# Ensure src is in path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from foodspec.viz.uncertainty import (
    get_uncertainty_statistics,
    plot_abstention_distribution,
    plot_confidence_map,
    plot_coverage_efficiency,
    plot_set_size_distribution,
)


def generate_synthetic_confidence_data(n_samples=100, seed=42):
    """Generate synthetic confidence and prediction data."""
    np.random.seed(seed)
    confidences = np.random.beta(5, 2, n_samples)
    class_predictions = np.random.randint(0, 3, n_samples)
    return confidences, class_predictions


def generate_synthetic_conformal_data(n_samples=100, n_alpha=5, seed=42):
    """Generate synthetic conformal prediction data."""
    np.random.seed(seed)

    # Set sizes typically increase as alpha increases
    alphas = np.linspace(0.05, 0.3, n_alpha)
    set_sizes = []
    coverages = []
    avg_set_sizes = []

    for alpha in alphas:
        # Simulate conformal prediction
        sizes = np.random.poisson(1/alpha + 0.5, n_samples) + 1
        coverage = 1 - (alpha / 2)  # Typical relationship
        set_sizes.append(sizes)
        coverages.append(coverage)
        avg_set_sizes.append(np.mean(sizes))

    return alphas, coverages, avg_set_sizes, set_sizes


def demo_confidence_basic():
    """Basic confidence map visualization."""
    print("\n=== Demo 1: Basic Confidence Map ===")
    confidences, _ = generate_synthetic_confidence_data(n_samples=50)

    fig = plot_confidence_map(
        confidences,
        title="Model Prediction Confidence",
        colormap="RdYlGn",
        save_path=Path("outputs/uncertainty_demo") / "confidence_basic.png",
    )
    print("✓ Generated: confidence_basic.png")
    return fig


def demo_confidence_with_classes():
    """Confidence map with class predictions."""
    print("\n=== Demo 2: Confidence by Predicted Class ===")
    confidences, class_preds = generate_synthetic_confidence_data(n_samples=50)

    class_names = [f"Sample {i}" for i in range(len(confidences))]
    fig = plot_confidence_map(
        confidences,
        class_predictions=class_preds,
        sample_labels=class_names,
        title="Confidence Colored by Predicted Class",
        colormap="tab10",
        save_path=Path("outputs/uncertainty_demo") / "confidence_by_class.png",
    )
    print("✓ Generated: confidence_by_class.png")
    return fig


def demo_confidence_sorted():
    """Confidence map with custom sorting."""
    print("\n=== Demo 3: Confidence Map (Sorted) ===")
    confidences, class_preds = generate_synthetic_confidence_data(n_samples=80)

    fig = plot_confidence_map(
        confidences,
        class_predictions=class_preds,
        sort_by_confidence=True,
        confidence_thresholds=[0.5, 0.7, 0.85],
        show_values=True,
        title="Confidence Predictions (Sorted)",
        save_path=Path("outputs/uncertainty_demo") / "confidence_sorted.png",
    )
    print("✓ Generated: confidence_sorted.png")
    return fig


def demo_set_size_basic():
    """Basic set size distribution."""
    print("\n=== Demo 4: Basic Set Size Distribution ===")
    alphas, _, _, set_sizes_list = generate_synthetic_conformal_data(n_samples=200, n_alpha=5)
    set_sizes = np.concatenate(set_sizes_list)

    fig = plot_set_size_distribution(
        set_sizes,
        show_violin=True,
        show_box=True,
        title="Conformal Prediction Set Sizes",
        save_path=Path("outputs/uncertainty_demo") / "setsize_basic.png",
    )
    print("✓ Generated: setsize_basic.png")
    return fig


def demo_set_size_by_batch():
    """Set size distribution by batch."""
    print("\n=== Demo 5: Set Size by Batch ===")
    alphas, _, _, set_sizes_list = generate_synthetic_conformal_data(n_samples=100, n_alpha=5)
    set_sizes = np.concatenate(set_sizes_list)

    # Create batch labels
    batch_labels = np.repeat(np.arange(len(set_sizes_list)), 100)

    fig = plot_set_size_distribution(
        set_sizes,
        batch_labels=batch_labels,
        show_violin=True,
        show_box=True,
        title="Conformal Set Sizes by Significance Level",
        save_path=Path("outputs/uncertainty_demo") / "setsize_by_batch.png",
    )
    print("✓ Generated: setsize_by_batch.png")
    return fig


def demo_coverage_efficiency():
    """Coverage vs efficiency trade-off."""
    print("\n=== Demo 6: Coverage vs Efficiency ===")
    alphas, coverages, avg_sizes, _ = generate_synthetic_conformal_data(n_alpha=10)

    fig = plot_coverage_efficiency(
        alphas,
        np.array(coverages),
        np.array(avg_sizes),
        target_coverage=0.9,
        colormap="viridis",
        title="Coverage-Efficiency Trade-off Curve",
        save_path=Path("outputs/uncertainty_demo") / "coverage_efficiency.png",
    )
    print("✓ Generated: coverage_efficiency.png")
    return fig


def demo_abstention_overall():
    """Overall abstention distribution."""
    print("\n=== Demo 7: Overall Abstention Distribution ===")
    np.random.seed(42)
    abstain_flags = np.random.binomial(1, 0.25, 500)

    fig = plot_abstention_distribution(
        abstain_flags,
        title="Prediction Abstention Rate",
        save_path=Path("outputs/uncertainty_demo") / "abstention_overall.png",
    )
    print("✓ Generated: abstention_overall.png")
    return fig


def demo_abstention_by_class():
    """Abstention distribution by class."""
    print("\n=== Demo 8: Abstention by Predicted Class ===")
    np.random.seed(42)
    n_samples = 300

    # Create class-specific abstention rates
    class_labels = np.random.randint(0, 3, n_samples)
    abstain_flags = np.zeros(n_samples, dtype=int)

    # Class 0: 10% abstain
    abstain_flags[class_labels == 0] = np.random.binomial(1, 0.1, np.sum(class_labels == 0))
    # Class 1: 25% abstain
    abstain_flags[class_labels == 1] = np.random.binomial(1, 0.25, np.sum(class_labels == 1))
    # Class 2: 40% abstain
    abstain_flags[class_labels == 2] = np.random.binomial(1, 0.40, np.sum(class_labels == 2))

    fig = plot_abstention_distribution(
        abstain_flags,
        class_labels=class_labels,
        title="Abstention Rates by Predicted Class",
        save_path=Path("outputs/uncertainty_demo") / "abstention_by_class.png",
    )
    print("✓ Generated: abstention_by_class.png")
    return fig


def demo_integrated_workflow():
    """Complete integrated uncertainty workflow."""
    print("\n=== Demo 9: Integrated Uncertainty Workflow ===")

    # Generate comprehensive dataset
    np.random.seed(123)
    n_samples = 300
    n_alpha = 8

    # Confidence data
    confidences = np.random.beta(6, 2, n_samples)
    class_preds = np.random.randint(0, 4, n_samples)

    # Conformal prediction data
    alphas = np.linspace(0.02, 0.25, n_alpha)
    all_set_sizes = []
    coverages = []
    avg_sizes = []

    for alpha in alphas:
        sizes = np.random.poisson(2/alpha, n_samples) + 1
        all_set_sizes.append(sizes)
        coverage = 1 - alpha * 0.8  # Typical conformal behavior
        coverages.append(coverage)
        avg_sizes.append(np.mean(sizes))

    set_sizes = np.concatenate(all_set_sizes)

    # Abstention data
    abstain_flags = np.random.binomial(1, 0.2, n_samples)

    # Extract statistics
    stats = get_uncertainty_statistics(
        confidences,
        set_sizes=set_sizes,
        abstain_flags=abstain_flags
    )

    print(f"Confidence mean: {stats['confidence']['mean']:.4f}")
    print(f"Confidence std: {stats['confidence']['std']:.4f}")
    print(f"Set size mean: {stats['set_size']['mean']:.2f}")
    print(f"Abstention rate: {stats['abstention']['rate']:.1%}")

    # Generate visualizations
    fig1 = plot_confidence_map(
        confidences,
        class_predictions=class_preds,
        sort_by_confidence=True,
        title="Integrated: Model Confidence",
        save_path=Path("outputs/uncertainty_demo") / "integrated_confidence.png",
    )

    # Batch labels for set sizes
    batch_labels = np.repeat(np.arange(n_alpha), n_samples)
    fig2 = plot_set_size_distribution(
        set_sizes,
        batch_labels=batch_labels,
        title="Integrated: Conformal Set Sizes",
        save_path=Path("outputs/uncertainty_demo") / "integrated_setsize.png",
    )

    fig3 = plot_coverage_efficiency(
        alphas,
        np.array(coverages),
        np.array(avg_sizes),
        target_coverage=0.95,
        title="Integrated: Coverage-Efficiency Trade-off",
        save_path=Path("outputs/uncertainty_demo") / "integrated_coverage.png",
    )

    fig4 = plot_abstention_distribution(
        abstain_flags,
        class_labels=class_preds,
        title="Integrated: Abstention by Class",
        save_path=Path("outputs/uncertainty_demo") / "integrated_abstention.png",
    )

    print("✓ Generated: integrated_confidence.png")
    print("✓ Generated: integrated_setsize.png")
    print("✓ Generated: integrated_coverage.png")
    print("✓ Generated: integrated_abstention.png")

    return fig1, fig2, fig3, fig4


def main():
    """Run all demonstrations."""
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("FoodSpec: Uncertainty Quantification Visualizations")
    print("=" * 70)

    # Create output directory
    output_dir = Path("outputs/uncertainty_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run demos
        demo_confidence_basic()
        demo_confidence_with_classes()
        demo_confidence_sorted()
        demo_set_size_basic()
        demo_set_size_by_batch()
        demo_coverage_efficiency()
        demo_abstention_overall()
        demo_abstention_by_class()
        demo_integrated_workflow()

        print("\n" + "=" * 70)
        print("✓ All demonstrations completed successfully!")
        print(f"✓ Output directory: {output_dir.absolute()}")
        print("✓ Generated 12 visualizations")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        raise
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
