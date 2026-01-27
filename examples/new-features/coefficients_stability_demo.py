"""
Demonstration script for coefficient heatmaps and feature stability visualizations.

Showcases:
- Coefficient heatmap with various normalization and sorting methods
- Stability heatmap with bar summary and clustering
- Integration with feature selection workflows
"""

import sys
from pathlib import Path

# Ensure src is in path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from foodspec.viz.coefficients import (
    get_coefficient_statistics,
    plot_coefficients_heatmap,
)
from foodspec.viz.stability import (
    get_stability_statistics,
    plot_feature_stability,
)


def generate_synthetic_coefficients(n_features=20, n_classes=4, seed=42):
    """Generate synthetic model coefficients."""
    np.random.seed(seed)
    # Create coefficients with varying magnitudes
    coefs = np.random.randn(n_features, n_classes)
    # Add structure: some features more important than others
    coefs[:5] *= 5  # Important features
    coefs[5:10] *= 0.1  # Weak features
    return coefs


def generate_synthetic_stability(n_features=20, n_folds=5, seed=42):
    """Generate synthetic feature stability matrix."""
    np.random.seed(seed)
    # Create stability matrix (feature selection across folds)
    stability = np.random.binomial(1, 0.6, (n_features, n_folds))
    # Add structure: some features consistently selected
    stability[:5, :] = 1  # Always selected
    stability[-5:, :] = 0  # Never selected
    return stability


def demo_coefficients_basic():
    """Basic coefficient heatmap."""
    print("\n=== Demo 1: Basic Coefficient Heatmap ===")
    coefs = generate_synthetic_coefficients()

    fig = plot_coefficients_heatmap(
        coefs,
        title="Model Coefficients - Basic",
        colormap="RdBu_r",
        show_values=False,
        save_path=Path("outputs/coefficients_demo") / "basic.png",
    )
    print("✓ Generated: basic.png")
    return fig


def demo_coefficients_normalized():
    """Normalized coefficients with sorting."""
    print("\n=== Demo 2: Normalized & Sorted Coefficients ===")
    coefs = generate_synthetic_coefficients()
    class_names = ["Control", "Treatment A", "Treatment B", "Treatment C"]

    fig = plot_coefficients_heatmap(
        coefs,
        class_names=class_names,
        normalize="standard",
        sort_features="mean",
        title="Standardized Coefficients (Sorted by Magnitude)",
        colormap="RdBu_r",
        show_values=True,
        value_decimals=2,
        save_path=Path("outputs/coefficients_demo") / "normalized_sorted.png",
    )
    print("✓ Generated: normalized_sorted.png")
    return fig


def demo_coefficients_statistics():
    """Extract and display coefficient statistics."""
    print("\n=== Demo 3: Coefficient Statistics ===")
    coefs = generate_synthetic_coefficients()
    class_names = ["Control", "Treatment A", "Treatment B", "Treatment C"]
    feature_names = [f"Feature {i:02d}" for i in range(len(coefs))]

    stats = get_coefficient_statistics(coefs, class_names, feature_names)

    print(f"Global Mean Coefficient: {stats['global']['mean']:.4f}")
    print(f"Global Std Deviation: {stats['global']['std']:.4f}")
    print("\nTop 3 Most Important Features:")
    for ranking in stats["rankings"]["by_mean_magnitude"][:3]:
        print(
            f"  {ranking['rank']:2d}. {ranking['feature']:20s} "
            f"(magnitude: {ranking['magnitude']:.4f})"
        )

    # Plot with statistics
    fig = plot_coefficients_heatmap(
        coefs,
        class_names=class_names,
        feature_names=feature_names,
        normalize="minmax",
        sort_features="norm",
        title="Coefficients with Normalized Min-Max Scaling",
        colormap="coolwarm",
        save_path=Path("outputs/coefficients_demo") / "statistics.png",
    )
    print("✓ Generated: statistics.png")
    return fig


def demo_stability_basic():
    """Basic stability heatmap."""
    print("\n=== Demo 4: Basic Feature Stability ===")
    stability = generate_synthetic_stability()

    fig = plot_feature_stability(
        stability,
        show_bar_summary=True,
        bar_position="right",
        title="Feature Selection Stability Across Folds",
        colormap="RdYlGn",
        save_path=Path("outputs/coefficients_demo") / "stability_basic.png",
    )
    print("✓ Generated: stability_basic.png")
    return fig


def demo_stability_normalized():
    """Normalized stability with sorting."""
    print("\n=== Demo 5: Normalized & Sorted Stability ===")
    stability = generate_synthetic_stability()
    feature_names = [f"Feature {i:02d}" for i in range(len(stability))]
    fold_names = [f"Fold {i}" for i in range(stability.shape[1])]

    fig = plot_feature_stability(
        stability,
        feature_names=feature_names,
        fold_names=fold_names,
        normalize="frequency",
        sort_features="frequency",
        show_bar_summary=True,
        bar_position="right",
        show_values=True,
        title="Feature Stability (Normalized by Fold Count)",
        colormap="YlOrRd",
        save_path=Path("outputs/coefficients_demo") / "stability_normalized.png",
    )
    print("✓ Generated: stability_normalized.png")
    return fig


def demo_stability_bottom_bar():
    """Stability with bar at bottom."""
    print("\n=== Demo 6: Stability with Bottom Bar ===")
    stability = generate_synthetic_stability()

    fig = plot_feature_stability(
        stability,
        normalize="frequency",
        sort_features="frequency",
        show_bar_summary=True,
        bar_position="bottom",
        title="Feature Stability with Bottom Summary",
        colormap="viridis",
        figure_size=(14, 8),
        save_path=Path("outputs/coefficients_demo") / "stability_bottom_bar.png",
    )
    print("✓ Generated: stability_bottom_bar.png")
    return fig


def demo_stability_clustering():
    """Stability with hierarchical clustering."""
    print("\n=== Demo 7: Stability with Hierarchical Clustering ===")
    stability = generate_synthetic_stability()
    feature_names = [f"Feature {i:02d}" for i in range(len(stability))]

    fig = plot_feature_stability(
        stability,
        feature_names=feature_names,
        normalize="frequency",
        cluster_features=True,
        show_bar_summary=False,
        title="Feature Selection Patterns (Hierarchically Clustered)",
        colormap="RdYlGn",
        save_path=Path("outputs/coefficients_demo") / "stability_clustered.png",
    )
    print("✓ Generated: stability_clustered.png")
    return fig


def demo_stability_statistics():
    """Extract stability statistics."""
    print("\n=== Demo 8: Stability Statistics ===")
    stability = generate_synthetic_stability()
    feature_names = [f"Feature {i:02d}" for i in range(len(stability))]

    stats = get_stability_statistics(stability, feature_names)

    print(f"Mean Selection Frequency: {stats['global']['mean_frequency']:.4f}")
    print(f"Min Selection Frequency: {stats['global']['min_frequency']:.4f}")
    print(f"Max Selection Frequency: {stats['global']['max_frequency']:.4f}")

    print(f"\nMost Stable Features ({len(stats['consistency_metrics']['stable_features'])} total):")
    for feat in stats["consistency_metrics"]["stable_features"][:3]:
        print(f"  - {feat}")

    print(f"\nLeast Stable Features ({len(stats['consistency_metrics']['unstable_features'])} total):")
    for feat in stats["consistency_metrics"]["unstable_features"][:3]:
        print(f"  - {feat}")

    # Plot with feature names and improved styling
    fig = plot_feature_stability(
        stability,
        feature_names=feature_names,
        normalize="minmax",
        sort_features="std",
        show_bar_summary=True,
        bar_position="right",
        show_values=False,
        title="Feature Stability with Min-Max Normalization",
        colormap="RdYlGn",
        save_path=Path("outputs/coefficients_demo") / "stability_statistics.png",
    )
    print("✓ Generated: stability_statistics.png")
    return fig


def demo_integrated_workflow():
    """Complete integrated workflow combining coefficients and stability."""
    print("\n=== Demo 9: Integrated Workflow ===")

    # Generate larger datasets
    np.random.seed(123)
    n_features = 25
    n_classes = 3
    n_folds = 10

    # Coefficients: 25 features × 3 classes
    coefs = np.random.randn(n_features, n_classes) * np.array([1, 5, 2])
    coefs[:8] *= 3  # Important class 1 features

    # Stability: 25 features × 10 folds
    stability = np.random.binomial(1, 0.5, (n_features, n_folds))
    # Features that contribute to coefficients are more stable
    stability[:8, :] = np.random.binomial(1, 0.8, (8, n_folds))

    class_names = ["Control", "Treated 1", "Treated 2"]
    feature_names = [f"Feature {i:02d}" for i in range(n_features)]
    fold_names = [f"Fold {i}" for i in range(n_folds)]

    # Get statistics
    coef_stats = get_coefficient_statistics(coefs, class_names, feature_names)
    stability_stats = get_stability_statistics(stability, feature_names)

    print("Top 5 Important Features (by coefficient magnitude):")
    for ranking in coef_stats["rankings"]["by_mean_magnitude"][:5]:
        print(f"  {ranking['rank']:2d}. {ranking['feature']:15s}: {ranking['magnitude']:.4f}")

    print("\nTop 5 Most Stable Features (by selection frequency):")
    for ranking in stability_stats["rankings"]["by_frequency"][:5]:
        print(
            f"  {ranking['rank']:2d}. {ranking['feature']:15s}: "
            f"{ranking['frequency']:.4f} (in {ranking['appearances']}/{n_folds} folds)"
        )

    # Visualize both
    fig1 = plot_coefficients_heatmap(
        coefs,
        class_names=class_names,
        feature_names=feature_names,
        normalize="standard",
        sort_features="mean",
        title="Model Coefficients - Integrated Workflow",
        colormap="RdBu_r",
        show_values=False,
        save_path=Path("outputs/coefficients_demo") / "integrated_coefs.png",
    )
    print("✓ Generated: integrated_coefs.png")

    fig2 = plot_feature_stability(
        stability,
        feature_names=feature_names,
        fold_names=fold_names,
        normalize="frequency",
        sort_features="frequency",
        show_bar_summary=True,
        bar_position="right",
        title="Feature Stability - Integrated Workflow",
        colormap="RdYlGn",
        save_path=Path("outputs/coefficients_demo") / "integrated_stability.png",
    )
    print("✓ Generated: integrated_stability.png")

    return fig1, fig2


def main():
    """Run all demonstrations."""
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("FoodSpec: Coefficient Heatmaps & Feature Stability Visualizations")
    print("=" * 70)

    # Create output directory
    output_dir = Path("outputs/coefficients_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run demos
        demo_coefficients_basic()
        demo_coefficients_normalized()
        demo_coefficients_statistics()
        demo_stability_basic()
        demo_stability_normalized()
        demo_stability_bottom_bar()
        demo_stability_clustering()
        demo_stability_statistics()
        demo_integrated_workflow()

        print("\n" + "=" * 70)
        print("✓ All demonstrations completed successfully!")
        print(f"✓ Output directory: {output_dir.absolute()}")
        print("✓ Generated 10 visualizations")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        raise
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
