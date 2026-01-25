"""
Demo script for embeddings visualization module.

Demonstrates:
1. Basic 2D embedding visualization
2. 3D embedding projection
3. Class-based coloring
4. Batch + class coloring
5. Stage-based faceting
6. Confidence ellipses (68% and 95%)
7. Density contours
8. Integrated PCA vs UMAP comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from foodspec.viz import plot_embedding, plot_embedding_comparison, get_embedding_statistics


def demo_basic_2d_embedding():
    """Basic 2D embedding visualization."""
    print("Demo 1: Basic 2D Embedding")
    
    # Generate synthetic 2D embedding
    np.random.seed(42)
    embedding = np.random.randn(100, 2)
    
    fig = plot_embedding(
        embedding,
        embedding_name="Random 2D",
        title="Basic 2D Embedding Visualization",
        save_path=Path("outputs/embeddings_demo/01_basic_2d.png")
    )
    print("✓ Generated: 01_basic_2d.png")
    plt.close(fig)


def demo_3d_embedding():
    """3D embedding projection."""
    print("Demo 2: 3D Embedding Projection")
    
    # Generate synthetic 3D embedding
    np.random.seed(42)
    embedding = np.random.randn(100, 3)
    
    fig = plot_embedding(
        embedding,
        embedding_name="3D Projection",
        title="3D Embedding Visualization",
        figure_size=(10, 8),
        save_path=Path("outputs/embeddings_demo/02_3d_embedding.png")
    )
    print("✓ Generated: 02_3d_embedding.png")
    plt.close(fig)


def demo_class_coloring():
    """Class-based coloring."""
    print("Demo 3: Class-Based Coloring")
    
    # Generate embedding with class structure
    np.random.seed(42)
    n_per_class = 50
    classes = np.repeat(['Type-A', 'Type-B'], n_per_class)
    
    # Create embeddings with class separation
    class_a = np.random.randn(n_per_class, 2) + np.array([2, 2])
    class_b = np.random.randn(n_per_class, 2) + np.array([-2, -2])
    embedding = np.vstack([class_a, class_b])
    
    fig = plot_embedding(
        embedding,
        class_labels=classes,
        embedding_name="PCA",
        title="Embedding with Class Coloring",
        save_path=Path("outputs/embeddings_demo/03_class_coloring.png")
    )
    print("✓ Generated: 03_class_coloring.png")
    plt.close(fig)


def demo_batch_class_coloring():
    """Batch and class coloring combined."""
    print("Demo 4: Batch + Class Coloring")
    
    # Generate embedding with batch and class structure
    np.random.seed(42)
    n_samples = 120
    classes = np.tile(['ClassA', 'ClassB', 'ClassC'], n_samples // 3)
    batches = np.repeat(['Batch1', 'Batch2'], n_samples // 2)
    
    # Create embeddings with separation
    embedding = np.random.randn(n_samples, 2)
    for i in range(3):
        mask = classes == ['ClassA', 'ClassB', 'ClassC'][i]
        embedding[mask] += np.array([i * 1.5, i * 1.5])
    
    fig = plot_embedding(
        embedding,
        class_labels=classes,
        batch_labels=batches,
        embedding_name="UMAP",
        title="Embedding with Class and Batch Information",
        save_path=Path("outputs/embeddings_demo/04_batch_class.png")
    )
    print("✓ Generated: 04_batch_class.png")
    plt.close(fig)


def demo_stage_faceting():
    """Stage-based faceting."""
    print("Demo 5: Stage-Based Faceting")
    
    # Generate embedding with stage structure
    np.random.seed(42)
    n_per_stage = 100
    stages = np.repeat(['Raw', 'Preprocessed', 'Normalized'], n_per_stage)
    classes = np.tile(['Sample1', 'Sample2'], n_per_stage * 3 // 2)[:n_per_stage * 3]
    
    # Create embeddings with stage-dependent structure
    embedding_list = []
    for stage_idx in range(3):
        stage_emb = np.random.randn(n_per_stage, 2)
        # Add stage-specific shift
        stage_emb += np.array([stage_idx * 0.5, -stage_idx * 0.5])
        embedding_list.append(stage_emb)
    
    embedding = np.vstack(embedding_list)
    
    fig = plot_embedding(
        embedding,
        class_labels=classes,
        stage_labels=stages,
        embedding_name="t-SNE",
        title="Embedding by Processing Stage",
        figure_size=(15, 4),
        save_path=Path("outputs/embeddings_demo/05_stage_faceting.png")
    )
    print("✓ Generated: 05_stage_faceting.png")
    plt.close(fig)


def demo_confidence_ellipses_68():
    """Confidence ellipses (68% - 1 sigma)."""
    print("Demo 6: Confidence Ellipses (68%)")
    
    # Generate embedding with class structure
    np.random.seed(42)
    n_per_class = 80
    classes = np.repeat(['GroupA', 'GroupB'], n_per_class)
    
    # Create clustered embeddings
    group_a = np.random.randn(n_per_class, 2) * 0.5 + np.array([1, 1])
    group_b = np.random.randn(n_per_class, 2) * 0.5 + np.array([-1, -1])
    embedding = np.vstack([group_a, group_b])
    
    fig = plot_embedding(
        embedding,
        class_labels=classes,
        embedding_name="PCA",
        show_ellipses=True,
        ellipse_confidence=0.68,
        title="Embedding with 1σ Confidence Ellipses",
        alpha=0.6,
        save_path=Path("outputs/embeddings_demo/06_ellipses_68.png")
    )
    print("✓ Generated: 06_ellipses_68.png")
    plt.close(fig)


def demo_confidence_ellipses_95():
    """Confidence ellipses (95% - 2 sigma)."""
    print("Demo 7: Confidence Ellipses (95%)")
    
    # Generate embedding with class structure
    np.random.seed(42)
    n_per_class = 80
    classes = np.repeat(['TypeX', 'TypeY'], n_per_class)
    
    # Create clustered embeddings
    type_x = np.random.randn(n_per_class, 2) * 0.5 + np.array([1.5, 0])
    type_y = np.random.randn(n_per_class, 2) * 0.5 + np.array([-1.5, 0])
    embedding = np.vstack([type_x, type_y])
    
    fig = plot_embedding(
        embedding,
        class_labels=classes,
        embedding_name="UMAP",
        show_ellipses=True,
        ellipse_confidence=0.95,
        title="Embedding with 2σ Confidence Ellipses",
        alpha=0.6,
        save_path=Path("outputs/embeddings_demo/07_ellipses_95.png")
    )
    print("✓ Generated: 07_ellipses_95.png")
    plt.close(fig)


def demo_density_contours():
    """Density contours visualization."""
    print("Demo 8: Density Contours")
    
    # Generate embedding with overlapping classes
    np.random.seed(42)
    n_per_class = 80
    classes = np.repeat(['Cluster1', 'Cluster2'], n_per_class)
    
    # Create overlapping clusters
    cluster1 = np.random.randn(n_per_class, 2) * 0.7 + np.array([0.5, 0.5])
    cluster2 = np.random.randn(n_per_class, 2) * 0.7 + np.array([-0.5, -0.5])
    embedding = np.vstack([cluster1, cluster2])
    
    fig = plot_embedding(
        embedding,
        class_labels=classes,
        embedding_name="PCA",
        show_contours=True,
        n_contours=8,
        title="Embedding with Density Contours",
        alpha=0.6,
        save_path=Path("outputs/embeddings_demo/08_density_contours.png")
    )
    print("✓ Generated: 08_density_contours.png")
    plt.close(fig)


def demo_ellipses_and_contours():
    """Combining ellipses and contours."""
    print("Demo 9: Ellipses + Density Contours")
    
    # Generate embedding with class structure
    np.random.seed(42)
    n_per_class = 100
    classes = np.repeat(['ClassI', 'ClassII', 'ClassIII'], n_per_class)
    
    # Create well-separated clusters
    class_i = np.random.randn(n_per_class, 2) * 0.4 + np.array([2, 0])
    class_ii = np.random.randn(n_per_class, 2) * 0.4 + np.array([-2, 2])
    class_iii = np.random.randn(n_per_class, 2) * 0.4 + np.array([-2, -2])
    embedding = np.vstack([class_i, class_ii, class_iii])
    
    fig = plot_embedding(
        embedding,
        class_labels=classes,
        embedding_name="t-SNE",
        show_ellipses=True,
        ellipse_confidence=0.68,
        show_contours=True,
        n_contours=6,
        title="Embedding with Ellipses and Density Contours",
        alpha=0.6,
        marker_size=60,
        save_path=Path("outputs/embeddings_demo/09_ellipses_contours.png")
    )
    print("✓ Generated: 09_ellipses_contours.png")
    plt.close(fig)


def demo_pca_vs_umap():
    """Compare PCA vs UMAP embeddings."""
    print("Demo 10: PCA vs UMAP Comparison")
    
    # Generate synthetic high-dimensional data with class structure
    np.random.seed(42)
    n_per_class = 50
    n_features = 20
    
    # Create 3 classes in high-dimensional space
    class_a = np.random.randn(n_per_class, n_features) + np.random.randn(n_features) * 0.5
    class_b = np.random.randn(n_per_class, n_features) + np.random.randn(n_features) * 0.5
    class_c = np.random.randn(n_per_class, n_features) + np.random.randn(n_features) * 0.5
    
    X = np.vstack([class_a, class_b, class_c])
    classes = np.repeat(['ClassA', 'ClassB', 'ClassC'], n_per_class)
    
    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    pca_emb = pca.fit_transform(X)
    
    # t-SNE projection (faster alternative to UMAP)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_emb = tsne.fit_transform(X)
    
    embeddings = {"PCA": pca_emb, "t-SNE": tsne_emb}
    
    fig = plot_embedding_comparison(
        embeddings,
        class_labels=classes,
        show_ellipses=True,
        ellipse_confidence=0.68,
        alpha=0.7,
        title="PCA vs t-SNE Dimensionality Reduction",
        figure_size=(14, 5),
        save_path=Path("outputs/embeddings_demo/10_pca_vs_tsne.png")
    )
    print("✓ Generated: 10_pca_vs_tsne.png")
    plt.close(fig)


def demo_statistics_extraction():
    """Statistics extraction and analysis."""
    print("Demo 11: Statistics Extraction")
    
    # Generate embedding with class structure
    np.random.seed(42)
    n_per_class = 100
    classes = np.repeat(['TypeA', 'TypeB', 'TypeC'], n_per_class)
    
    # Create embeddings with distinct separation
    type_a = np.random.randn(n_per_class, 2) * 0.5 + np.array([2, 2])
    type_b = np.random.randn(n_per_class, 2) * 0.5 + np.array([-2, 2])
    type_c = np.random.randn(n_per_class, 2) * 0.5 + np.array([0, -2])
    embedding = np.vstack([type_a, type_b, type_c])
    
    # Extract statistics
    stats = get_embedding_statistics(embedding, classes)
    
    # Print statistics
    print("\n  Embedding Statistics:")
    for cls_name in ['TypeA', 'TypeB', 'TypeC']:
        s = stats[cls_name]
        print(f"\n  {cls_name}:")
        print(f"    Samples: {s['n_samples']}")
        print(f"    Mean (x,y): ({s['mean_x']:.3f}, {s['mean_y']:.3f})")
        print(f"    Std (x,y): ({s['std_x']:.3f}, {s['std_y']:.3f})")
        print(f"    Separation: {s['separation']:.3f}")
    
    # Visualize
    fig = plot_embedding(
        embedding,
        class_labels=classes,
        embedding_name="PCA",
        show_ellipses=True,
        show_contours=True,
        title="Embedding with Extracted Statistics",
        save_path=Path("outputs/embeddings_demo/11_statistics.png")
    )
    print("\n✓ Generated: 11_statistics.png")
    plt.close(fig)


def demo_integrated_workflow():
    """Complete integrated workflow."""
    print("Demo 12: Integrated Workflow (3 Samples)")
    
    # Generate data for 3 samples with batch and stage information
    np.random.seed(42)
    samples = ['Sample1', 'Sample2', 'Sample3']
    stages = ['Raw', 'Preprocessed', 'Normalized']
    batches = ['Batch1', 'Batch2']
    
    all_figs = []
    
    for sample_idx, sample_name in enumerate(samples):
        print(f"\n  Processing {sample_name}...")
        
        # Generate embedding for this sample
        n_per_stage = 60
        embedding_list = []
        stage_labels_list = []
        batch_labels_list = []
        
        for stage_idx, stage_name in enumerate(stages):
            # Create embeddings with stage progression
            stage_emb = np.random.randn(n_per_stage, 2)
            stage_emb = stage_emb * (1 - 0.2 * stage_idx)  # Reduce variance as we process
            stage_emb += np.array([stage_idx * 1.0, -stage_idx * 0.5])
            
            embedding_list.append(stage_emb)
            stage_labels_list.extend([stage_name] * n_per_stage)
            batch_labels_list.extend(np.tile(batches, n_per_stage // 2))
        
        embedding = np.vstack(embedding_list)
        stage_labels = np.array(stage_labels_list)
        batch_labels = np.array(batch_labels_list)
        
        # Create faceted visualization
        fig = plot_embedding(
            embedding,
            batch_labels=batch_labels,
            stage_labels=stage_labels,
            embedding_name="PCA",
            show_ellipses=True,
            show_contours=True,
            title=f"Integrated Workflow - {sample_name}",
            figure_size=(15, 4),
            save_path=Path(f"outputs/embeddings_demo/12_workflow_{sample_idx + 1}.png")
        )
        print(f"  ✓ Generated: 12_workflow_{sample_idx + 1}.png")
        all_figs.append(fig)
    
    for fig in all_figs:
        plt.close(fig)


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("EMBEDDINGS VISUALIZATION MODULE - DEMONSTRATIONS")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("outputs/embeddings_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all demos
    demo_basic_2d_embedding()
    demo_3d_embedding()
    demo_class_coloring()
    demo_batch_class_coloring()
    demo_stage_faceting()
    demo_confidence_ellipses_68()
    demo_confidence_ellipses_95()
    demo_density_contours()
    demo_ellipses_and_contours()
    demo_pca_vs_umap()
    demo_statistics_extraction()
    demo_integrated_workflow()
    
    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nGenerated visualizations: {output_dir}/")
    print("\nFiles created:")
    for i, filepath in enumerate(sorted(output_dir.glob("*.png")), 1):
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  {i:2d}. {filepath.name:40s} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
