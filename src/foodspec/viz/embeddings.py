"""
PCA/UMAP Embedding Visualization Module

Provides visualization functions for dimensionality reduction results (PCA, UMAP, t-SNE).
Supports multi-factor coloring (batch, stage, class), confidence ellipses, and density contours.

Functions:
    plot_embedding(): Main embedding visualization with coloring and ellipses
    plot_embedding_comparison(): Side-by-side comparison of different embeddings
    get_embedding_statistics(): Extract per-group statistics from embeddings
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from foodspec.viz.style import apply_style

if TYPE_CHECKING:
    from foodspec.reporting.schema import RunBundle


def _validate_embedding(embedding: np.ndarray) -> None:
    """
    Validate embedding array dimensions and values.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding coordinates, shape (n_samples, n_dimensions) where n_dimensions in [2, 3]

    Raises
    ------
    ValueError
        If embedding is not 2D, empty, non-numeric, or has wrong dimensions
    """
    if not isinstance(embedding, np.ndarray):
        raise ValueError("embedding must be a numpy array")

    if embedding.ndim != 2:
        raise ValueError(f"embedding must be 2D, got shape {embedding.shape}")

    if embedding.size == 0:
        raise ValueError("embedding cannot be empty")

    if embedding.shape[1] not in [2, 3]:
        raise ValueError(f"embedding must have 2 or 3 dimensions, got {embedding.shape[1]}")

    if not np.issubdtype(embedding.dtype, np.number):
        raise ValueError("embedding must contain numeric values")

    if np.any(~np.isfinite(embedding)):
        raise ValueError("embedding contains non-finite values (NaN or Inf)")


def _validate_labels(labels: Optional[np.ndarray], expected_length: int) -> None:
    """
    Validate label array for embedding samples.

    Parameters
    ----------
    labels : np.ndarray or None
        Labels for samples, shape (n_samples,)
    expected_length : int
        Expected number of labels

    Raises
    ------
    ValueError
        If labels length doesn't match embedding or contains invalid types
    """
    if labels is None:
        return

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    if len(labels) != expected_length:
        raise ValueError(f"labels length {len(labels)} doesn't match embedding {expected_length}")


def _get_embedding_colors(labels: Optional[np.ndarray], colormap: str) -> Dict[Any, Tuple[float, float, float]]:
    """
    Generate color mapping for embedding labels.

    Parameters
    ----------
    labels : np.ndarray or None
        Labels for samples
    colormap : str
        Matplotlib colormap name

    Returns
    -------
    dict
        Mapping from unique label to RGB color tuple
    """
    if labels is None:
        return {}

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    if n_labels == 1:
        return {unique_labels[0]: (0.2, 0.4, 0.8)}

    cmap = plt.get_cmap(colormap)
    colors = {}
    for i, label in enumerate(unique_labels):
        color = cmap(i / (n_labels - 1))
        colors[label] = color[:3]  # RGB only, ignore alpha

    return colors


def _fit_confidence_ellipse(points: np.ndarray, confidence: float = 0.68) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit confidence ellipse to 2D point cloud.

    Parameters
    ----------
    points : np.ndarray
        2D points, shape (n_points, 2)
    confidence : float
        Confidence level (default 0.68 for 1-sigma)

    Returns
    -------
    center : np.ndarray
        Ellipse center (2,)
    width_height : np.ndarray
        [width, height] of ellipse at given confidence
    angle : float
        Rotation angle in degrees
    """
    if len(points) < 2:
        center = points[0]
        return center, np.array([0.1, 0.1]), 0.0

    center = np.mean(points, axis=0)

    # Compute covariance
    cov_matrix = np.cov(points.T)

    # Handle singular covariance
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix, 0], [0, cov_matrix]])

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Scale by chi-squared quantile for confidence level
    chi2_val = chi2.ppf(confidence, df=2)
    scales = 2 * np.sqrt(chi2_val * eigenvalues)

    # Rotation angle
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

    return center, scales, angle


def _extract_contour_region(
    points: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract points within bounding region for contour calculation.

    Parameters
    ----------
    points : np.ndarray
        2D points, shape (n_points, 2)
    xlim : tuple
        (xmin, xmax) bounds
    ylim : tuple
        (ymin, ymax) bounds

    Returns
    -------
    xx : np.ndarray
        Grid X coordinates
    yy : np.ndarray
        Grid Y coordinates
    """
    mask = (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) & (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1])

    return mask


def plot_embedding(
    embedding: np.ndarray,
    class_labels: Optional[np.ndarray] = None,
    batch_labels: Optional[np.ndarray] = None,
    stage_labels: Optional[np.ndarray] = None,
    embedding_name: str = "Embedding",
    class_colormap: str = "tab10",
    batch_colormap: str = "Set1",
    stage_colormap: str = "Pastel1",
    show_ellipses: bool = False,
    ellipse_confidence: float = 0.68,
    show_contours: bool = False,
    n_contours: int = 5,
    alpha: float = 0.7,
    marker_size: int = 50,
    title: Optional[str] = None,
    figure_size: Tuple[int, int] = (12, 9),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize dimensionality reduction embedding with multi-factor coloring.

    Creates scatter plot of embedding coordinates with optional confidence ellipses,
    density contours, and multi-factor coloring (batch, stage, class).

    Parameters
    ----------
    embedding : np.ndarray
        Embedding coordinates, shape (n_samples, 2) or (n_samples, 3)
    class_labels : np.ndarray, optional
        Class labels for samples, shape (n_samples,). Default: None (no class coloring)
    batch_labels : np.ndarray, optional
        Batch labels for samples, shape (n_samples,). Used as marker style. Default: None
    stage_labels : np.ndarray, optional
        Stage labels for samples, shape (n_samples,). Used for faceting. Default: None
    embedding_name : str, optional
        Name of embedding (e.g., "PCA", "UMAP"). Default: "Embedding"
    class_colormap : str, optional
        Colormap for class labels. Default: "tab10"
    batch_colormap : str, optional
        Colormap for batch labels. Default: "Set1"
    stage_colormap : str, optional
        Colormap for stage labels. Default: "Pastel1"
    show_ellipses : bool, optional
        Show confidence ellipses per class. Default: False
    ellipse_confidence : float, optional
        Confidence level for ellipses (0.68 for 1-sigma, 0.95 for 2-sigma). Default: 0.68
    show_contours : bool, optional
        Show density contours (KDE). Default: False
    n_contours : int, optional
        Number of contour levels. Default: 5
    alpha : float, optional
        Transparency of points (0-1). Default: 0.7
    marker_size : int, optional
        Marker size in points. Default: 50
    title : str, optional
        Figure title. Default: Auto-generated from embedding_name
    figure_size : tuple, optional
        Figure dimensions (width, height) in inches. Default: (12, 9)
    save_path : str or Path, optional
        Path to save PNG. Default: None (display only)
    dpi : int, optional
        DPI for saved PNG. Default: 300

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the embedding visualization

    Raises
    ------
    ValueError
        If embedding has invalid shape, non-finite values, or label length mismatches

    Examples
    --------
    >>> import numpy as np
    >>> from foodspec.viz import plot_embedding
    >>> embedding = np.random.randn(100, 2)
    >>> classes = np.random.choice(['A', 'B', 'C'], 100)
    >>> fig = plot_embedding(embedding, class_labels=classes)
    >>> fig.savefig('embedding.png')

    Basic example with class coloring:

    >>> embedding = np.random.randn(100, 2)
    >>> classes = np.repeat(['Class1', 'Class2', 'Class3', 'Class4'], 25)
    >>> fig = plot_embedding(
    ...     embedding,
    ...     class_labels=classes,
    ...     embedding_name="PCA",
    ...     show_ellipses=True
    ... )

    Advanced example with batch and stage coloring:

    >>> batches = np.repeat(['Batch1', 'Batch2'], 50)
    >>> stages = np.tile(['Raw', 'Processed'], 50)
    >>> fig = plot_embedding(
    ...     embedding,
    ...     class_labels=classes,
    ...     batch_labels=batches,
    ...     stage_labels=stages,
    ...     show_ellipses=True,
    ...     show_contours=True
    ... )
    """
    # Validate inputs
    _validate_embedding(embedding)
    _validate_labels(class_labels, embedding.shape[0])
    _validate_labels(batch_labels, embedding.shape[0])
    _validate_labels(stage_labels, embedding.shape[0])

    # Determine dimensionality and projection
    n_dims = embedding.shape[1]
    is_3d = n_dims == 3

    # If stage labels provided, create faceted plot
    if stage_labels is not None:
        unique_stages = np.unique(stage_labels)
        n_stages = len(unique_stages)

        # Determine grid layout
        if n_stages <= 2:
            n_rows, n_cols = 1, n_stages
        elif n_stages <= 4:
            n_rows, n_cols = 2, (n_stages + 1) // 2
        else:
            n_rows, n_cols = (n_stages + 2) // 3, 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size)
        axes = np.atleast_1d(axes).flatten()

        for ax_idx, stage in enumerate(unique_stages):
            ax = axes[ax_idx]
            mask = stage_labels == stage
            stage_embedding = embedding[mask]
            stage_classes = class_labels[mask] if class_labels is not None else None
            stage_batches = batch_labels[mask] if batch_labels is not None else None

            _plot_embedding_2d(
                ax,
                stage_embedding,
                stage_classes,
                stage_batches,
                class_colormap,
                batch_colormap,
                alpha,
                marker_size,
                show_ellipses,
                ellipse_confidence,
                show_contours,
                n_contours,
            )
            ax.set_title(f"{embedding_name} - {stage}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(unique_stages), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title or f"{embedding_name} by Stage", fontsize=14, fontweight="bold")
        plt.tight_layout()

    else:
        # Single plot (2D or 3D)
        if is_3d:
            fig = plt.figure(figsize=figure_size)
            ax = fig.add_subplot(111, projection="3d")
            _plot_embedding_3d(
                ax,
                embedding,
                class_labels,
                batch_labels,
                class_colormap,
                batch_colormap,
                alpha,
                marker_size,
                show_ellipses,
                ellipse_confidence,
            )
        else:
            fig, ax = plt.subplots(figsize=figure_size)
            _plot_embedding_2d(
                ax,
                embedding,
                class_labels,
                batch_labels,
                class_colormap,
                batch_colormap,
                alpha,
                marker_size,
                show_ellipses,
                ellipse_confidence,
                show_contours,
                n_contours,
            )

        ax.set_title(title or f"{embedding_name} Embedding", fontsize=14, fontweight="bold")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        if not is_3d:
            ax.set_zlabel("Component 3") if is_3d else None
            ax.grid(True, alpha=0.3)

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_pca_scatter(
    data_bundle: "RunBundle | dict",
    *,
    outdir=None,
    name=None,
    fmt=("png", "svg"),
    dpi=300,
    seed=0,
):
    """Plot PCA scatter from a feature matrix payload."""
    from foodspec._version import __version__
    from foodspec.reporting.schema import RunBundle
    from foodspec.viz.save import save_figure

    apply_style()
    payload = data_bundle if isinstance(data_bundle, dict) else {}
    if isinstance(data_bundle, RunBundle):
        payload = {}
    X = np.asarray(payload.get("X", np.random.default_rng(seed).normal(size=(50, 5))), dtype=float)
    labels = payload.get("labels")
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(X)
    except Exception:
        coords = X[:, :2]

    fig, ax = plt.subplots(figsize=(4, 3))
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, c="#2a6fdb", alpha=0.8)
    else:
        labels = np.asarray(labels)
        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1], s=30, alpha=0.8, label=str(lbl))
        ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Scatter")

    if outdir is not None:
        base = Path(outdir) / "figures" / (name or "pca_scatter")
        save_figure(
            fig,
            base,
            metadata={
                "description": "PCA scatter plot",
                "inputs": {"shape": list(X.shape)},
                "code_version": __version__,
                "seed": seed,
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_umap_scatter(
    data_bundle: "RunBundle | dict",
    *,
    outdir=None,
    name=None,
    fmt=("png", "svg"),
    dpi=300,
    seed=0,
):
    """Plot UMAP scatter from a feature matrix payload (fallbacks to PCA)."""
    from foodspec._version import __version__
    from foodspec.reporting.schema import RunBundle
    from foodspec.viz.save import save_figure

    apply_style()
    payload = data_bundle if isinstance(data_bundle, dict) else {}
    if isinstance(data_bundle, RunBundle):
        payload = {}
    X = np.asarray(payload.get("X", np.random.default_rng(seed).normal(size=(50, 5))), dtype=float)
    labels = payload.get("labels")
    used_fallback = False
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(random_state=seed)
        coords = reducer.fit_transform(X)
    except Exception:
        used_fallback = True
        try:
            from sklearn.decomposition import PCA

            coords = PCA(n_components=2, random_state=seed).fit_transform(X)
        except Exception:
            coords = X[:, :2]

    fig, ax = plt.subplots(figsize=(4, 3))
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, c="#2a6fdb", alpha=0.8)
    else:
        labels = np.asarray(labels)
        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1], s=30, alpha=0.8, label=str(lbl))
        ax.legend()
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    title = "UMAP Scatter" + (" (PCA fallback)" if used_fallback else "")
    ax.set_title(title)

    if outdir is not None:
        base = Path(outdir) / "figures" / (name or "umap_scatter")
        save_figure(
            fig,
            base,
            metadata={
                "description": "UMAP scatter plot",
                "inputs": {"shape": list(X.shape), "fallback": used_fallback},
                "code_version": __version__,
                "seed": seed,
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def _plot_embedding_2d(
    ax: plt.Axes,
    embedding: np.ndarray,
    class_labels: Optional[np.ndarray],
    batch_labels: Optional[np.ndarray],
    class_colormap: str,
    batch_colormap: str,
    alpha: float,
    marker_size: int,
    show_ellipses: bool,
    ellipse_confidence: float,
    show_contours: bool,
    n_contours: int,
) -> None:
    """Helper function to plot 2D embedding."""
    if class_labels is None:
        # Plot all points in single color
        ax.scatter(embedding[:, 0], embedding[:, 1], s=marker_size, alpha=alpha, c="steelblue")
    else:
        # Color by class labels
        class_colors = _get_embedding_colors(class_labels, class_colormap)
        unique_classes = np.unique(class_labels)

        for cls in unique_classes:
            mask = class_labels == cls
            color = class_colors[cls]
            ax.scatter(embedding[mask, 0], embedding[mask, 1], label=str(cls), s=marker_size, alpha=alpha, c=[color])

        ax.legend(loc="best", framealpha=0.9)

        # Add confidence ellipses if requested
        if show_ellipses and len(embedding) > 2:
            for cls in unique_classes:
                mask = class_labels == cls
                class_points = embedding[mask]
                if len(class_points) > 2:
                    center, scales, angle = _fit_confidence_ellipse(class_points[:, :2], ellipse_confidence)
                    ellipse = Ellipse(
                        center,
                        scales[0],
                        scales[1],
                        angle=angle,
                        edgecolor=class_colors[cls],
                        facecolor="none",
                        linewidth=2,
                        linestyle="--",
                        alpha=0.8,
                    )
                    ax.add_patch(ellipse)

    # Add density contours if requested
    if show_contours and len(embedding) > 10:
        try:
            from scipy.stats import gaussian_kde

            x = embedding[:, 0]
            y = embedding[:, 1]

            # Create grid
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            margin_x = (x_max - x_min) * 0.1
            margin_y = (y_max - y_min) * 0.1

            xx, yy = np.mgrid[x_min - margin_x : x_max + margin_x : 100j, y_min - margin_y : y_max + margin_y : 100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])

            # Compute KDE
            kde = gaussian_kde(np.vstack([x, y]))
            Z = kde(positions).reshape(xx.shape)

            # Plot contours
            ax.contour(xx, yy, Z, levels=n_contours, colors="gray", alpha=0.5, linewidths=1)
        except Exception:
            pass  # Silently skip contours if KDE fails


def _plot_embedding_3d(
    ax: plt.Axes,
    embedding: np.ndarray,
    class_labels: Optional[np.ndarray],
    batch_labels: Optional[np.ndarray],
    class_colormap: str,
    batch_colormap: str,
    alpha: float,
    marker_size: int,
    show_ellipses: bool,
    ellipse_confidence: float,
) -> None:
    """Helper function to plot 3D embedding."""
    if class_labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=marker_size, alpha=alpha, c="steelblue")
    else:
        class_colors = _get_embedding_colors(class_labels, class_colormap)
        unique_classes = np.unique(class_labels)

        for cls in unique_classes:
            mask = class_labels == cls
            color = class_colors[cls]
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                label=str(cls),
                s=marker_size,
                alpha=alpha,
                c=[color],
            )

        ax.legend(loc="best", framealpha=0.9)

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")


def plot_embedding_comparison(
    embeddings: Dict[str, np.ndarray],
    class_labels: Optional[np.ndarray] = None,
    class_colormap: str = "tab10",
    show_ellipses: bool = False,
    ellipse_confidence: float = 0.68,
    alpha: float = 0.7,
    marker_size: int = 50,
    title: Optional[str] = None,
    figure_size: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Compare multiple embeddings side-by-side with unified coloring.

    Parameters
    ----------
    embeddings : dict
        Dictionary mapping embedding name to coordinates array (n_samples, 2)
    class_labels : np.ndarray, optional
        Class labels for samples, shape (n_samples,). Default: None
    class_colormap : str, optional
        Colormap for class labels. Default: "tab10"
    show_ellipses : bool, optional
        Show confidence ellipses per class. Default: False
    ellipse_confidence : float, optional
        Confidence level for ellipses. Default: 0.68
    alpha : float, optional
        Transparency of points. Default: 0.7
    marker_size : int, optional
        Marker size in points. Default: 50
    title : str, optional
        Figure title. Default: "Embedding Comparison"
    figure_size : tuple, optional
        Figure dimensions. Default: (15, 5)
    save_path : str or Path, optional
        Path to save PNG. Default: None
    dpi : int, optional
        DPI for saved PNG. Default: 300

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with side-by-side embedding plots

    Raises
    ------
    ValueError
        If embeddings have mismatched sample counts or invalid shapes
    """
    # Validate embeddings
    n_embeddings = len(embeddings)
    embedding_names = list(embeddings.keys())

    if n_embeddings == 0:
        raise ValueError("embeddings dictionary is empty")

    # Check all have same number of samples
    n_samples = embeddings[embedding_names[0]].shape[0]
    for name, emb in embeddings.items():
        _validate_embedding(emb)
        if emb.shape[0] != n_samples:
            raise ValueError(f"Embedding '{name}' has {emb.shape[0]} samples, expected {n_samples}")

    if class_labels is not None:
        _validate_labels(class_labels, n_samples)

    # Create subplots
    fig, axes = plt.subplots(1, n_embeddings, figsize=figure_size)
    axes = np.atleast_1d(axes)

    # Plot each embedding
    for ax, name in zip(axes, embedding_names):
        emb = embeddings[name]

        _plot_embedding_2d(
            ax,
            emb,
            class_labels,
            None,
            class_colormap,
            "Set1",
            alpha,
            marker_size,
            show_ellipses,
            ellipse_confidence,
            False,
            5,
        )

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title or "Embedding Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def get_embedding_statistics(
    embedding: np.ndarray, class_labels: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Extract per-group statistics from embedding.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding coordinates, shape (n_samples, 2 or 3)
    class_labels : np.ndarray, optional
        Class labels for samples. Default: None (global statistics only)

    Returns
    -------
    dict
        Statistics per class/group:
        {
            'global' or class_name: {
                'n_samples': int,
                'mean_x': float,
                'mean_y': float,
                'std_x': float,
                'std_y': float,
                'range_x': float,
                'range_y': float,
                'separation': float (distance to nearest other class mean)
            }
        }

    Examples
    --------
    >>> embedding = np.random.randn(100, 2)
    >>> classes = np.repeat(['A', 'B'], 50)
    >>> stats = get_embedding_statistics(embedding, classes)
    >>> print(f"Class A samples: {stats['A']['n_samples']}")
    """
    _validate_embedding(embedding)
    _validate_labels(class_labels, embedding.shape[0])

    stats = {}

    if class_labels is None:
        # Global statistics
        stats["global"] = {
            "n_samples": len(embedding),
            "mean_x": float(np.mean(embedding[:, 0])),
            "mean_y": float(np.mean(embedding[:, 1])),
            "std_x": float(np.std(embedding[:, 0])),
            "std_y": float(np.std(embedding[:, 1])),
            "range_x": float(np.ptp(embedding[:, 0])),
            "range_y": float(np.ptp(embedding[:, 1])),
            "separation": 0.0,
        }
    else:
        unique_classes = np.unique(class_labels)
        means = {}

        # Compute per-class statistics
        for cls in unique_classes:
            mask = class_labels == cls
            class_emb = embedding[mask]

            stats[str(cls)] = {
                "n_samples": len(class_emb),
                "mean_x": float(np.mean(class_emb[:, 0])),
                "mean_y": float(np.mean(class_emb[:, 1])),
                "std_x": float(np.std(class_emb[:, 0])),
                "std_y": float(np.std(class_emb[:, 1])),
                "range_x": float(np.ptp(class_emb[:, 0])),
                "range_y": float(np.ptp(class_emb[:, 1])),
                "separation": 0.0,
            }

            means[str(cls)] = np.array([stats[str(cls)]["mean_x"], stats[str(cls)]["mean_y"]])

        # Compute separation (distance to nearest other class)
        for cls in unique_classes:
            cls_mean = means[str(cls)]
            min_dist = float("inf")

            for other_cls in unique_classes:
                if other_cls != cls:
                    other_mean = means[str(other_cls)]
                    dist = np.linalg.norm(cls_mean - other_mean)
                    min_dist = min(min_dist, dist)

            stats[str(cls)]["separation"] = float(min_dist) if min_dist != float("inf") else 0.0

    return stats
