"""
Feature stability map visualization module.

Provides visualization of feature selection frequency across cross-validation
folds and bootstrap samples. Includes stability heatmaps, bar summaries, and
clustering analysis.

Main Functions:
    - plot_feature_stability: Visualize feature stability across folds
    - get_stability_statistics: Extract stability metrics and rankings
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist


def _validate_stability_matrix(
    stability_matrix: np.ndarray,
) -> Tuple[int, int]:
    """
    Validate stability matrix shape and return dimensions.

    Parameters
    ----------
    stability_matrix : np.ndarray
        Feature × fold/sample binary or count matrix

    Returns
    -------
    Tuple[int, int]
        (n_features, n_folds)

    Raises
    ------
    ValueError
        If matrix is empty or invalid shape
    """
    stability_matrix = np.asarray(stability_matrix)
    if stability_matrix.size == 0:
        raise ValueError("Stability matrix cannot be empty")
    if stability_matrix.ndim != 2:
        raise ValueError(
            f"Stability matrix must be 2D, got shape {stability_matrix.shape}"
        )
    return stability_matrix.shape


def _normalize_stability(
    stability_matrix: np.ndarray, method: str = "minmax"
) -> np.ndarray:
    """
    Normalize stability values to [0, 1].

    Parameters
    ----------
    stability_matrix : np.ndarray
        Feature × fold matrix with selection counts
    method : str
        "minmax": Per-feature min-max to [0, 1]
        "zscore": Per-feature z-score normalization
        "frequency": Divide by total folds

    Returns
    -------
    np.ndarray
        Normalized stability matrix in [0, 1]
    """
    if stability_matrix.size == 0:
        return stability_matrix.copy()

    result = stability_matrix.copy().astype(float)

    if method == "frequency":
        # Normalize by number of folds (assume max value = n_folds)
        max_val = np.max(result)
        if max_val > 0:
            result = result / max_val
        return result

    elif method == "minmax":
        # Per-feature min-max scaling
        mins = np.min(result, axis=1, keepdims=True)
        maxs = np.max(result, axis=1, keepdims=True)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        return (result - mins) / ranges

    elif method == "zscore":
        # Per-feature z-score normalization, then shift to [0, 1]
        means = np.mean(result, axis=1, keepdims=True)
        stds = np.std(result, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        normalized = (result - means) / stds
        return (normalized + 3) / 6  # Shift to [0, 1] assuming ±3σ

    return result


def _compute_feature_frequency(
    stability_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute selection frequency (mean across folds) for each feature.

    Parameters
    ----------
    stability_matrix : np.ndarray
        Feature × fold matrix

    Returns
    -------
    np.ndarray
        Selection frequency per feature
    """
    return np.mean(stability_matrix, axis=1)


def _sort_by_stability(
    stability_matrix: np.ndarray, method: str = "frequency"
) -> np.ndarray:
    """
    Sort features by stability metric.

    Parameters
    ----------
    stability_matrix : np.ndarray
        Feature × fold matrix
    method : str
        "frequency": Sort by mean selection frequency
        "std": Sort by std (least to most variable)
        "consistency": Sort by % appearing in all/most folds

    Returns
    -------
    np.ndarray
        Indices of sorted features (ascending)
    """
    if method == "frequency":
        stability_scores = np.mean(stability_matrix, axis=1)
    elif method == "std":
        stability_scores = np.std(stability_matrix, axis=1)
    elif method == "consistency":
        # Higher frequency of appearance in more folds = lower variance
        stability_scores = np.std(stability_matrix, axis=1)
    else:
        return np.arange(len(stability_matrix))

    return np.argsort(stability_scores)


def _apply_clustering(
    stability_matrix: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """
    Apply hierarchical clustering to features by stability pattern.

    Parameters
    ----------
    stability_matrix : np.ndarray
        Feature × fold matrix

    Returns
    -------
    Tuple[np.ndarray, dict]
        Sorted feature indices and linkage matrix
    """
    if len(stability_matrix) < 2:
        return np.arange(len(stability_matrix)), {}

    try:
        # Compute pairwise distances using binary features
        distances = pdist(stability_matrix, metric="euclidean")
        Z = linkage(distances, method="ward")

        # Get order from dendrogram
        from scipy.cluster.hierarchy import dendrogram as dendro_fn

        dendro = dendro_fn(Z, no_plot=True)
        return np.array(dendro["leaves"]), Z

    except Exception:
        # Fallback to no clustering
        return np.arange(len(stability_matrix)), {}


def plot_feature_stability(
    stability_matrix: np.ndarray,
    fold_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    normalize: Union[bool, str] = "frequency",
    sort_features: Union[bool, str] = "frequency",
    cluster_features: bool = False,
    colormap: str = "RdYlGn",
    show_bar_summary: bool = True,
    bar_position: str = "right",
    show_values: bool = False,
    value_decimals: int = 2,
    title: str = "Feature Stability Across Folds",
    figure_size: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize feature selection stability across cross-validation folds.

    Creates a features × folds heatmap showing feature selection patterns,
    with optional frequency bar summary and hierarchical clustering.

    Parameters
    ----------
    stability_matrix : np.ndarray
        Binary or count matrix (n_features, n_folds) indicating feature selection
    fold_names : List[str], optional
        Names for folds. Default: Fold 0, 1, ...
    feature_names : List[str], optional
        Names for features. Default: Feature 0, 1, ...
    normalize : bool or str, default="frequency"
        Normalization method:
        - False/"none": No normalization (use raw counts)
        - True/"frequency": Divide by max (convert to [0, 1])
        - "minmax": Per-feature min-max scaling
        - "zscore": Per-feature z-score normalization
    sort_features : bool or str, default="frequency"
        Sort features by metric:
        - False/"none": Original order
        - True/"frequency": Sort by mean selection frequency
        - "std": Sort by consistency (least to most variable)
        - "consistency": Same as std
    cluster_features : bool, default=False
        Apply hierarchical clustering instead of sorting
    colormap : str, default="RdYlGn"
        Colormap name (RdYlGn, YlOrRd, viridis, etc.)
    show_bar_summary : bool, default=True
        Show bar plot summarizing mean frequency per feature
    bar_position : str, default="right"
        Position of bar summary: "right", "bottom", "left", "none"
    show_values : bool, default=False
        Annotate heatmap cells with values
    value_decimals : int, default=2
        Decimal places in annotations
    title : str
        Figure title
    figure_size : Tuple[int, int]
        Figure size (width, height)
    save_path : Path, optional
        Path to save PNG
    dpi : int
        DPI for saved figure

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Raises
    ------
    ValueError
        If stability_matrix is empty or invalid shape
    """
    # Validate input
    n_features, n_folds = _validate_stability_matrix(stability_matrix)
    stability_matrix = np.asarray(stability_matrix)

    # Generate default names
    if fold_names is None:
        fold_names = [f"Fold {i}" for i in range(n_folds)]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    # Validate name lengths
    if len(fold_names) != n_folds:
        raise ValueError(
            f"fold_names length {len(fold_names)} "
            f"does not match matrix columns {n_folds}"
        )
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length {len(feature_names)} "
            f"does not match matrix rows {n_features}"
        )

    # Normalize
    if normalize is True:
        normalize = "frequency"
    elif normalize is False:
        normalize = "none"
    data_to_plot = (
        stability_matrix.copy()
        if normalize == "none"
        else _normalize_stability(stability_matrix, method=normalize)
    )

    # Sort or cluster features
    feature_order = np.arange(n_features)
    if cluster_features:
        feature_order, _ = _apply_clustering(data_to_plot)
    elif sort_features not in (False, "none"):
        if sort_features is True:
            sort_features = "frequency"
        feature_order = _sort_by_stability(data_to_plot, method=sort_features)

    data_to_plot = data_to_plot[feature_order]
    feature_names = [feature_names[i] for i in feature_order]

    # Compute frequency for bar summary
    frequencies = _compute_feature_frequency(data_to_plot)

    # Create figure with subplots if bar summary requested
    if show_bar_summary and bar_position == "right":
        fig = plt.figure(figsize=figure_size)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
    elif show_bar_summary and bar_position == "bottom":
        fig = plt.figure(figsize=figure_size)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
    else:
        fig, ax_main = plt.subplots(figsize=figure_size)
        ax_bar = None

    # Plot heatmap
    im = ax_main.imshow(data_to_plot, cmap=colormap, aspect="auto", vmin=0, vmax=1)

    # Set axis labels
    ax_main.set_xticks(np.arange(n_folds))
    ax_main.set_yticks(np.arange(len(feature_names)))
    ax_main.set_xticklabels(fold_names, rotation=45, ha="right")
    ax_main.set_yticklabels(feature_names, fontsize=9)

    # Grid
    ax_main.set_xticks(np.arange(n_folds) - 0.5, minor=True)
    ax_main.set_yticks(np.arange(len(feature_names)) - 0.5, minor=True)
    ax_main.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    # Value annotations
    if show_values:
        for i in range(len(feature_names)):
            for j in range(n_folds):
                value = data_to_plot[i, j]
                text_color = "white" if value > 0.5 else "black"
                annotation = f"{value:.{value_decimals}f}"
                ax_main.text(
                    j, i, annotation, ha="center", va="center",
                    color=text_color, fontsize=8
                )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label("Selection Frequency", rotation=270, labelpad=15)

    # Labels
    ax_main.set_xlabel("Fold", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Feature", fontsize=12, fontweight="bold")

    # Bar summary
    if show_bar_summary and ax_bar is not None:
        if bar_position == "right":
            ax_bar.barh(range(len(feature_names)), frequencies, color="steelblue")
            ax_bar.set_yticks([])
            ax_bar.set_xlabel("Mean Frequency", fontsize=10)
            ax_bar.set_xlim(0, 1)
        else:  # bottom
            ax_bar.bar(range(len(feature_names)), frequencies, color="steelblue")
            ax_bar.set_xticks([])
            ax_bar.set_ylabel("Mean Frequency", fontsize=10)
            ax_bar.set_ylim(0, 1)

        ax_bar.grid(axis="y" if bar_position == "right" else "x", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    if ax_bar is None:
        plt.tight_layout()

    # Save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def get_stability_statistics(
    stability_matrix: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Extract stability metrics from feature selection matrix.

    Parameters
    ----------
    stability_matrix : np.ndarray
        Feature × fold binary/count matrix
    feature_names : List[str], optional
        Feature names (default: Feature 0, 1, ...)

    Returns
    -------
    Dict
        Statistics with keys:
        - "per_feature": Per-feature metrics
        - "global": Overall stability statistics
        - "rankings": Feature rankings by stability
        - "consistency_metrics": Consistency scores
    """
    n_features, n_folds = _validate_stability_matrix(stability_matrix)
    stability_matrix = np.asarray(stability_matrix)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    frequencies = np.mean(stability_matrix, axis=1)
    stds = np.std(stability_matrix, axis=1)

    stats = {
        "per_feature": {},
        "global": {
            "mean_frequency": float(np.mean(frequencies)),
            "std_frequency": float(np.std(frequencies)),
            "min_frequency": float(np.min(frequencies)),
            "max_frequency": float(np.max(frequencies)),
        },
        "rankings": {},
        "consistency_metrics": {},
    }

    # Per-feature stats
    for i, fname in enumerate(feature_names):
        n_appearances = int(np.sum(stability_matrix[i, :]))
        freq = frequencies[i]
        stability = 1.0 - (stds[i] / (freq + 1e-10))  # Stability as consistency

        stats["per_feature"][f"feature_{i}_{fname}"] = {
            "frequency": float(freq),
            "appearances": n_appearances,
            "consistency": float(stability),
            "std": float(stds[i]),
        }

    # Rankings by frequency
    sorted_indices = np.argsort(-frequencies)  # Descending
    stats["rankings"]["by_frequency"] = [
        {
            "rank": int(rank + 1),
            "feature": feature_names[idx],
            "frequency": float(frequencies[idx]),
            "appearances": int(np.sum(stability_matrix[idx, :])),
        }
        for rank, idx in enumerate(sorted_indices)
    ]

    # Consistency metrics
    stats["consistency_metrics"]["stable_features"] = [
        feature_names[i]
        for i in range(n_features)
        if frequencies[i] >= np.percentile(frequencies, 75)
    ]
    stats["consistency_metrics"]["unstable_features"] = [
        feature_names[i]
        for i in range(n_features)
        if frequencies[i] <= np.percentile(frequencies, 25)
    ]

    return stats
