"""
Coefficient heatmap visualization module.

Provides visualization of model coefficients as heatmaps with support for
feature×class matrices, diverging colormaps, and statistical annotations.

Main Functions:
    - plot_coefficients_heatmap: Visualize feature coefficients across classes
    - get_coefficient_statistics: Extract coefficient statistics and rankings
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def _normalize_coefficients(coefficients: np.ndarray, method: str = "standard") -> np.ndarray:
    """
    Normalize coefficients using specified method.

    Parameters
    ----------
    coefficients : np.ndarray
        Array of coefficients to normalize (features × classes)
    method : str, default="standard"
        Normalization method:
        - "standard": Per-feature z-score normalization
        - "minmax": Per-feature min-max scaling
        - "none": No normalization

    Returns
    -------
    np.ndarray
        Normalized coefficients
    """
    if method == "none" or coefficients.size == 0:
        return coefficients.copy()

    if method == "standard":
        # Per-feature z-score normalization
        means = np.mean(coefficients, axis=1, keepdims=True)
        stds = np.std(coefficients, axis=1, keepdims=True)
        # Avoid division by zero
        stds[stds == 0] = 1.0
        return (coefficients - means) / stds

    elif method == "minmax":
        # Per-feature min-max scaling to [-1, 1]
        mins = np.min(coefficients, axis=1, keepdims=True)
        maxs = np.max(coefficients, axis=1, keepdims=True)
        ranges = maxs - mins
        # Avoid division by zero
        ranges[ranges == 0] = 1.0
        return 2 * (coefficients - mins) / ranges - 1

    return coefficients.copy()


def _sort_features_by_magnitude(coefficients: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Sort features by coefficient magnitude.

    Parameters
    ----------
    coefficients : np.ndarray
        Feature × class coefficient matrix
    method : str
        "mean": Sort by absolute mean coefficient
        "max": Sort by max absolute coefficient
        "norm": Sort by L2 norm

    Returns
    -------
    np.ndarray
        Indices of sorted features (ascending magnitude)
    """
    if method == "mean":
        magnitudes = np.mean(np.abs(coefficients), axis=1)
    elif method == "max":
        magnitudes = np.max(np.abs(coefficients), axis=1)
    elif method == "norm":
        magnitudes = np.linalg.norm(coefficients, axis=1)
    else:
        return np.arange(len(coefficients))

    return np.argsort(magnitudes)


def _format_coefficient_annotation(value: float, decimals: int = 2) -> str:
    """
    Format coefficient value for annotation.

    Parameters
    ----------
    value : float
        Coefficient value
    decimals : int
        Number of decimal places

    Returns
    -------
    str
        Formatted annotation text
    """
    if value >= 0:
        return f"+{value:.{decimals}f}"
    else:
        return f"{value:.{decimals}f}"


def plot_coefficients_heatmap(
    coefficients: np.ndarray,
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    normalize: Union[bool, str] = "standard",
    sort_features: Union[bool, str] = "mean",
    colormap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: float = 0.0,
    show_values: bool = True,
    value_decimals: int = 2,
    cbar_label: str = "Coefficient Value",
    title: str = "Feature Coefficients Heatmap",
    figure_size: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize model coefficients as a heatmap.

    Creates a features × classes heatmap with diverging colormap centering
    on zero. Supports normalization, feature sorting, and value annotations.

    Parameters
    ----------
    coefficients : np.ndarray
        Feature × class coefficient matrix (n_features, n_classes)
    class_names : List[str], optional
        Names for classes (columns). If None, uses Class 0, 1, ...
    feature_names : List[str], optional
        Names for features (rows). If None, uses Feature 0, 1, ...
    normalize : bool or str, default="standard"
        Normalization method:
        - False/"none": No normalization
        - True/"standard": Z-score per-feature (mean=0, std=1)
        - "minmax": Min-max scaling per-feature to [-1, 1]
    sort_features : bool or str, default="mean"
        Sort features by magnitude:
        - False/"none": No sorting
        - True/"mean": Sort by mean absolute coefficient
        - "max": Sort by max absolute coefficient
        - "norm": Sort by L2 norm
    colormap : str, default="RdBu_r"
        Diverging colormap name (RdBu_r, RdYlBu_r, coolwarm, etc.)
    vmin : float, optional
        Minimum colormap value. If None, auto-computed.
    vmax : float, optional
        Maximum colormap value. If None, auto-computed.
    center : float, default=0.0
        Center point for diverging colormap
    show_values : bool, default=True
        Annotate heatmap cells with coefficient values
    value_decimals : int, default=2
        Decimal places in value annotations
    cbar_label : str
        Label for colorbar
    title : str
        Figure title
    figure_size : Tuple[int, int]
        Figure dimensions (width, height)
    save_path : Path, optional
        Path to save figure as PNG
    dpi : int
        DPI for saved figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object

    Raises
    ------
    ValueError
        If coefficients is empty or has invalid shape
    """
    # Validate input
    coefficients = np.asarray(coefficients)
    if coefficients.size == 0:
        raise ValueError("Coefficients array cannot be empty")
    if coefficients.ndim != 2:
        raise ValueError(f"Coefficients must be 2D array, got shape {coefficients.shape}")

    n_features, n_classes = coefficients.shape

    # Generate default names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    # Validate name lengths
    if len(class_names) != n_classes:
        raise ValueError(f"class_names length {len(class_names)} does not match coefficients shape {n_classes}")
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names length {len(feature_names)} does not match coefficients shape {n_features}")

    # Normalize if requested
    if normalize is True:
        normalize = "standard"
    elif normalize is False:
        normalize = "none"
    data_to_plot = _normalize_coefficients(coefficients, method=normalize)

    # Sort features if requested
    feature_order = np.arange(n_features)
    if sort_features is True:
        sort_features = "mean"
    if sort_features not in (False, "none"):
        feature_order = _sort_features_by_magnitude(data_to_plot, method=sort_features)
        data_to_plot = data_to_plot[feature_order]
        feature_names = [feature_names[i] for i in feature_order]

    # Determine color scale limits
    if vmin is None or vmax is None:
        abs_max = np.max(np.abs(data_to_plot))
        if vmin is None:
            vmin = -abs_max
        if vmax is None:
            vmax = abs_max

    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Create heatmap using imshow
    im = ax.imshow(
        data_to_plot,
        cmap=colormap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    # Set axis ticks and labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names, fontsize=9)

    # Add grid
    ax.set_xticks(np.arange(n_classes) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(feature_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    # Add value annotations if requested
    if show_values:
        for i in range(len(feature_names)):
            for j in range(n_classes):
                value = data_to_plot[i, j]
                text_color = "white" if np.abs(value) > abs_max / 2 else "black"
                annotation = _format_coefficient_annotation(value, value_decimals)
                ax.text(
                    j,
                    i,
                    annotation,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    # Labels and title
    ax.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def get_coefficient_statistics(
    coefficients: np.ndarray,
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Extract statistics from coefficient matrix.

    Parameters
    ----------
    coefficients : np.ndarray
        Feature × class coefficient matrix
    class_names : List[str], optional
        Class names (default: Class 0, 1, ...)
    feature_names : List[str], optional
        Feature names (default: Feature 0, 1, ...)

    Returns
    -------
    Dict
        Statistics dict with keys:
        - "per_feature": Dict per-feature stats (mean, std, min, max, norm)
        - "per_class": Dict per-class stats (mean, std, magnitude)
        - "global": Global statistics (mean, std, min, max)
        - "rankings": Feature rankings by magnitude
    """
    coefficients = np.asarray(coefficients)
    n_features, n_classes = coefficients.shape

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    stats = {
        "per_feature": {},
        "per_class": {},
        "global": {
            "mean": float(np.mean(coefficients)),
            "std": float(np.std(coefficients)),
            "min": float(np.min(coefficients)),
            "max": float(np.max(coefficients)),
            "abs_mean": float(np.mean(np.abs(coefficients))),
            "abs_max": float(np.max(np.abs(coefficients))),
        },
        "rankings": {},
    }

    # Per-feature statistics
    for i, fname in enumerate(feature_names):
        feat_coefs = coefficients[i, :]
        stats["per_feature"][f"feature_{i}_{fname}"] = {
            "mean": float(np.mean(feat_coefs)),
            "std": float(np.std(feat_coefs)),
            "min": float(np.min(feat_coefs)),
            "max": float(np.max(feat_coefs)),
            "norm": float(np.linalg.norm(feat_coefs)),
            "abs_mean": float(np.mean(np.abs(feat_coefs))),
        }

    # Per-class statistics
    for j, cname in enumerate(class_names):
        class_coefs = coefficients[:, j]
        stats["per_class"][f"class_{j}_{cname}"] = {
            "mean": float(np.mean(class_coefs)),
            "std": float(np.std(class_coefs)),
            "magnitude": float(np.linalg.norm(class_coefs)),
        }

    # Feature rankings by absolute mean coefficient
    magnitudes = np.mean(np.abs(coefficients), axis=1)
    sorted_indices = np.argsort(-magnitudes)  # Descending
    stats["rankings"]["by_mean_magnitude"] = [
        {"rank": int(rank + 1), "feature": feature_names[idx], "magnitude": float(magnitudes[idx])}
        for rank, idx in enumerate(sorted_indices)
    ]

    return stats
