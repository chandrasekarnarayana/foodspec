"""
Uncertainty quantification and conformal prediction visualization module.

Provides visualization tools for analyzing model confidence, conformal prediction
set sizes, coverage-efficiency trade-offs, and abstention patterns.

Main Functions:
    - plot_confidence_map: Visualize max prediction probability per sample
    - plot_set_size_distribution: Analyze conformal prediction set sizes
    - plot_coverage_efficiency: Coverage vs efficiency trade-off curves
    - plot_abstention_distribution: Abstention rates by class/batch
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _validate_confidence_array(confidences: np.ndarray) -> int:
    """Validate confidence array and return length."""
    confidences = np.asarray(confidences)
    if confidences.ndim != 1:
        raise ValueError(f"Confidences must be 1D array, got shape {confidences.shape}")
    if confidences.size == 0:
        raise ValueError("Confidences array cannot be empty")
    return len(confidences)


def _normalize_confidences(confidences: np.ndarray) -> np.ndarray:
    """Ensure confidences are in [0, 1]."""
    confidences = np.asarray(confidences, dtype=float)
    if np.any(confidences < 0) or np.any(confidences > 1):
        # Clip to [0, 1]
        confidences = np.clip(confidences, 0, 1)
    return confidences


def _sort_by_confidence(
    confidences: np.ndarray, descending: bool = True
) -> np.ndarray:
    """Return indices sorted by confidence."""
    if descending:
        return np.argsort(-confidences)  # Descending
    else:
        return np.argsort(confidences)  # Ascending


def _get_confidence_class(confidence: float, thresholds: List[float]) -> str:
    """Classify confidence into category based on thresholds."""
    thresholds = sorted(thresholds)
    for i, threshold in enumerate(thresholds):
        if confidence < threshold:
            return f"[{i}]"
    return f"[{len(thresholds)}]"


def plot_confidence_map(
    confidences: np.ndarray,
    class_predictions: Optional[np.ndarray] = None,
    sample_labels: Optional[List[str]] = None,
    sort_by_confidence: bool = True,
    confidence_thresholds: Optional[List[float]] = None,
    colormap: str = "RdYlGn",
    show_values: bool = True,
    value_decimals: int = 2,
    title: str = "Sample Confidence Map",
    figure_size: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize maximum prediction confidence per sample.

    Creates a horizontal bar plot showing prediction confidence for each sample,
    optionally colored by predicted class or confidence category.

    Parameters
    ----------
    confidences : np.ndarray
        Model prediction confidences [0, 1] for each sample (n_samples,)
    class_predictions : np.ndarray, optional
        Predicted class for each sample. If provided, bars colored by class.
        If None, colored by confidence level.
    sample_labels : List[str], optional
        Labels for samples (e.g., "Sample 0", "Sample 1"). If None, uses index.
    sort_by_confidence : bool, default=True
        Sort samples by confidence (ascending for visual hierarchy)
    confidence_thresholds : List[float], optional
        Thresholds for confidence categorization [0.5, 0.7, 0.9].
        If None, uses [0.5, 0.7, 0.9].
    colormap : str, default="RdYlGn"
        Colormap for confidence-based coloring
    show_values : bool, default=True
        Annotate bars with confidence values
    value_decimals : int, default=2
        Decimal places in value annotations
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
        If confidences is empty or misshapen
    """
    # Validate input
    n_samples = _validate_confidence_array(confidences)
    confidences = _normalize_confidences(confidences)

    # Default thresholds
    if confidence_thresholds is None:
        confidence_thresholds = [0.5, 0.7, 0.9]

    # Generate sample labels if not provided
    if sample_labels is None:
        sample_labels = [f"Sample {i}" for i in range(n_samples)]
    elif len(sample_labels) != n_samples:
        raise ValueError(
            f"sample_labels length {len(sample_labels)} "
            f"does not match confidences length {n_samples}"
        )

    # Validate class predictions
    if class_predictions is not None:
        class_predictions = np.asarray(class_predictions)
        if len(class_predictions) != n_samples:
            raise ValueError(
                f"class_predictions length {len(class_predictions)} "
                f"does not match confidences length {n_samples}"
            )

    # Sort if requested
    indices = _sort_by_confidence(confidences, descending=False)
    confidences_sorted = confidences[indices]
    sample_labels = [sample_labels[i] for i in indices]
    class_pred_sorted = (
        class_predictions[indices] if class_predictions is not None else None
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Determine bar colors
    if class_pred_sorted is not None:
        # Color by predicted class
        n_classes = int(np.max(class_pred_sorted)) + 1
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_classes, 10)))
        bar_colors = [colors[int(c)] for c in class_pred_sorted]
    else:
        # Color by confidence level
        cmap = plt.colormaps[colormap]
        bar_colors = [cmap(c) for c in confidences_sorted]

    # Create horizontal bar plot
    bars = ax.barh(range(len(sample_labels)), confidences_sorted, color=bar_colors, edgecolor="black", linewidth=0.5)

    # Add value annotations
    if show_values:
        for i, (conf, bar) in enumerate(zip(confidences_sorted, bars)):
            annotation = f"{conf:.{value_decimals}f}"
            ax.text(
                conf + 0.01,
                i,
                annotation,
                va="center",
                fontsize=8,
            )

    # Set y-axis labels and limits
    ax.set_yticks(range(len(sample_labels)))
    ax.set_yticklabels(sample_labels, fontsize=9)
    ax.set_xlabel("Confidence", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.1 if show_values else 1.0)

    # Add threshold lines
    for threshold in confidence_thresholds:
        ax.axvline(threshold, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Title and grid
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_set_size_distribution(
    set_sizes: np.ndarray,
    batch_labels: Optional[np.ndarray] = None,
    stage_labels: Optional[np.ndarray] = None,
    show_violin: bool = True,
    show_box: bool = True,
    colormap: str = "Set2",
    title: str = "Conformal Set Size Distribution",
    figure_size: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize distribution of conformal prediction set sizes.

    Creates histogram and optional violin/box plots, with faceting by batch or stage.

    Parameters
    ----------
    set_sizes : np.ndarray
        Conformal set sizes for each sample (n_samples,)
    batch_labels : np.ndarray, optional
        Batch identifier for each sample for faceting
    stage_labels : np.ndarray, optional
        Stage/phase identifier for each sample for faceting
    show_violin : bool, default=True
        Include violin plots
    show_box : bool, default=True
        Include box plots over violins
    colormap : str, default="Set2"
        Colormap for batch/stage colors
    title : str
        Figure title
    figure_size : Tuple[int, int]
        Figure dimensions
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
        If set_sizes is empty or misshapen
    """
    # Validate input
    set_sizes = np.asarray(set_sizes, dtype=int)
    if set_sizes.ndim != 1:
        raise ValueError(f"set_sizes must be 1D, got shape {set_sizes.shape}")
    if set_sizes.size == 0:
        raise ValueError("set_sizes array cannot be empty")

    # Validate optional labels
    if batch_labels is not None:
        batch_labels = np.asarray(batch_labels)
        if len(batch_labels) != len(set_sizes):
            raise ValueError("batch_labels length mismatch")
    if stage_labels is not None:
        stage_labels = np.asarray(stage_labels)
        if len(stage_labels) != len(set_sizes):
            raise ValueError("stage_labels length mismatch")

    # Create figure
    if batch_labels is not None or stage_labels is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    else:
        fig, ax1 = plt.subplots(figsize=figure_size)
        ax2 = None

    # Main histogram
    n_bins = min(20, int(np.max(set_sizes)) + 1)
    ax1.hist(set_sizes, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Set Size", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title("Overall Distribution", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add statistics
    mean_size = np.mean(set_sizes)
    median_size = np.median(set_sizes)
    ax1.axvline(mean_size, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_size:.2f}")
    ax1.axvline(median_size, color="green", linestyle="--", linewidth=2, label=f"Median: {median_size:.1f}")
    ax1.legend()

    # Faceted plot by batch or stage
    if ax2 is not None:
        if batch_labels is not None:
            labels = batch_labels
            label_name = "Batch"
        else:
            labels = stage_labels
            label_name = "Stage"

        unique_labels = np.unique(labels)
        cmap = plt.colormaps[colormap]
        colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]

        # Create violin plot
        data_by_label = [set_sizes[labels == label] for label in unique_labels]

        if show_violin:
            ax2.violinplot(
                data_by_label,
                positions=range(len(unique_labels)),
                showmeans=True,
                showmedians=True,
            )

        if show_box:
            bp = ax2.boxplot(
                data_by_label,
                positions=range(len(unique_labels)),
                widths=0.3,
                patch_artist=True,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

        ax2.set_xticks(range(len(unique_labels)))
        ax2.set_xticklabels(unique_labels, rotation=45)
        ax2.set_xlabel(label_name, fontsize=12, fontweight="bold")
        ax2.set_ylabel("Set Size", fontsize=12, fontweight="bold")
        ax2.set_title(f"Distribution by {label_name}", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_coverage_efficiency(
    alphas: np.ndarray,
    coverages: np.ndarray,
    avg_set_sizes: np.ndarray,
    target_coverage: float = 0.9,
    marker_size: int = 100,
    colormap: str = "viridis",
    title: str = "Coverage vs Efficiency Trade-off",
    figure_size: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize coverage vs efficiency trade-off for conformal prediction.

    Plots average set size (efficiency) vs coverage (validity) with alpha values
    as parameters.

    Parameters
    ----------
    alphas : np.ndarray
        Significance levels (1 - confidence) tested
    coverages : np.ndarray
        Achieved coverage for each alpha
    avg_set_sizes : np.ndarray
        Average conformal set size for each alpha
    target_coverage : float, default=0.9
        Target coverage level to highlight
    marker_size : int, default=100
        Marker size for scatter points
    colormap : str, default="viridis"
        Colormap for alpha values
    title : str
        Figure title
    figure_size : Tuple[int, int]
        Figure dimensions
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
        If inputs are misaligned or empty
    """
    # Validate inputs
    alphas = np.asarray(alphas)
    coverages = np.asarray(coverages)
    avg_set_sizes = np.asarray(avg_set_sizes)

    if not (len(alphas) == len(coverages) == len(avg_set_sizes)):
        raise ValueError("Input arrays must have matching lengths")
    if len(alphas) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Scatter plot with alpha as color
    cmap = plt.colormaps[colormap]
    scatter = ax.scatter(
        avg_set_sizes,
        coverages,
        c=alphas,
        cmap=cmap,
        s=marker_size,
        alpha=0.7,
        edgecolors="black",
        linewidth=1,
    )

    # Add alpha labels to points
    for i, alpha in enumerate(alphas):
        ax.annotate(
            f"α={alpha:.3f}",
            (avg_set_sizes[i], coverages[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Add target coverage line
    ax.axhline(target_coverage, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"Target: {target_coverage:.1%}")

    # Add perfect coverage line
    ax.axhline(1.0, color="green", linestyle=":", linewidth=1.5, alpha=0.5, label="Perfect (100%)")

    # Labels and limits
    ax.set_xlabel("Average Set Size (Efficiency)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coverage", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, np.max(avg_set_sizes) * 1.1)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("α (Significance Level)", rotation=270, labelpad=20)

    # Title, grid, legend
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_abstention_distribution(
    abstain_flags: np.ndarray,
    class_labels: Optional[np.ndarray] = None,
    batch_labels: Optional[np.ndarray] = None,
    show_table: bool = True,
    colormap: str = "Set2",
    title: str = "Abstention Distribution",
    figure_size: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize abstention rates across classes or batches.

    Creates stacked bar chart showing abstention rate, with optional summary table.

    Parameters
    ----------
    abstain_flags : np.ndarray
        Binary array (1=abstain, 0=predict) for each sample
    class_labels : np.ndarray, optional
        Class label for each sample. If provided, facet by class.
    batch_labels : np.ndarray, optional
        Batch identifier for each sample. If provided, facet by batch.
    show_table : bool, default=True
        Show summary statistics table
    colormap : str, default="Set2"
        Colormap for bar colors
    title : str
        Figure title
    figure_size : Tuple[int, int]
        Figure dimensions
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
        If abstain_flags is empty or invalid
    """
    # Validate input
    abstain_flags = np.asarray(abstain_flags, dtype=int)
    if abstain_flags.ndim != 1:
        raise ValueError(f"abstain_flags must be 1D, got shape {abstain_flags.shape}")
    if abstain_flags.size == 0:
        raise ValueError("abstain_flags array cannot be empty")

    n_samples = len(abstain_flags)
    overall_abstain_rate = np.mean(abstain_flags)

    # Determine faceting
    facet_by = None
    facet_labels = None
    if class_labels is not None:
        class_labels = np.asarray(class_labels)
        if len(class_labels) != n_samples:
            raise ValueError("class_labels length mismatch")
        facet_by = "Class"
        facet_labels = class_labels
    elif batch_labels is not None:
        batch_labels = np.asarray(batch_labels)
        if len(batch_labels) != n_samples:
            raise ValueError("batch_labels length mismatch")
        facet_by = "Batch"
        facet_labels = batch_labels

    # Create figure
    if facet_by is not None:
        if show_table:
            fig = plt.figure(figsize=figure_size)
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
            ax = fig.add_subplot(gs[0])
            ax_table = fig.add_subplot(gs[1])
        else:
            fig, ax = plt.subplots(figsize=figure_size)
            ax_table = None

        # Compute abstain rates by facet
        unique_facets = np.unique(facet_labels)
        abstain_rates = []
        predict_rates = []
        counts = []

        for facet in unique_facets:
            mask = facet_labels == facet
            count = np.sum(mask)
            abstain_count = np.sum(abstain_flags[mask])
            abstain_rate = abstain_count / count if count > 0 else 0
            abstain_rates.append(abstain_rate)
            predict_rates.append(1 - abstain_rate)
            counts.append(count)

        # Stacked bar chart
        x_pos = np.arange(len(unique_facets))
        cmap = plt.colormaps[colormap]

        ax.bar(x_pos, predict_rates, label="Predicted", color=cmap(0), edgecolor="black")
        ax.bar(
            x_pos,
            abstain_rates,
            bottom=predict_rates,
            label="Abstained",
            color=cmap(1),
            edgecolor="black",
        )

        # Value annotations
        for i, (predict_rate, abstain_rate) in enumerate(zip(predict_rates, abstain_rates)):
            ax.text(i, predict_rate / 2, f"{predict_rate:.1%}", ha="center", va="center", fontweight="bold")
            ax.text(i, predict_rate + abstain_rate / 2, f"{abstain_rate:.1%}", ha="center", va="center", fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(unique_facets)
        ax.set_ylabel("Rate", fontsize=12, fontweight="bold")
        ax.set_xlabel(facet_by, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        # Summary table
        if ax_table is not None:
            table_data = [
                [str(f) for f in unique_facets],
                [f"{rate:.1%}" for rate in abstain_rates],
                [str(int(c)) for c in counts],
            ]
            table = ax_table.table(
                cellText=table_data,
                rowLabels=[facet_by, "Abstain Rate", "Count"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            ax_table.axis("off")

    else:
        # Overall rate only
        fig, ax = plt.subplots(figsize=figure_size)

        bars = ax.bar(
            ["Predicted", "Abstained"],
            [1 - overall_abstain_rate, overall_abstain_rate],
            color=[plt.cm.Set2(0), plt.cm.Set2(1)],
            edgecolor="black",
            width=0.5,
        )

        # Value annotations
        for bar, rate in zip(bars, [1 - overall_abstain_rate, overall_abstain_rate]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{rate:.1%}",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
            )

        ax.set_ylabel("Rate", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def get_uncertainty_statistics(
    confidences: np.ndarray,
    set_sizes: Optional[np.ndarray] = None,
    abstain_flags: Optional[np.ndarray] = None,
) -> Dict:
    """
    Extract uncertainty quantification statistics.

    Parameters
    ----------
    confidences : np.ndarray
        Prediction confidences [0, 1]
    set_sizes : np.ndarray, optional
        Conformal set sizes
    abstain_flags : np.ndarray, optional
        Binary abstention indicators

    Returns
    -------
    Dict
        Statistics dictionary with confidence, set size, and abstention metrics
    """
    stats = {
        "confidence": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences)),
            "q25": float(np.percentile(confidences, 25)),
            "q75": float(np.percentile(confidences, 75)),
        }
    }

    if set_sizes is not None:
        set_sizes = np.asarray(set_sizes)
        stats["set_size"] = {
            "mean": float(np.mean(set_sizes)),
            "median": float(np.median(set_sizes)),
            "min": int(np.min(set_sizes)),
            "max": int(np.max(set_sizes)),
            "std": float(np.std(set_sizes)),
        }

    if abstain_flags is not None:
        abstain_flags = np.asarray(abstain_flags, dtype=int)
        stats["abstention"] = {
            "rate": float(np.mean(abstain_flags)),
            "count": int(np.sum(abstain_flags)),
            "n_samples": int(len(abstain_flags)),
        }

    return stats
