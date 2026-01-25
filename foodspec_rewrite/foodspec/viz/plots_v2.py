"""
FoodSpec visualization utilities with reproducibility, artifact saving, and publication-quality exports.

Key principles:
- All plots are deterministic (seeded where randomness exists)
- All plots auto-save via ArtifactRegistry
- All plots are headless/batch-compatible (Agg backend)
- All plots export at ≥300 dpi for publication
- All plots support metadata-based grouping (batch/stage/instrument)
- All plots return Figure objects
- All plots include standardized titles, subtitles (protocol hash, run_id), legends
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from foodspec.core.artifacts import ArtifactRegistry


@dataclass
class PlotConfig:
    """Standard configuration for all plots."""

    dpi: int = 300
    figure_size: Tuple[float, float] = (10, 6)
    font_size: int = 10
    seed: Optional[int] = None

    def __post_init__(self):
        if self.dpi < 300:
            raise ValueError(f"dpi must be ≥ 300 for publication, got {self.dpi}")


def _init_plot(
    config: PlotConfig,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Any]:
    """Initialize figure with reproducibility and standard config."""
    if config.seed is not None:
        np.random.seed(config.seed)
    
    size = figsize or config.figure_size
    fig, ax = plt.subplots(figsize=size)
    plt.rcParams['font.size'] = config.font_size
    return fig, ax


def _add_metadata_subtitle(
    ax: Any,
    protocol_hash: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Add standardized subtitle with protocol hash and run ID."""
    subtitle = []
    if protocol_hash:
        subtitle.append(f"Protocol: {protocol_hash[:8]}")
    if run_id:
        subtitle.append(f"Run: {run_id[:8]}")
    
    if subtitle:
        ax.text(
            0.5,
            1.02,
            " | ".join(subtitle),
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )


def _save_and_close(
    fig: Figure,
    artifacts: ArtifactRegistry,
    filename: str,
    dpi: int = 300,
) -> Path:
    """Save figure at high DPI and close."""
    artifacts.ensure_layout()
    path = artifacts.plots_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


# ============================================================================
# Model Evaluation Plots
# ============================================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    artifacts: Optional[ArtifactRegistry] = None,
    filename: str = "confusion_matrix.png",
    protocol_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Figure:
    """Plot confusion matrix with standardized formatting.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list of str, optional
        Class names for tick labels
    artifacts : ArtifactRegistry, optional
        Registry for auto-saving
    filename : str
        Output filename if saving
    protocol_hash : str, optional
        Protocol hash for subtitle
    run_id : str, optional
        Run ID for subtitle
    config : PlotConfig, optional
        Plot configuration (uses defaults if None)
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    config = config or PlotConfig()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    fig, ax = _init_plot(config)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", 
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    
    _add_metadata_subtitle(ax, protocol_hash, run_id)
    fig.colorbar(im, ax=ax, label="Count")
    
    if artifacts:
        _save_and_close(fig, artifacts, filename, dpi=config.dpi)
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
    artifacts: Optional[ArtifactRegistry] = None,
    filename: str = "calibration_curve.png",
    protocol_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Figure:
    """Plot calibration curve with perfect calibration reference.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    proba : np.ndarray
        Predicted probabilities (n_samples, 2) for binary or (n_samples,) for binary probabilities
    n_bins : int
        Number of confidence bins
    artifacts : ArtifactRegistry, optional
        Registry for auto-saving
    filename : str
        Output filename
    protocol_hash : str, optional
        Protocol hash for subtitle
    run_id : str, optional
        Run ID for subtitle
    config : PlotConfig, optional
        Plot configuration
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    config = config or PlotConfig()
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    
    # Extract binary probabilities
    if proba.ndim == 2:
        proba_binary = proba[:, 1]
    else:
        proba_binary = proba
    
    fig, ax = _init_plot(config)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect Calibration")
    
    # Compute calibration curve
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for i in range(n_bins):
        in_bin = (proba_binary >= bin_edges[i]) & (proba_binary < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (proba_binary >= bin_edges[i]) & (proba_binary <= bin_edges[i + 1])
        
        if in_bin.sum() > 0:
            bin_sums[i] = proba_binary[in_bin].sum()
            bin_true[i] = y_true[in_bin].sum()
            bin_total[i] = in_bin.sum()
    
    # Plot calibration points
    bin_accs = np.divide(bin_true, bin_total, where=bin_total > 0, 
                         out=np.zeros_like(bin_total, dtype=float))
    valid = bin_total > 0
    ax.plot(bin_centers[valid], bin_accs[valid], "o-", label="Model", linewidth=2, markersize=8)
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Probability (Accuracy)")
    ax.set_title("Calibration Curve", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    
    _add_metadata_subtitle(ax, protocol_hash, run_id)
    
    if artifacts:
        _save_and_close(fig, artifacts, filename, dpi=config.dpi)
    
    return fig


def plot_metrics_by_fold(
    fold_metrics: List[Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    artifacts: Optional[ArtifactRegistry] = None,
    filename: str = "metrics_by_fold.png",
    protocol_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Figure:
    """Plot metric values across CV folds.
    
    Parameters
    ----------
    fold_metrics : list of dict
        Metrics per fold, e.g., [{'fold_id': 0, 'accuracy': 0.8, 'f1': 0.75}, ...]
    metric_names : list of str, optional
        Metrics to plot (excludes 'fold_id'). If None, plots all numeric columns.
    artifacts : ArtifactRegistry, optional
        Registry for auto-saving
    filename : str
        Output filename
    protocol_hash : str, optional
        Protocol hash for subtitle
    run_id : str, optional
        Run ID for subtitle
    config : PlotConfig, optional
        Plot configuration
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    config = config or PlotConfig()
    df = pd.DataFrame(fold_metrics)
    
    if metric_names is None:
        metric_names = [c for c in df.columns if c not in {'fold_id'} and df[c].dtype in [np.float32, np.float64]]
    
    fig, ax = _init_plot(config, figsize=(max(10, len(df) + 2), 6))
    
    x = np.arange(len(df))
    width = 0.8 / len(metric_names)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metric_names)))
    
    for i, metric in enumerate(metric_names):
        offset = (i - len(metric_names) / 2 + 0.5) * width
        ax.bar(x + offset, df[metric], width, label=metric, color=colors[i])
    
    ax.set_xlabel("Fold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics by Fold", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in range(len(df))])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    _add_metadata_subtitle(ax, protocol_hash, run_id)
    
    if artifacts:
        _save_and_close(fig, artifacts, filename, dpi=config.dpi)
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 15,
    artifacts: Optional[ArtifactRegistry] = None,
    filename: str = "feature_importance.png",
    protocol_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Figure:
    """Plot top-k feature importance (coefficients or permutation).
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance table with 'feature' and 'importance_mean' (or 'coefficient') columns
    top_k : int
        Number of top features to display
    artifacts : ArtifactRegistry, optional
        Registry for auto-saving
    filename : str
        Output filename
    protocol_hash : str, optional
        Protocol hash for subtitle
    run_id : str, optional
        Run ID for subtitle
    config : PlotConfig, optional
        Plot configuration
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    config = config or PlotConfig()
    df = importance_df.copy().head(top_k)
    
    # Determine importance column
    if 'importance_mean' in df.columns:
        importance_col = 'importance_mean'
    elif 'coefficient' in df.columns:
        importance_col = 'coefficient'
    elif 'abs_coefficient' in df.columns:
        importance_col = 'abs_coefficient'
    else:
        raise ValueError("No importance/coefficient column found")
    
    fig, ax = _init_plot(config, figsize=(10, max(6, top_k * 0.3)))
    
    y_pos = np.arange(len(df))
    colors = plt.cm.RdYlBu_r(df[importance_col] / df[importance_col].max())
    
    ax.barh(y_pos, df[importance_col], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importance", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    _add_metadata_subtitle(ax, protocol_hash, run_id)
    
    if artifacts:
        _save_and_close(fig, artifacts, filename, dpi=config.dpi)
    
    return fig


# ============================================================================
# Trust & Uncertainty Plots
# ============================================================================


def plot_conformal_coverage_by_group(
    coverage_table: pd.DataFrame,
    target_coverage: float = 0.9,
    artifacts: Optional[ArtifactRegistry] = None,
    filename: str = "conformal_coverage.png",
    protocol_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Figure:
    """Plot conformal coverage per group with target line.
    
    Parameters
    ----------
    coverage_table : pd.DataFrame
        Coverage aggregated by group, with columns: group, coverage, n_samples, avg_set_size
    target_coverage : float
        Target coverage level (e.g., 0.9 for 90%)
    artifacts : ArtifactRegistry, optional
        Registry for auto-saving
    filename : str
        Output filename
    protocol_hash : str, optional
        Protocol hash for subtitle
    run_id : str, optional
        Run ID for subtitle
    config : PlotConfig, optional
        Plot configuration
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    config = config or PlotConfig()
    df = coverage_table.copy()
    
    fig, ax = _init_plot(config)
    
    x = np.arange(len(df))
    colors = ['green' if c >= target_coverage else 'red' for c in df['coverage']]
    
    bars = ax.bar(x, df['coverage'], color=colors, alpha=0.7)
    ax.axhline(target_coverage, color='blue', linestyle='--', linewidth=2, label=f'Target ({target_coverage:.1%})')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['group'], rotation=45, ha='right')
    ax.set_ylabel("Coverage")
    ax.set_title("Conformal Coverage by Group", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    _add_metadata_subtitle(ax, protocol_hash, run_id)
    
    if artifacts:
        _save_and_close(fig, artifacts, filename, dpi=config.dpi)
    
    return fig


def plot_prediction_set_sizes(
    set_sizes: np.ndarray,
    grouping: Optional[np.ndarray] = None,
    group_names: Optional[Dict[Any, str]] = None,
    artifacts: Optional[ArtifactRegistry] = None,
    filename: str = "set_sizes.png",
    protocol_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Figure:
    """Plot distribution of conformal prediction set sizes.
    
    Parameters
    ----------
    set_sizes : np.ndarray
        Prediction set sizes (n_samples,)
    grouping : np.ndarray, optional
        Group/batch identifiers for coloring
    group_names : dict, optional
        Mapping of group ID to display name
    artifacts : ArtifactRegistry, optional
        Registry for auto-saving
    filename : str
        Output filename
    protocol_hash : str, optional
        Protocol hash for subtitle
    run_id : str, optional
        Run ID for subtitle
    config : PlotConfig, optional
        Plot configuration
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    config = config or PlotConfig()
    set_sizes = np.asarray(set_sizes)
    
    fig, ax = _init_plot(config)
    
    if grouping is None:
        ax.hist(set_sizes, bins=int(np.sqrt(len(set_sizes))), color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Prediction Set Size")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Prediction Set Sizes", fontsize=12, fontweight="bold")
    else:
        grouping = np.asarray(grouping)
        groups = np.unique(grouping)
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        
        for i, group_id in enumerate(groups):
            mask = grouping == group_id
            label = group_names[group_id] if group_names and group_id in group_names else str(group_id)
            ax.hist(set_sizes[mask], bins=int(np.sqrt(mask.sum())), alpha=0.6, label=label, color=colors[i])
        
        ax.set_xlabel("Prediction Set Size")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Prediction Set Sizes by Group", fontsize=12, fontweight="bold")
        ax.legend()
    
    ax.grid(axis="y", alpha=0.3)
    _add_metadata_subtitle(ax, protocol_hash, run_id)
    
    if artifacts:
        _save_and_close(fig, artifacts, filename, dpi=config.dpi)
    
    return fig


def plot_abstention_rate(
    abstention_rates: Dict[str, float],
    artifacts: Optional[ArtifactRegistry] = None,
    filename: str = "abstention_rate.png",
    protocol_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Figure:
    """Plot abstention rates per fold/group.
    
    Parameters
    ----------
    abstention_rates : dict
        Mapping of fold/group name to abstention rate
    artifacts : ArtifactRegistry, optional
        Registry for auto-saving
    filename : str
        Output filename
    protocol_hash : str, optional
        Protocol hash for subtitle
    run_id : str, optional
        Run ID for subtitle
    config : PlotConfig, optional
        Plot configuration
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    config = config or PlotConfig()
    
    fig, ax = _init_plot(config)
    
    names = list(abstention_rates.keys())
    rates = list(abstention_rates.values())
    
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(rates)))
    bars = ax.bar(range(len(names)), rates, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Abstention Rate")
    ax.set_title("Abstention Rate by Fold/Group", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    _add_metadata_subtitle(ax, protocol_hash, run_id)
    
    if artifacts:
        _save_and_close(fig, artifacts, filename, dpi=config.dpi)
    
    return fig


__all__ = [
    "PlotConfig",
    "plot_confusion_matrix",
    "plot_calibration_curve",
    "plot_metrics_by_fold",
    "plot_feature_importance",
    "plot_conformal_coverage_by_group",
    "plot_prediction_set_sizes",
    "plot_abstention_rate",
]
