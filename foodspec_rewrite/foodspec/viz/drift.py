"""
Batch drift and stage difference visualizations for spectral data.

This module provides tools for analyzing and visualizing:
- Batch-to-batch variations in spectral data
- Differences between processing stages
- Drift detection with confidence bands
- Replicate similarity analysis
- Temporal drift trends

Functions:
    plot_batch_drift: Visualize spectral drift across batches
    plot_stage_differences: Visualize differences between processing stages
    plot_replicate_similarity: Visualize similarity between replicates
    plot_temporal_drift: Visualize temporal drift trends
    get_batch_statistics: Extract batch drift statistics
    get_stage_statistics: Extract stage difference statistics
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from datetime import datetime


def _compute_batch_statistics(
    spectra: np.ndarray, batch_labels: np.ndarray, confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Compute mean and confidence intervals per batch.
    
    Args:
        spectra: Spectral data (n_samples, n_features)
        batch_labels: Batch identifiers (n_samples,)
        confidence: Confidence level for bands (default 0.95)
    
    Returns:
        Dictionary with batch statistics
    """
    unique_batches = np.unique(batch_labels)
    stats = {}
    
    for batch in unique_batches:
        mask = batch_labels == batch
        batch_spectra = spectra[mask]
        
        mean_spectrum = np.mean(batch_spectra, axis=0)
        std_spectrum = np.std(batch_spectra, axis=0)
        n_samples = batch_spectra.shape[0]
        
        # Compute confidence interval using t-distribution approximation
        # For large samples, this approaches z-distribution
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        ci_margin = z_score * std_spectrum / np.sqrt(n_samples)
        
        stats[str(batch)] = {
            "mean": mean_spectrum,
            "std": std_spectrum,
            "ci_lower": mean_spectrum - ci_margin,
            "ci_upper": mean_spectrum + ci_margin,
            "n_samples": n_samples,
        }
    
    return stats


def _compute_difference_from_reference(
    batch_stats: Dict[str, Any], reference_batch: str
) -> Dict[str, np.ndarray]:
    """
    Compute differences from reference batch.
    
    Args:
        batch_stats: Batch statistics dictionary
        reference_batch: Reference batch identifier
    
    Returns:
        Dictionary of difference spectra
    """
    reference_mean = batch_stats[reference_batch]["mean"]
    differences = {}
    
    for batch, stats in batch_stats.items():
        if batch != reference_batch:
            differences[batch] = stats["mean"] - reference_mean
    
    return differences


def plot_batch_drift(
    spectra: np.ndarray,
    meta: Dict[str, Any],
    batch_key: str = "batch",
    wavenumbers: Optional[np.ndarray] = None,
    reference_batch: Optional[str] = None,
    confidence: float = 0.95,
    save_path: Optional[Union[str, Path]] = None,
    figure_size: Tuple[float, float] = (14, 10),
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize batch drift with confidence bands and difference plots.
    
    Creates a two-panel visualization:
    - Top panel: Overlay of mean spectra per batch with confidence bands
    - Bottom panel: Difference from reference batch
    
    Args:
        spectra: Spectral data array (n_samples, n_features)
        meta: Metadata dictionary containing batch information
        batch_key: Key in meta dict for batch labels (default "batch")
        wavenumbers: Wavenumber array for x-axis (optional)
        reference_batch: Reference batch for difference calculation (auto-selected if None)
        confidence: Confidence level for bands (default 0.95, or 0.99)
        save_path: Directory to save plot (default None)
        figure_size: Figure size in inches (default 14x10)
        dpi: Resolution for saved figure (default 300)
    
    Returns:
        Matplotlib Figure object
    
    Raises:
        KeyError: If batch_key not found in meta
        ValueError: If spectra shape incompatible with meta
    
    Example:
        >>> spectra = np.random.randn(100, 500)
        >>> meta = {"batch": np.repeat(["A", "B", "C"], [30, 35, 35])}
        >>> fig = plot_batch_drift(spectra, meta, batch_key="batch")
    """
    # Validate inputs
    if batch_key not in meta:
        raise KeyError(f"Batch key '{batch_key}' not found in metadata")
    
    batch_labels = np.array(meta[batch_key])
    
    if len(batch_labels) != spectra.shape[0]:
        raise ValueError(
            f"Batch labels length ({len(batch_labels)}) does not match "
            f"spectra samples ({spectra.shape[0]})"
        )
    
    # Use feature indices if wavenumbers not provided
    if wavenumbers is None:
        wavenumbers = np.arange(spectra.shape[1])
    
    # Compute batch statistics
    batch_stats = _compute_batch_statistics(spectra, batch_labels, confidence)
    unique_batches = sorted(batch_stats.keys())
    
    # Auto-select reference batch (first batch or most samples)
    if reference_batch is None:
        # Select batch with most samples as reference
        reference_batch = max(
            batch_stats.keys(), key=lambda b: batch_stats[b]["n_samples"]
        )
    
    # Compute differences from reference
    differences = _compute_difference_from_reference(batch_stats, reference_batch)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size, height_ratios=[1.2, 1])
    
    # Color map for batches
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_batches)))
    
    # Top panel: Overlay with confidence bands
    for idx, batch in enumerate(unique_batches):
        stats = batch_stats[batch]
        color = colors[idx]
        label = f"Batch {batch} (n={stats['n_samples']})"
        
        # Plot mean spectrum
        ax1.plot(wavenumbers, stats["mean"], color=color, linewidth=2, label=label)
        
        # Plot confidence band
        ax1.fill_between(
            wavenumbers,
            stats["ci_lower"],
            stats["ci_upper"],
            color=color,
            alpha=0.2,
        )
    
    ax1.set_xlabel("Wavenumber (cm⁻¹)" if wavenumbers.max() > 100 else "Feature Index", fontsize=11)
    ax1.set_ylabel("Intensity (a.u.)", fontsize=11)
    ax1.set_title(
        f"Batch Drift Analysis ({int(confidence*100)}% Confidence Bands)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Difference from reference
    for idx, batch in enumerate(unique_batches):
        if batch == reference_batch:
            continue
        
        color = colors[idx]
        diff = differences[batch]
        
        ax2.plot(
            wavenumbers,
            diff,
            color=color,
            linewidth=1.5,
            label=f"Batch {batch} - Ref",
        )
        ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    ax2.set_xlabel("Wavenumber (cm⁻¹)" if wavenumbers.max() > 100 else "Feature Index", fontsize=11)
    ax2.set_ylabel("Difference (a.u.)", fontsize=11)
    ax2.set_title(
        f"Difference from Reference Batch '{reference_batch}'",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        png_path = save_path / "batch_drift.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def _compute_stage_statistics(
    spectra_by_stage: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute mean and std per stage.
    
    Args:
        spectra_by_stage: Dictionary mapping stage names to spectral arrays
    
    Returns:
        Dictionary with stage statistics
    """
    stats = {}
    
    for stage, stage_spectra in spectra_by_stage.items():
        mean_spectrum = np.mean(stage_spectra, axis=0)
        std_spectrum = np.std(stage_spectra, axis=0)
        
        stats[stage] = {
            "mean": mean_spectrum,
            "std": std_spectrum,
            "n_samples": stage_spectra.shape[0],
        }
    
    return stats


def _compute_pairwise_differences(
    stage_stats: Dict[str, Dict[str, np.ndarray]], baseline_stage: str
) -> Dict[str, np.ndarray]:
    """
    Compute pairwise differences from baseline stage.
    
    Args:
        stage_stats: Stage statistics dictionary
        baseline_stage: Baseline stage identifier
    
    Returns:
        Dictionary of difference spectra
    """
    baseline_mean = stage_stats[baseline_stage]["mean"]
    differences = {}
    
    for stage, stats in stage_stats.items():
        if stage != baseline_stage:
            differences[stage] = stats["mean"] - baseline_mean
    
    return differences


def _auto_select_baseline_stage(
    spectra_by_stage: Dict[str, np.ndarray], stage_order: Optional[List[str]] = None
) -> str:
    """
    Auto-select baseline stage.
    
    Priority:
    1. First stage in stage_order (if provided)
    2. Stage with name containing "raw" or "original"
    3. Stage with most samples
    4. First stage alphabetically
    
    Args:
        spectra_by_stage: Dictionary mapping stage names to spectral arrays
        stage_order: Optional ordering of stages
    
    Returns:
        Selected baseline stage name
    """
    if stage_order and len(stage_order) > 0:
        # Use first stage in order
        return stage_order[0]
    
    # Check for "raw" or "original" in names
    for stage in spectra_by_stage.keys():
        if "raw" in stage.lower() or "original" in stage.lower():
            return stage
    
    # Use stage with most samples
    max_stage = max(
        spectra_by_stage.keys(), key=lambda s: spectra_by_stage[s].shape[0]
    )
    return max_stage


def plot_stage_differences(
    spectra_by_stage: Dict[str, np.ndarray],
    wavenumbers: Optional[np.ndarray] = None,
    baseline_stage: Optional[str] = None,
    stage_order: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figure_size: Tuple[float, float] = (14, 10),
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize differences between processing stages.
    
    Creates a two-panel visualization:
    - Top panel: Overlay of mean spectra per stage
    - Bottom panel: Pairwise differences from baseline stage
    
    Args:
        spectra_by_stage: Dictionary mapping stage names to spectral arrays (n_samples, n_features)
        wavenumbers: Wavenumber array for x-axis (optional)
        baseline_stage: Baseline stage for difference calculation (auto-selected if None)
        stage_order: Optional list defining stage ordering for display
        save_path: Directory to save plot (default None)
        figure_size: Figure size in inches (default 14x10)
        dpi: Resolution for saved figure (default 300)
    
    Returns:
        Matplotlib Figure object
    
    Raises:
        ValueError: If spectra_by_stage is empty or has incompatible shapes
    
    Example:
        >>> spectra_dict = {
        ...     "raw": np.random.randn(50, 500),
        ...     "baseline_corrected": np.random.randn(50, 500),
        ...     "normalized": np.random.randn(50, 500),
        ... }
        >>> fig = plot_stage_differences(spectra_dict, baseline_stage="raw")
    """
    # Validate inputs
    if not spectra_by_stage:
        raise ValueError("spectra_by_stage cannot be empty")
    
    # Check all spectra have same number of features
    n_features = next(iter(spectra_by_stage.values())).shape[1]
    for stage, spectra in spectra_by_stage.items():
        if spectra.shape[1] != n_features:
            raise ValueError(
                f"Stage '{stage}' has {spectra.shape[1]} features, "
                f"expected {n_features}"
            )
    
    # Use feature indices if wavenumbers not provided
    if wavenumbers is None:
        wavenumbers = np.arange(n_features)
    
    # Auto-select baseline stage if not provided
    if baseline_stage is None:
        baseline_stage = _auto_select_baseline_stage(spectra_by_stage, stage_order)
    
    # Compute stage statistics
    stage_stats = _compute_stage_statistics(spectra_by_stage)
    
    # Compute pairwise differences
    differences = _compute_pairwise_differences(stage_stats, baseline_stage)
    
    # Determine display order
    if stage_order is None:
        # Put baseline first, then alphabetical
        stages = [baseline_stage] + sorted(
            [s for s in stage_stats.keys() if s != baseline_stage]
        )
    else:
        # Filter to only stages present in data
        stages = [s for s in stage_order if s in stage_stats]
        # Add any missing stages at the end
        for stage in stage_stats.keys():
            if stage not in stages:
                stages.append(stage)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size, height_ratios=[1.2, 1])
    
    # Color map for stages
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(stages)))
    
    # Top panel: Overlay of mean spectra
    for idx, stage in enumerate(stages):
        stats = stage_stats[stage]
        color = colors[idx]
        marker = "●" if stage == baseline_stage else ""
        label = f"{marker} {stage} (n={stats['n_samples']})"
        
        # Plot mean spectrum
        ax1.plot(wavenumbers, stats["mean"], color=color, linewidth=2, label=label)
        
        # Plot std band (lighter)
        ax1.fill_between(
            wavenumbers,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=color,
            alpha=0.15,
        )
    
    ax1.set_xlabel("Wavenumber (cm⁻¹)" if wavenumbers.max() > 100 else "Feature Index", fontsize=11)
    ax1.set_ylabel("Intensity (a.u.)", fontsize=11)
    ax1.set_title(
        "Stage-wise Mean Spectra (±1 std shaded)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Differences from baseline
    for idx, stage in enumerate(stages):
        if stage == baseline_stage:
            continue
        
        color = colors[idx]
        diff = differences[stage]
        
        ax2.plot(
            wavenumbers,
            diff,
            color=color,
            linewidth=1.5,
            label=f"{stage} - {baseline_stage}",
        )
    
    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Wavenumber (cm⁻¹)" if wavenumbers.max() > 100 else "Feature Index", fontsize=11)
    ax2.set_ylabel("Difference (a.u.)", fontsize=11)
    ax2.set_title(
        f"Differences from Baseline Stage '{baseline_stage}'",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        png_path = save_path / "stage_differences.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def get_batch_statistics(
    spectra: np.ndarray,
    meta: Dict[str, Any],
    batch_key: str = "batch",
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Extract batch drift statistics.
    
    Args:
        spectra: Spectral data array (n_samples, n_features)
        meta: Metadata dictionary containing batch information
        batch_key: Key in meta dict for batch labels
        confidence: Confidence level for intervals (default 0.95)
    
    Returns:
        Dictionary with batch statistics and summary metrics
    
    Example:
        >>> spectra = np.random.randn(100, 500)
        >>> meta = {"batch": np.repeat(["A", "B"], [50, 50])}
        >>> stats = get_batch_statistics(spectra, meta)
        >>> print(stats["summary"]["total_batches"])
    """
    batch_labels = np.array(meta[batch_key])
    batch_stats = _compute_batch_statistics(spectra, batch_labels, confidence)
    
    # Compute summary statistics
    unique_batches = sorted(batch_stats.keys())
    total_samples = sum(s["n_samples"] for s in batch_stats.values())
    
    # Compute max pairwise difference
    max_diff = 0.0
    max_diff_pair = None
    for i, batch1 in enumerate(unique_batches):
        for batch2 in unique_batches[i + 1 :]:
            diff = np.abs(
                batch_stats[batch1]["mean"] - batch_stats[batch2]["mean"]
            ).max()
            if diff > max_diff:
                max_diff = diff
                max_diff_pair = (batch1, batch2)
    
    summary = {
        "total_batches": len(unique_batches),
        "total_samples": total_samples,
        "batch_names": unique_batches,
        "samples_per_batch": {b: s["n_samples"] for b, s in batch_stats.items()},
        "max_pairwise_difference": float(max_diff),
        "max_difference_pair": max_diff_pair,
    }
    
    return {
        "batch_stats": batch_stats,
        "summary": summary,
    }


def get_stage_statistics(
    spectra_by_stage: Dict[str, np.ndarray],
    baseline_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract stage difference statistics.
    
    Args:
        spectra_by_stage: Dictionary mapping stage names to spectral arrays
        baseline_stage: Baseline stage for difference calculation (auto-selected if None)
    
    Returns:
        Dictionary with stage statistics and summary metrics
    
    Example:
        >>> spectra_dict = {
        ...     "raw": np.random.randn(50, 500),
        ...     "processed": np.random.randn(50, 500),
        ... }
        >>> stats = get_stage_statistics(spectra_dict)
        >>> print(stats["summary"]["total_stages"])
    """
    # Auto-select baseline if not provided
    if baseline_stage is None:
        baseline_stage = _auto_select_baseline_stage(spectra_by_stage)
    
    # Compute stage statistics
    stage_stats = _compute_stage_statistics(spectra_by_stage)
    
    # Compute differences
    differences = _compute_pairwise_differences(stage_stats, baseline_stage)
    
    # Compute summary statistics
    total_samples = sum(s["n_samples"] for s in stage_stats.values())
    stage_names = sorted(stage_stats.keys())
    
    # Compute max difference from baseline
    max_diff = 0.0
    max_diff_stage = None
    for stage, diff in differences.items():
        stage_max_diff = np.abs(diff).max()
        if stage_max_diff > max_diff:
            max_diff = stage_max_diff
            max_diff_stage = stage
    
    summary = {
        "total_stages": len(stage_names),
        "total_samples": total_samples,
        "stage_names": stage_names,
        "baseline_stage": baseline_stage,
        "samples_per_stage": {s: stats["n_samples"] for s, stats in stage_stats.items()},
        "max_difference_from_baseline": float(max_diff),
        "max_difference_stage": max_diff_stage,
    }
    
    return {
        "stage_stats": stage_stats,
        "differences": differences,
        "summary": summary,
    }


def _compute_similarity_matrix(
    spectra: np.ndarray, metric: str = "cosine"
) -> np.ndarray:
    """
    Compute pairwise similarity matrix.
    
    Args:
        spectra: Spectral data (n_samples, n_features)
        metric: Similarity metric ("cosine" or "correlation")
    
    Returns:
        Similarity matrix (n_samples, n_samples)
    """
    if metric == "cosine":
        # Cosine similarity: 1 - cosine_distance
        distances = pdist(spectra, metric="cosine")
        similarities = 1 - squareform(distances)
    elif metric == "correlation":
        # Pearson correlation
        distances = pdist(spectra, metric="correlation")
        similarities = 1 - squareform(distances)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'correlation'")
    
    return similarities


def _perform_hierarchical_clustering(
    similarity_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering on similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix (n_samples, n_samples)
    
    Returns:
        Tuple of (row_linkage, col_linkage) for dendrogram
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Ensure distances are valid (non-negative, zero diagonal)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.maximum(distance_matrix, 0)
    
    # Convert to condensed distance matrix
    condensed = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage = hierarchy.linkage(condensed, method="average")
    
    return linkage, linkage


def plot_replicate_similarity(
    spectra: np.ndarray,
    labels: Optional[List[str]] = None,
    metric: str = "cosine",
    cluster: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figure_size: Tuple[float, float] = (12, 10),
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize similarity between replicates using clustered heatmap.
    
    Creates a heatmap showing pairwise similarities between all samples,
    optionally with hierarchical clustering dendrograms.
    
    Args:
        spectra: Spectral data array (n_samples, n_features)
        labels: Sample labels for axes (default: Sample_0, Sample_1, ...)
        metric: Similarity metric ("cosine" or "correlation", default "cosine")
        cluster: Whether to perform hierarchical clustering (default True)
        save_path: Directory to save plot (default None)
        figure_size: Figure size in inches (default 12x10)
        dpi: Resolution for saved figure (default 300)
    
    Returns:
        Matplotlib Figure object
    
    Raises:
        ValueError: If metric is not "cosine" or "correlation"
    
    Example:
        >>> spectra = np.random.randn(50, 500)
        >>> labels = [f"Rep_{i}" for i in range(50)]
        >>> fig = plot_replicate_similarity(spectra, labels, metric="cosine")
    """
    n_samples = spectra.shape[0]
    
    # Generate default labels if not provided
    if labels is None:
        labels = [f"Sample_{i}" for i in range(n_samples)]
    
    # Compute similarity matrix
    similarity = _compute_similarity_matrix(spectra, metric)
    
    # Perform clustering if requested
    if cluster:
        row_linkage, col_linkage = _perform_hierarchical_clustering(similarity)
        row_dendrogram = hierarchy.dendrogram(row_linkage, no_plot=True)
        row_order = row_dendrogram["leaves"]
        
        # Reorder similarity matrix and labels
        similarity_ordered = similarity[row_order, :][:, row_order]
        labels_ordered = [labels[i] for i in row_order]
    else:
        similarity_ordered = similarity
        labels_ordered = labels
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Plot heatmap
    im = ax.imshow(
        similarity_ordered,
        cmap="RdYlGn",
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{metric.capitalize()} Similarity", fontsize=11)
    
    # Set ticks and labels
    tick_positions = np.arange(n_samples)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels_ordered, rotation=90, ha="right", fontsize=8)
    ax.set_yticklabels(labels_ordered, fontsize=8)
    
    # Add title
    cluster_text = " (Hierarchical Clustering)" if cluster else ""
    ax.set_title(
        f"Replicate Similarity Matrix{cluster_text}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    
    # Add grid
    ax.set_xticks(tick_positions - 0.5, minor=True)
    ax.set_yticks(tick_positions - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        png_path = save_path / "replicate_similarity.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def _parse_timestamps(
    time_values: np.ndarray, time_format: Optional[str] = None
) -> np.ndarray:
    """
    Parse timestamp strings to numeric values.
    
    Args:
        time_values: Array of timestamps (strings, floats, or datetime objects)
        time_format: Strptime format string (optional)
    
    Returns:
        Numeric time array (float)
    """
    if isinstance(time_values[0], (int, float, np.integer, np.floating)):
        # Already numeric
        return time_values.astype(float)
    
    if isinstance(time_values[0], datetime):
        # Convert datetime to timestamp
        return np.array([dt.timestamp() for dt in time_values])
    
    # Parse strings
    if time_format is not None:
        timestamps = []
        for val in time_values:
            dt = datetime.strptime(str(val), time_format)
            timestamps.append(dt.timestamp())
        return np.array(timestamps)
    
    # Try ISO format
    try:
        timestamps = []
        for val in time_values:
            dt = datetime.fromisoformat(str(val))
            timestamps.append(dt.timestamp())
        return np.array(timestamps)
    except (ValueError, TypeError):
        # Last resort: treat as numeric indices
        return np.arange(len(time_values), dtype=float)


def _compute_rolling_average(
    values: np.ndarray, window_size: int
) -> np.ndarray:
    """
    Compute rolling average.
    
    Args:
        values: Time series values
        window_size: Window size for rolling average
    
    Returns:
        Smoothed values
    """
    if window_size <= 1:
        return values
    
    # Pad edges to handle boundaries
    padded = np.pad(values, (window_size // 2, window_size // 2), mode="edge")
    
    # Compute rolling average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode="valid")
    
    return smoothed[: len(values)]


def plot_temporal_drift(
    spectra: np.ndarray,
    meta: Dict[str, Any],
    time_key: str = "timestamp",
    band_indices: Optional[List[int]] = None,
    wavenumbers: Optional[np.ndarray] = None,
    band_ranges: Optional[List[Tuple[float, float]]] = None,
    rolling_window: int = 1,
    time_format: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figure_size: Tuple[float, float] = (14, 8),
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize temporal drift trends in key spectral bands.
    
    Creates a multi-panel plot showing intensity trends over time for
    selected spectral bands, with optional rolling average smoothing.
    
    Args:
        spectra: Spectral data array (n_samples, n_features)
        meta: Metadata dictionary containing time information
        time_key: Key in meta dict for timestamps (default "timestamp")
        band_indices: List of feature indices to plot (optional)
        wavenumbers: Wavenumber array for x-axis interpretation (optional)
        band_ranges: List of (min, max) wavenumber ranges to average (optional)
        rolling_window: Window size for rolling average (default 1, no smoothing)
        time_format: Strptime format for timestamp parsing (optional)
        save_path: Directory to save plot (default None)
        figure_size: Figure size in inches (default 14x8)
        dpi: Resolution for saved figure (default 300)
    
    Returns:
        Matplotlib Figure object
    
    Raises:
        KeyError: If time_key not found in meta
        ValueError: If spectra shape incompatible with meta
    
    Example:
        >>> spectra = np.random.randn(100, 500)
        >>> meta = {"timestamp": np.arange(100)}
        >>> fig = plot_temporal_drift(
        ...     spectra, meta, band_indices=[100, 200, 300], rolling_window=5
        ... )
    """
    # Validate inputs
    if time_key not in meta:
        raise KeyError(f"Time key '{time_key}' not found in metadata")
    
    time_values = np.array(meta[time_key])
    
    if len(time_values) != spectra.shape[0]:
        raise ValueError(
            f"Time values length ({len(time_values)}) does not match "
            f"spectra samples ({spectra.shape[0]})"
        )
    
    # Parse timestamps
    time_numeric = _parse_timestamps(time_values, time_format)
    
    # Sort by time
    sort_idx = np.argsort(time_numeric)
    time_sorted = time_numeric[sort_idx]
    spectra_sorted = spectra[sort_idx]
    
    # Determine bands to plot
    if band_ranges is not None and wavenumbers is not None:
        # Average over specified wavenumber ranges
        bands_data = []
        band_labels = []
        
        for wn_min, wn_max in band_ranges:
            # Find indices in range
            mask = (wavenumbers >= wn_min) & (wavenumbers <= wn_max)
            if not np.any(mask):
                continue
            
            # Average intensity in range
            band_intensities = np.mean(spectra_sorted[:, mask], axis=1)
            bands_data.append(band_intensities)
            band_labels.append(f"{wn_min:.0f}-{wn_max:.0f} cm⁻¹")
    
    elif band_indices is not None:
        # Use specific indices
        bands_data = [spectra_sorted[:, idx] for idx in band_indices]
        
        if wavenumbers is not None:
            band_labels = [f"{wavenumbers[idx]:.0f} cm⁻¹" for idx in band_indices]
        else:
            band_labels = [f"Feature {idx}" for idx in band_indices]
    
    else:
        # Auto-select: 5 evenly spaced bands
        n_features = spectra.shape[1]
        band_indices = np.linspace(0, n_features - 1, 5, dtype=int)
        bands_data = [spectra_sorted[:, idx] for idx in band_indices]
        
        if wavenumbers is not None:
            band_labels = [f"{wavenumbers[idx]:.0f} cm⁻¹" for idx in band_indices]
        else:
            band_labels = [f"Feature {idx}" for idx in band_indices]
    
    n_bands = len(bands_data)
    
    # Create subplots
    fig, axes = plt.subplots(n_bands, 1, figsize=figure_size, sharex=True)
    if n_bands == 1:
        axes = [axes]
    
    # Plot each band
    colors = plt.cm.tab10(np.arange(n_bands))
    
    for idx, (band_data, label, color, ax) in enumerate(
        zip(bands_data, band_labels, colors, axes)
    ):
        # Apply rolling average if requested
        if rolling_window > 1:
            band_smoothed = _compute_rolling_average(band_data, rolling_window)
            
            # Plot original data as light scatter
            ax.scatter(
                time_sorted,
                band_data,
                color=color,
                alpha=0.3,
                s=20,
                label="Original",
            )
            
            # Plot smoothed as line
            ax.plot(
                time_sorted,
                band_smoothed,
                color=color,
                linewidth=2,
                label=f"Rolling avg (window={rolling_window})",
            )
            ax.legend(loc="best", fontsize=8)
        else:
            # Plot as scatter with line
            ax.plot(
                time_sorted,
                band_data,
                color=color,
                linewidth=1.5,
                marker="o",
                markersize=4,
                alpha=0.7,
            )
        
        # Labels and grid
        ax.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
    
    # Bottom axis label
    axes[-1].set_xlabel("Time", fontsize=11)
    
    # Overall title
    fig.suptitle(
        "Temporal Drift Analysis",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        png_path = save_path / "temporal_drift.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    
    return fig
