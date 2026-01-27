"""Comprehensive visualization functions for FoodSpec reporting.

This module provides all missing visualization functions needed for complete
experiment reports. Includes preprocessing overlays, dimensionality reduction,
uncertainty metrics, and more.

All functions are deterministic (seed-controlled) and save to stable filenames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from foodspec.viz.style import apply_style


def plot_raw_vs_processed_overlay(
    wavenumbers: np.ndarray,
    X_raw: np.ndarray,
    X_processed: np.ndarray,
    n_samples: int = 5,
    *,
    seed: int = 0,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot overlay of raw vs processed spectra for visual QC.
    
    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber axis (n_features,)
    X_raw : np.ndarray
        Raw spectra (n_samples, n_features)
    X_processed : np.ndarray
        Processed spectra (n_samples, n_features)
    n_samples : int, optional
        Number of samples to plot (default: 5)
    seed : int, optional
        Random seed for sample selection
    save_path : Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    apply_style()

    rng = np.random.default_rng(seed)
    n_total = X_raw.shape[0]
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(8, 2 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        ax.plot(wavenumbers, X_raw[idx], 'gray', alpha=0.7, label='Raw', linewidth=1)
        ax.plot(wavenumbers, X_processed[idx], '#2a6fdb', label='Processed', linewidth=1.5)
        ax.set_ylabel('Intensity (a.u.)')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'Sample {idx}', fontsize=9)

    axes[-1].set_xlabel('Wavenumber (cm$^{-1}$)')
    axes[-1].invert_xaxis()
    fig.suptitle('Raw vs Processed Spectra', fontsize=11, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_pca_umap(
    X_embedded: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'PCA',
    *,
    seed: int = 0,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot 2D embedding (PCA or UMAP) with optional class coloring.
    
    Parameters
    ----------
    X_embedded : np.ndarray
        2D embedded coordinates (n_samples, 2)
    labels : np.ndarray, optional
        Class labels for coloring
    method : str, optional
        Embedding method name for title (default: 'PCA')
    seed : int, optional
        Random seed (for consistency)
    save_path : Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(6, 5))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                      c=[color], label=str(label), alpha=0.7, s=30, edgecolors='k', linewidths=0.5)
        ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                  c='#2a6fdb', alpha=0.6, s=30, edgecolors='k', linewidths=0.5)

    ax.set_xlabel(f'{method} Component 1')
    ax.set_ylabel(f'{method} Component 2')
    ax.set_title(f'{method} Projection', fontweight='bold')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_coverage_efficiency_curve(
    alpha_values: np.ndarray,
    coverage: np.ndarray,
    efficiency: np.ndarray,
    target_alpha: float = 0.1,
    *,
    seed: int = 0,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot coverage vs efficiency trade-off for conformal prediction.
    
    Parameters
    ----------
    alpha_values : np.ndarray
        Array of alpha (significance) values
    coverage : np.ndarray
        Empirical coverage at each alpha
    efficiency : np.ndarray
        Efficiency (1 - avg set size / n_classes) at each alpha
    target_alpha : float, optional
        Target alpha to highlight (default: 0.1)
    seed : int, optional
        Random seed (for consistency)
    save_path : Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot coverage and efficiency
    ax.plot(alpha_values, coverage, 'o-', label='Coverage', color='#2a6fdb', linewidth=2)
    ax.plot(alpha_values, efficiency, 's-', label='Efficiency', color='#e67e22', linewidth=2)

    # Highlight target alpha
    if target_alpha in alpha_values:
        idx = np.where(alpha_values == target_alpha)[0][0]
        ax.axvline(target_alpha, color='red', linestyle='--', alpha=0.5, label=f'α={target_alpha}')
        ax.plot(target_alpha, coverage[idx], 'ro', markersize=8)
        ax.plot(target_alpha, efficiency[idx], 'ro', markersize=8)

    ax.set_xlabel('Significance Level (α)')
    ax.set_ylabel('Metric Value')
    ax.set_title('Coverage vs Efficiency Trade-off', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, max(alpha_values) * 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_conformal_set_sizes(
    set_sizes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    seed: int = 0,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot distribution of conformal prediction set sizes.
    
    Parameters
    ----------
    set_sizes : np.ndarray
        Prediction set size for each sample (n_samples,)
    labels : np.ndarray, optional
        Class labels for grouped analysis
    seed : int, optional
        Random seed (for consistency)
    save_path : Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram of set sizes
    axes[0].hist(set_sizes, bins=np.arange(0.5, max(set_sizes) + 1.5, 1),
                color='#2a6fdb', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Prediction Set Size')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Set Sizes', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.2)

    # Statistics
    mean_size = np.mean(set_sizes)
    median_size = np.median(set_sizes)
    axes[0].axvline(mean_size, color='red', linestyle='--', label=f'Mean: {mean_size:.2f}')
    axes[0].axvline(median_size, color='orange', linestyle='--', label=f'Median: {median_size:.1f}')
    axes[0].legend()

    # Box plot by class (if labels provided)
    if labels is not None:
        unique_labels = sorted(np.unique(labels))
        data_by_class = [set_sizes[labels == lbl] for lbl in unique_labels]
        axes[1].boxplot(data_by_class, labels=[str(lbl) for lbl in unique_labels])
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Set Size')
        axes[1].set_title('Set Sizes by Class', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.2)
    else:
        axes[1].text(0.5, 0.5, 'No class labels\navailable',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_abstention_distribution(
    abstain_flags: np.ndarray,
    labels: Optional[np.ndarray] = None,
    batch_ids: Optional[np.ndarray] = None,
    *,
    seed: int = 0,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot abstention distribution by class and/or batch.
    
    Parameters
    ----------
    abstain_flags : np.ndarray
        Boolean array indicating abstention (n_samples,)
    labels : np.ndarray, optional
        Class labels for grouped analysis
    batch_ids : np.ndarray, optional
        Batch IDs for batch-wise analysis
    seed : int, optional
        Random seed (for consistency)
    save_path : Path, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    apply_style()

    n_plots = sum([labels is not None, batch_ids is not None])
    if n_plots == 0:
        n_plots = 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Overall abstention rate
    abstention_rate = np.mean(abstain_flags)

    # By class
    if labels is not None and plot_idx < len(axes):
        unique_labels = sorted(np.unique(labels))
        rates = [np.mean(abstain_flags[labels == lbl]) for lbl in unique_labels]

        axes[plot_idx].bar(range(len(unique_labels)), rates, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[plot_idx].axhline(abstention_rate, color='black', linestyle='--',
                              label=f'Overall: {abstention_rate:.1%}')
        axes[plot_idx].set_xlabel('Class')
        axes[plot_idx].set_ylabel('Abstention Rate')
        axes[plot_idx].set_title('Abstention by Class', fontweight='bold')
        axes[plot_idx].set_xticks(range(len(unique_labels)))
        axes[plot_idx].set_xticklabels([str(lbl) for lbl in unique_labels])
        axes[plot_idx].set_ylim(0, 1)
        axes[plot_idx].legend()
        axes[plot_idx].grid(axis='y', alpha=0.2)
        plot_idx += 1

    # By batch
    if batch_ids is not None and plot_idx < len(axes):
        unique_batches = sorted(np.unique(batch_ids))
        rates = [np.mean(abstain_flags[batch_ids == b]) for b in unique_batches]

        axes[plot_idx].bar(range(len(unique_batches)), rates, color='#e67e22', alpha=0.7, edgecolor='black')
        axes[plot_idx].axhline(abstention_rate, color='black', linestyle='--',
                              label=f'Overall: {abstention_rate:.1%}')
        axes[plot_idx].set_xlabel('Batch')
        axes[plot_idx].set_ylabel('Abstention Rate')
        axes[plot_idx].set_title('Abstention by Batch', fontweight='bold')
        axes[plot_idx].set_xticks(range(len(unique_batches)))
        axes[plot_idx].set_xticklabels([str(b) for b in unique_batches], rotation=45)
        axes[plot_idx].set_ylim(0, 1)
        axes[plot_idx].legend()
        axes[plot_idx].grid(axis='y', alpha=0.2)
        plot_idx += 1

    # If no grouping, show overall summary
    if labels is None and batch_ids is None:
        n_abstained = np.sum(abstain_flags)
        n_total = len(abstain_flags)

        axes[0].bar(['Predicted', 'Abstained'],
                   [n_total - n_abstained, n_abstained],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Abstention Summary\n({abstention_rate:.1%} abstained)', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.2)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


__all__ = [
    "plot_raw_vs_processed_overlay",
    "plot_pca_umap",
    "plot_coverage_efficiency_curve",
    "plot_conformal_set_sizes",
    "plot_abstention_distribution",
]
