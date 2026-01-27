"""QC visualization for preprocessing results.

Provides plots for:
- Raw vs processed overlay
- Baseline estimates
- Outlier detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_raw_vs_processed(
    X_raw: np.ndarray,
    X_processed: np.ndarray,
    wavenumbers: np.ndarray,
    n_samples: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str | Path] = None,
):
    """Plot raw vs processed spectra overlay.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw spectral data (n_samples, n_features).
    X_processed : np.ndarray
        Processed spectral data (n_samples, n_features).
    wavenumbers : np.ndarray
        Wavenumber axis.
    n_samples : int
        Number of samples to plot (default 5).
    figsize : Tuple[int, int]
        Figure size.
    save_path : str | Path | None
        If provided, save figure to this path.
    """
    n_samples = min(n_samples, X_raw.shape[0])
    indices = np.linspace(0, X_raw.shape[0] - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, sharex=True)
    if n_samples == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        ax.plot(wavenumbers, X_raw[idx], "b-", alpha=0.5, label="Raw", linewidth=1)
        ax.plot(wavenumbers, X_processed[idx], "r-", alpha=0.8, label="Processed", linewidth=1.5)
        ax.set_ylabel(f"Sample {idx}\nIntensity", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
    fig.suptitle("Raw vs Processed Spectra", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_baseline_overlay(
    X_raw: np.ndarray,
    baselines: np.ndarray,
    wavenumbers: np.ndarray,
    n_samples: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str | Path] = None,
):
    """Plot baseline estimates overlay.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw spectral data (n_samples, n_features).
    baselines : np.ndarray
        Estimated baselines (n_samples, n_features).
    wavenumbers : np.ndarray
        Wavenumber axis.
    n_samples : int
        Number of samples to plot (default 5).
    figsize : Tuple[int, int]
        Figure size.
    save_path : str | Path | None
        If provided, save figure to this path.
    """
    n_samples = min(n_samples, X_raw.shape[0])
    indices = np.linspace(0, X_raw.shape[0] - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, sharex=True)
    if n_samples == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        ax.plot(wavenumbers, X_raw[idx], "k-", alpha=0.6, label="Raw", linewidth=1)
        ax.plot(wavenumbers, baselines[idx], "r--", alpha=0.8, label="Baseline", linewidth=2)
        ax.fill_between(wavenumbers, 0, baselines[idx], alpha=0.2, color="red")
        ax.set_ylabel(f"Sample {idx}\nIntensity", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
    fig.suptitle("Baseline Estimation Overlay", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_outlier_summary(
    X: np.ndarray,
    outlier_mask: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str | Path] = None,
):
    """Plot outlier detection summary.

    Parameters
    ----------
    X : np.ndarray
        Spectral data (n_samples, n_features).
    outlier_mask : np.ndarray | None
        Boolean mask indicating outliers (n_samples,).
    figsize : Tuple[int, int]
        Figure size.
    save_path : str | Path | None
        If provided, save figure to this path.
    """
    # Compute spectral norms and mean distances
    norms = np.linalg.norm(X, axis=1)
    mean_spectrum = np.mean(X, axis=0)
    distances = np.linalg.norm(X - mean_spectrum, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Norm histogram
    ax1 = axes[0]
    ax1.hist(norms, bins=30, alpha=0.7, edgecolor="black")
    if outlier_mask is not None:
        outlier_norms = norms[outlier_mask]
        ax1.hist(outlier_norms, bins=30, alpha=0.7, color="red", label="Outliers")
        ax1.legend()
    ax1.set_xlabel("Spectral Norm", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.set_title("Distribution of Spectral Norms", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Plot 2: Distance from mean
    ax2 = axes[1]
    ax2.hist(distances, bins=30, alpha=0.7, edgecolor="black")
    if outlier_mask is not None:
        outlier_distances = distances[outlier_mask]
        ax2.hist(outlier_distances, bins=30, alpha=0.7, color="red", label="Outliers")
        ax2.legend()
    ax2.set_xlabel("Distance from Mean Spectrum", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.set_title("Distribution of Distances", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.suptitle("Outlier Detection Summary", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def generate_qc_report(
    X_raw: np.ndarray,
    X_processed: np.ndarray,
    wavenumbers: np.ndarray,
    baselines: Optional[np.ndarray] = None,
    outlier_mask: Optional[np.ndarray] = None,
    output_dir: str | Path = ".",
):
    """Generate complete QC report with all plots.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw spectral data.
    X_processed : np.ndarray
        Processed spectral data.
    wavenumbers : np.ndarray
        Wavenumber axis.
    baselines : np.ndarray | None
        Baseline estimates (if available).
    outlier_mask : np.ndarray | None
        Outlier mask (if available).
    output_dir : str | Path
        Output directory for plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot raw vs processed
    plot_raw_vs_processed(
        X_raw,
        X_processed,
        wavenumbers,
        save_path=output_dir / "raw_vs_processed_overlay.png",
    )

    # Plot baseline if available
    if baselines is not None:
        plot_baseline_overlay(
            X_raw,
            baselines,
            wavenumbers,
            save_path=output_dir / "baseline_estimate_overlay.png",
        )

    # Plot outlier summary
    plot_outlier_summary(
        X_processed,
        outlier_mask=outlier_mask,
        save_path=output_dir / "outlier_detection_summary.png",
    )


__all__ = [
    "plot_raw_vs_processed",
    "plot_baseline_overlay",
    "plot_outlier_summary",
    "generate_qc_report",
]
