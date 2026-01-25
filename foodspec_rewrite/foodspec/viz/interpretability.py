"""
Interpretability visualizations for spectral data analysis.

This module provides tools for understanding and interpreting spectral models:
- Feature importance visualization with spectral overlays
- Highlighting and labeling chemically relevant marker bands
- Multiple visualization styles (line overlay, color-coded peaks)
- Support for different importance metrics

Functions:
    plot_importance_overlay: Overlay importance values on spectrum
    plot_marker_bands: Highlight and label marker bands in spectrum
    get_band_statistics: Extract band-level statistics
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable


def _normalize_importance(importance: np.ndarray) -> np.ndarray:
    """
    Normalize importance values to [0, 1] range.
    
    Args:
        importance: Importance scores (n_features,)
    
    Returns:
        Normalized importance scores
    """
    if len(importance) == 0:
        return importance
    
    min_val = np.min(importance)
    max_val = np.max(importance)
    
    if max_val == min_val:
        # All same value
        return np.ones_like(importance) * 0.5
    
    normalized = (importance - min_val) / (max_val - min_val)
    return normalized


def _select_prominent_peaks(
    spectrum: np.ndarray, importance: np.ndarray, n_peaks: int = 5
) -> List[int]:
    """
    Select the most prominent peaks by importance.
    
    Args:
        spectrum: Spectral data (n_features,)
        importance: Importance scores (n_features,)
        n_peaks: Number of peaks to select
    
    Returns:
        Indices of top peaks
    """
    # Find local maxima in importance
    top_indices = np.argsort(importance)[-n_peaks:]
    # Sort by feature index for left-to-right ordering
    top_indices = np.sort(top_indices)
    return top_indices.tolist()


def _format_band_label(
    peak_idx: int,
    wavenumber: Optional[float] = None,
    importance: Optional[float] = None,
    name: Optional[str] = None,
) -> str:
    """
    Format a band label with available information.
    
    Args:
        peak_idx: Feature index
        wavenumber: Wavenumber value (optional)
        importance: Importance score (optional)
        name: Band name (optional)
    
    Returns:
        Formatted label string
    """
    parts = []
    
    # Build primary identifier
    if name:
        parts.append(name)
    
    if wavenumber is not None:
        if name:
            parts.append(f"({wavenumber:.1f})")
        else:
            parts.append(f"{wavenumber:.1f}")
    elif not name:
        parts.append(f"Band {peak_idx}")
    
    if importance is not None:
        parts.append(f"({importance:.2f})")
    
    return " ".join(parts)


def plot_importance_overlay(
    spectrum: np.ndarray,
    importance: np.ndarray,
    wavenumbers: Optional[np.ndarray] = None,
    style: str = "overlay",
    colormap: str = "RdYlGn",
    threshold: Optional[float] = None,
    highlight_peaks: bool = True,
    n_peaks: int = 5,
    band_names: Optional[Dict[int, str]] = None,
    save_path: Optional[Path] = None,
    figure_size: Tuple[int, int] = (14, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Overlay feature importance on spectral data.
    
    Creates a visualization showing the input spectrum with importance values
    indicated through color or line overlay, highlighting features that
    contribute most to model predictions.
    
    Args:
        spectrum: Spectral data (n_features,)
        importance: Importance scores (n_features,). Can be permutation
                   importance, coefficient magnitudes, etc.
        wavenumbers: Wavenumber array (n_features,), optional. If provided,
                    will be used as x-axis labels
        style: Visualization style - "overlay" (default), "bar", or "heat"
               - "overlay": Line overlay with color based on importance
               - "bar": Bar chart below spectrum
               - "heat": Heatmap background with color intensity
        colormap: Matplotlib colormap name (default "RdYlGn")
                 - Red = low importance
                 - Green = high importance
        threshold: Importance threshold for highlighting. Features below
                  threshold shown in gray. None = use median
        highlight_peaks: Show labels for most important peaks (default True)
        n_peaks: Number of top peaks to label (default 5)
        band_names: Dictionary mapping feature indices to band names
                   (optional). Used in peak labels
        save_path: Path to save PNG file (optional)
        figure_size: Figure dimensions in inches (default 14x6)
        dpi: Resolution for PNG export (default 300)
    
    Returns:
        matplotlib.pyplot.Figure
    
    Raises:
        ValueError: If spectrum and importance have mismatched lengths
        ValueError: If spectrum is empty or 1D
    """
    # Validate inputs
    spectrum = np.asarray(spectrum).ravel()
    importance = np.asarray(importance).ravel()
    
    if spectrum.shape[0] != importance.shape[0]:
        raise ValueError(
            f"Spectrum ({spectrum.shape[0]} features) and importance "
            f"({importance.shape[0]} values) must have same length"
        )
    
    if spectrum.shape[0] == 0:
        raise ValueError("Spectrum cannot be empty")
    
    # Set up x-axis
    if wavenumbers is not None:
        wavenumbers = np.asarray(wavenumbers).ravel()
        if wavenumbers.shape[0] != spectrum.shape[0]:
            raise ValueError("Wavenumbers must match spectrum length")
        x_data = wavenumbers
        x_label = "Wavenumber (cm⁻¹)"
    else:
        x_data = np.arange(spectrum.shape[0])
        x_label = "Feature Index"
    
    # Normalize importance
    importance_norm = _normalize_importance(importance)
    
    # Set threshold
    if threshold is None:
        threshold = np.median(importance_norm)
    else:
        threshold = np.clip(threshold, 0, 1)
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    if style == "overlay":
        # Plot spectrum with color overlay
        for i in range(len(x_data) - 1):
            x_segment = x_data[i : i + 2]
            y_segment = spectrum[i : i + 2]
            
            # Use importance at start of segment
            importance_val = importance_norm[i]
            if importance_val >= threshold:
                color = cmap(norm(importance_val))
                alpha = 0.8
                linewidth = 2
            else:
                color = "lightgray"
                alpha = 0.5
                linewidth = 1
            
            ax.plot(x_segment, y_segment, color=color, alpha=alpha, linewidth=linewidth)
        
        ax.set_ylabel("Intensity", fontsize=11)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_title("Spectrum with Feature Importance Overlay", fontsize=13, fontweight="bold")
        
    elif style == "bar":
        # Create dual-axis plot with spectrum and importance bars
        ax.plot(x_data, spectrum, color="black", linewidth=2, label="Spectrum", zorder=3)
        ax.set_ylabel("Spectrum Intensity", fontsize=11)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_title("Spectrum with Feature Importance", fontsize=13, fontweight="bold")
        
        # Create second y-axis for importance
        ax2 = ax.twinx()
        colors = [cmap(norm(val)) for val in importance_norm]
        ax2.bar(x_data, importance_norm, alpha=0.3, color=colors, width=1.0)
        ax2.set_ylabel("Importance (Normalized)", fontsize=11)
        ax2.set_ylim([0, 1])
        
    elif style == "heat":
        # Plot spectrum with heatmap background
        ax.plot(x_data, spectrum, color="black", linewidth=2.5, label="Spectrum", zorder=3)
        ax.set_ylabel("Intensity", fontsize=11)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_title("Spectrum with Importance Heat", fontsize=13, fontweight="bold")
        
        # Add background color shading based on importance
        ax2 = ax.twinx()
        ax2.remove()
        
        # Fill background with color based on importance
        y_min, y_max = ax.get_ylim()
        for i in range(len(x_data) - 1):
            x_segment = [x_data[i], x_data[i + 1]]
            importance_val = importance_norm[i]
            color = cmap(norm(importance_val))
            ax.axvspan(x_segment[0], x_segment[1], alpha=0.15, color=color)
    
    # Highlight and label peaks
    if highlight_peaks and len(spectrum) > 0:
        peak_indices = _select_prominent_peaks(spectrum, importance, n_peaks)
        
        for peak_idx in peak_indices:
            x_val = x_data[peak_idx]
            y_val = spectrum[peak_idx]
            importance_val = importance_norm[peak_idx]
            color = cmap(norm(importance_val))
            
            # Mark peak
            ax.plot(x_val, y_val, marker="o", markersize=8, color=color, 
                   markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            
            # Add label
            band_name = band_names.get(peak_idx) if band_names else None
            wavenumber = wavenumbers[peak_idx] if wavenumbers is not None else None
            label = _format_band_label(peak_idx, wavenumber, importance_val, band_name)
            
            ax.annotate(label, xy=(x_val, y_val), xytext=(0, 10),
                       textcoords="offset points", ha="center", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, 
                                alpha=0.3, edgecolor="black", linewidth=0.5),
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Normalized Importance", pad=0.02)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_marker_bands(
    spectrum: np.ndarray,
    marker_bands: Dict[int, str],
    wavenumbers: Optional[np.ndarray] = None,
    band_importance: Optional[np.ndarray] = None,
    show_peak_heights: bool = True,
    colors: Optional[Dict[int, str]] = None,
    fill_alpha: float = 0.2,
    colormap: str = "tab10",
    save_path: Optional[Path] = None,
    figure_size: Tuple[int, int] = (14, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Highlight and label marker bands in spectral data.
    
    Creates a visualization showing specific bands of chemical interest,
    with labels, highlight regions, and optional importance indicators.
    
    Args:
        spectrum: Spectral data (n_features,)
        marker_bands: Dictionary mapping feature indices to band names/descriptions
                     Example: {100: "C-H stretch", 200: "O-H stretch"}
        wavenumbers: Wavenumber array (n_features,), optional. If provided,
                    will be used for x-axis labels and band identification
        band_importance: Importance scores for marker bands, optional.
                        Used to color-code band importance
        show_peak_heights: Show numeric intensity values at peak locations
                          (default True)
        colors: Dictionary mapping band indices to explicit colors. If not
               provided, will be auto-assigned from colormap
        fill_alpha: Alpha transparency for band highlight regions (default 0.2)
        colormap: Matplotlib colormap for auto color assignment (default "tab10")
        save_path: Path to save PNG file (optional)
        figure_size: Figure dimensions in inches (default 14x6)
        dpi: Resolution for PNG export (default 300)
    
    Returns:
        matplotlib.pyplot.Figure
    
    Raises:
        ValueError: If marker_bands is empty
        ValueError: If spectrum is empty
    """
    # Validate inputs
    spectrum = np.asarray(spectrum).ravel()
    
    if spectrum.shape[0] == 0:
        raise ValueError("Spectrum cannot be empty")
    
    if not marker_bands:
        raise ValueError("marker_bands cannot be empty")
    
    # Validate marker band indices
    for idx in marker_bands.keys():
        if not (0 <= idx < spectrum.shape[0]):
            raise ValueError(f"Marker band index {idx} out of range [0, {spectrum.shape[0]-1}]")
    
    # Set up x-axis
    if wavenumbers is not None:
        wavenumbers = np.asarray(wavenumbers).ravel()
        if wavenumbers.shape[0] != spectrum.shape[0]:
            raise ValueError("Wavenumbers must match spectrum length")
        x_data = wavenumbers
        x_label = "Wavenumber (cm⁻¹)"
    else:
        x_data = np.arange(spectrum.shape[0])
        x_label = "Feature Index"
    
    # Prepare importance if provided
    if band_importance is not None:
        band_importance = np.asarray(band_importance).ravel()
        importance_norm = _normalize_importance(band_importance)
    else:
        importance_norm = None
    
    # Assign colors
    if colors is None:
        colors = {}
        cmap = plt.get_cmap(colormap)
        n_bands = len(marker_bands)
        for i, idx in enumerate(sorted(marker_bands.keys())):
            colors[idx] = cmap(i / max(n_bands - 1, 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Plot main spectrum
    ax.plot(x_data, spectrum, color="black", linewidth=2.5, label="Spectrum", zorder=2)
    
    # Highlight marker bands
    band_width = (x_data[-1] - x_data[0]) / len(x_data) * 2
    
    for idx, band_name in sorted(marker_bands.items()):
        x_val = x_data[idx]
        y_val = spectrum[idx]
        color = colors.get(idx, "blue")
        
        # Highlight region
        ax.axvspan(x_val - band_width/2, x_val + band_width/2, 
                  alpha=fill_alpha, color=color, zorder=1)
        
        # Mark peak with circle
        ax.plot(x_val, y_val, marker="o", markersize=10, color=color,
               markeredgecolor="black", markeredgewidth=2, zorder=4)
        
        # Add vertical line
        y_min = ax.get_ylim()[0]
        ax.plot([x_val, x_val], [y_min, y_val], color=color, linewidth=1.5,
               linestyle="--", alpha=0.5, zorder=1)
        
        # Create label text
        label_text = band_name
        if show_peak_heights:
            label_text += f"\nInt: {y_val:.2f}"
        if importance_norm is not None and idx < len(importance_norm):
            label_text += f"\nImp: {importance_norm[idx]:.2f}"
        
        # Add label
        ax.annotate(label_text, xy=(x_val, y_val), xytext=(0, 15),
                   textcoords="offset points", ha="center", fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                            alpha=0.4, edgecolor="black", linewidth=1),
                   arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                   zorder=5)
    
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Intensity", fontsize=12, fontweight="bold")
    ax.set_title("Marker Bands Visualization", fontsize=13, fontweight="bold")
    
    # Add legend for marker bands
    legend_elements = [
        mpatches.Patch(facecolor=colors[idx], edgecolor="black", linewidth=1.5,
                      label=band_name, alpha=0.7)
        for idx, band_name in sorted(marker_bands.items())
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
             title="Marker Bands", title_fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def get_band_statistics(
    spectrum: np.ndarray,
    importance: Optional[np.ndarray] = None,
    bands_of_interest: Optional[List[int]] = None,
    wavenumbers: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract band-level statistics from spectrum and importance.
    
    Args:
        spectrum: Spectral data (n_features,)
        importance: Importance scores (n_features,), optional
        bands_of_interest: List of band indices to analyze. If None, uses all
        wavenumbers: Wavenumber array (n_features,), optional
    
    Returns:
        Dictionary with band statistics including:
        - intensity: Peak intensity values
        - importance: Normalized importance (if provided)
        - wavenumber: Wavenumber values (if provided)
        - rank: Importance rank (if importance provided)
    """
    spectrum = np.asarray(spectrum).ravel()
    
    if bands_of_interest is None:
        bands_of_interest = list(range(len(spectrum)))
    
    stats = {}
    
    # Normalize importance if provided
    if importance is not None:
        importance = np.asarray(importance).ravel()
        importance_norm = _normalize_importance(importance)
        ranked_indices = np.argsort(importance)[::-1]
    else:
        importance_norm = None
        ranked_indices = None
    
    for band_idx in sorted(bands_of_interest):
        if band_idx >= len(spectrum):
            continue
        
        band_stats = {
            "intensity": float(spectrum[band_idx]),
        }
        
        if importance_norm is not None:
            band_stats["importance"] = float(importance_norm[band_idx])
            band_stats["importance_rank"] = int(np.where(ranked_indices == band_idx)[0][0]) + 1
        
        if wavenumbers is not None:
            band_stats["wavenumber"] = float(wavenumbers[band_idx])
        
        stats[f"band_{band_idx}"] = band_stats
    
    return stats
