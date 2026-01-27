"""
Processing stages visualization module.

Provides visualization tools for analyzing multi-stage spectral preprocessing:
    from foodspec.viz import plot_processing_stages

    fig = plot_processing_stages(
        wavenumbers,
        stages_data,
        stage_names=["Raw", "Baseline Corrected", "Normalized"],
        zoom_regions=[(1000, 1200), (2800, 3000)]
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _validate_wavenumbers(wavenumbers: np.ndarray) -> int:
    """
    Validate wavenumber array.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber values (1D array)

    Returns
    -------
    int
        Length of wavenumber array

    Raises
    ------
    ValueError
        If wavenumbers is not 1D, empty, or has non-numeric values
    """
    if not isinstance(wavenumbers, np.ndarray):
        wavenumbers = np.asarray(wavenumbers)

    if wavenumbers.ndim != 1:
        raise ValueError(f"Wavenumbers must be 1D, got shape {wavenumbers.shape}")

    if len(wavenumbers) == 0:
        raise ValueError("Wavenumbers array is empty")

    if not np.issubdtype(wavenumbers.dtype, np.number):
        raise ValueError(f"Wavenumbers must be numeric, got dtype {wavenumbers.dtype}")

    return len(wavenumbers)


def _validate_spectral_stages(
    stages_data: Dict[str, np.ndarray],
    expected_length: int
) -> int:
    """
    Validate spectral data for all stages.

    Parameters
    ----------
    stages_data : Dict[str, np.ndarray]
        Dictionary mapping stage names to spectral arrays
    expected_length : int
        Expected length of each spectrum

    Returns
    -------
    int
        Number of stages

    Raises
    ------
    ValueError
        If stages have mismatched lengths or invalid shapes
    """
    if not isinstance(stages_data, dict):
        raise ValueError("stages_data must be a dictionary")

    if len(stages_data) == 0:
        raise ValueError("stages_data dictionary is empty")

    for stage_name, spectrum in stages_data.items():
        if not isinstance(spectrum, np.ndarray):
            spectrum = np.asarray(spectrum)

        if spectrum.ndim != 1:
            raise ValueError(
                f"Stage '{stage_name}' spectrum must be 1D, got shape {spectrum.shape}"
            )

        if len(spectrum) != expected_length:
            raise ValueError(
                f"Stage '{stage_name}' has length {len(spectrum)}, "
                f"expected {expected_length}"
            )

    return len(stages_data)


def _get_stage_colors(
    n_stages: int,
    colormap: str = "viridis"
) -> List:
    """
    Generate colors for different stages.

    Parameters
    ----------
    n_stages : int
        Number of stages
    colormap : str
        Matplotlib colormap name

    Returns
    -------
    List
        List of RGB color tuples
    """
    if n_stages == 1:
        return [plt.colormaps[colormap](0.5)]

    cmap = plt.colormaps[colormap]
    indices = np.linspace(0, 1, n_stages)
    return [cmap(idx) for idx in indices]


def _extract_zoom_regions(
    wavenumbers: np.ndarray,
    zoom_regions: Optional[List[Tuple[float, float]]]
) -> List[Tuple[int, int]]:
    """
    Convert wavenumber ranges to array indices.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber array
    zoom_regions : Optional[List[Tuple[float, float]]]
        List of (min_wavenum, max_wavenum) tuples

    Returns
    -------
    List[Tuple[int, int]]
        List of (start_idx, end_idx) tuples

    Raises
    ------
    ValueError
        If zoom regions are invalid
    """
    if zoom_regions is None:
        return []

    if not isinstance(zoom_regions, list):
        raise ValueError("zoom_regions must be a list of tuples")

    if len(zoom_regions) > 3:
        raise ValueError("Maximum 3 zoom regions allowed")

    indices = []
    for min_wv, max_wv in zoom_regions:
        if min_wv >= max_wv:
            raise ValueError(f"Invalid zoom region: {min_wv} >= {max_wv}")

        start_idx = np.argmin(np.abs(wavenumbers - min_wv))
        end_idx = np.argmin(np.abs(wavenumbers - max_wv))

        indices.append((min(start_idx, end_idx), max(start_idx, end_idx)))

    return indices


def plot_processing_stages(
    wavenumbers: np.ndarray,
    stages_data: Dict[str, np.ndarray],
    stage_names: Optional[List[str]] = None,
    stage_colors: Optional[List] = None,
    zoom_regions: Optional[List[Tuple[float, float]]] = None,
    colormap: str = "viridis",
    alpha: float = 0.7,
    linewidth: float = 1.5,
    show_grid: bool = True,
    title: Optional[str] = None,
    figure_size: Tuple[float, float] = (16, 10),
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> Figure:
    """
    Visualize multi-stage spectral preprocessing with optional zoom windows.

    Creates a main plot showing all preprocessing stages overlaid, with optional
    inset zoom windows highlighting specific wavenumber regions. Each stage is
    displayed with distinct coloring and optional preprocessing names.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber values (1D array)
    stages_data : Dict[str, np.ndarray]
        Dictionary mapping stage identifiers to spectral arrays (1D).
        Example: {"raw": raw_spectrum, "baseline": baseline_spectrum}
    stage_names : Optional[List[str]]
        Display names for each stage. If None, uses dictionary keys.
        Example: ["Raw", "Baseline Corrected", "Normalized"]
    stage_colors : Optional[List]
        List of colors for each stage. If None, uses colormap.
    zoom_regions : Optional[List[Tuple[float, float]]]
        List of (min_wavenumber, max_wavenumber) regions to zoom.
        Maximum 3 regions. Each will create an inset plot.
        Example: [(1000, 1200), (2800, 3000)]
    colormap : str, default "viridis"
        Colormap for stage colors (if stage_colors not provided)
    alpha : float, default 0.7
        Transparency for spectral lines (0-1)
    linewidth : float, default 1.5
        Line width for spectra
    show_grid : bool, default True
        Whether to show grid on main plot
    title : Optional[str]
        Plot title. If None, auto-generated.
    figure_size : Tuple[float, float]
        Figure size as (width, height) in inches
    save_path : Optional[Path]
        Path to save PNG. If None, not saved.
    dpi : int, default 300
        DPI for PNG export

    Returns
    -------
    Figure
        Matplotlib Figure object

    Raises
    ------
    ValueError
        If inputs are invalid (empty arrays, shape mismatch, etc.)

    Examples
    --------
    >>> import numpy as np
    >>> from pathlib import Path
    >>> from foodspec.viz import plot_processing_stages
    >>>
    >>> # Generate sample spectral data
    >>> wavenumbers = np.linspace(400, 4000, 1000)
    >>> raw = np.sin(wavenumbers / 200) + np.random.normal(0, 0.1, 1000)
    >>> baseline_corrected = raw - np.mean(raw)
    >>> normalized = baseline_corrected / np.std(baseline_corrected)
    >>>
    >>> # Create visualization
    >>> fig = plot_processing_stages(
    ...     wavenumbers,
    ...     stages_data={
    ...         "raw": raw,
    ...         "baseline": baseline_corrected,
    ...         "normalized": normalized
    ...     },
    ...     stage_names=["Raw", "Baseline Corrected", "Normalized"],
    ...     zoom_regions=[(1000, 1200), (2800, 3000)],
    ...     save_path=Path("processing.png")
    ... )
    >>> plt.close(fig)
    """
    # Validate inputs
    n_wv = _validate_wavenumbers(wavenumbers)
    n_stages = _validate_spectral_stages(stages_data, n_wv)

    # Get stage names
    if stage_names is None:
        stage_names = list(stages_data.keys())
    elif len(stage_names) != n_stages:
        raise ValueError(
            f"stage_names length {len(stage_names)} doesn't match "
            f"stages_data length {n_stages}"
        )

    # Get colors
    if stage_colors is None:
        stage_colors = _get_stage_colors(n_stages, colormap)
    elif len(stage_colors) != n_stages:
        raise ValueError(
            f"stage_colors length {len(stage_colors)} doesn't match "
            f"n_stages {n_stages}"
        )

    # Extract zoom regions
    zoom_indices = _extract_zoom_regions(wavenumbers, zoom_regions)

    # Set title
    if title is None:
        title = f"Spectral Processing Stages ({n_stages} stages)"

    # Create figure layout based on zoom regions
    if zoom_indices:
        # Create grid: main plot + inset plots
        n_insets = len(zoom_indices)
        if n_insets == 1:
            fig = plt.figure(figsize=figure_size, dpi=dpi)
            ax_main = plt.subplot(1, 2, 1)
            axes_inset = [plt.subplot(1, 2, 2)]
        elif n_insets == 2:
            fig = plt.figure(figsize=figure_size, dpi=dpi)
            ax_main = plt.subplot(2, 2, 1)
            axes_inset = [plt.subplot(2, 2, 2), plt.subplot(2, 2, 3)]
        else:  # n_insets == 3
            fig = plt.figure(figsize=figure_size, dpi=dpi)
            ax_main = plt.subplot(2, 3, (1, 4))
            axes_inset = [plt.subplot(2, 3, 2), plt.subplot(2, 3, 3), plt.subplot(2, 3, 5)]
    else:
        fig, ax_main = plt.subplots(figsize=figure_size, dpi=dpi)
        axes_inset = []

    # Plot main spectral stages
    for (stage_key, spectrum), stage_name, color in zip(
        stages_data.items(), stage_names, stage_colors
    ):
        ax_main.plot(
            wavenumbers,
            spectrum,
            label=stage_name,
            color=color,
            alpha=alpha,
            linewidth=linewidth
        )

    # Formatting main plot
    ax_main.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Intensity", fontsize=12, fontweight="bold")
    ax_main.set_title(title, fontsize=13, fontweight="bold")
    ax_main.legend(loc="best", framealpha=0.95, fontsize=10)

    if show_grid:
        ax_main.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    ax_main.set_xlim(wavenumbers.min(), wavenumbers.max())

    # Add zoom regions as rectangles on main plot
    if zoom_indices:
        for (start_idx, end_idx), ax_inset in zip(zoom_indices, axes_inset):
            # Get wavenumber range for this zoom region
            wv_min = wavenumbers[start_idx]
            wv_max = wavenumbers[end_idx]

            # Add rectangle to main plot
            rect = mpatches.Rectangle(
                (wv_min, ax_main.get_ylim()[0]),
                wv_max - wv_min,
                ax_main.get_ylim()[1] - ax_main.get_ylim()[0],
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                alpha=0.6
            )
            ax_main.add_patch(rect)

            # Plot zoom window
            for (stage_key, spectrum), stage_name, color in zip(
                stages_data.items(), stage_names, stage_colors
            ):
                zoom_spectrum = spectrum[start_idx:end_idx+1]
                zoom_wv = wavenumbers[start_idx:end_idx+1]
                ax_inset.plot(
                    zoom_wv,
                    zoom_spectrum,
                    label=stage_name,
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth
                )

            # Format zoom window
            ax_inset.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
            ax_inset.set_ylabel("Intensity", fontsize=10)
            ax_inset.set_title(
                f"Zoom: {wv_min:.0f}-{wv_max:.0f} cm⁻¹",
                fontsize=11,
                fontweight="bold"
            )

            if show_grid:
                ax_inset.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_preprocessing_comparison(
    wavenumbers: np.ndarray,
    before_spectrum: np.ndarray,
    after_spectrum: np.ndarray,
    preprocessing_name: str = "Preprocessing",
    zoom_regions: Optional[List[Tuple[float, float]]] = None,
    color_before: str = "steelblue",
    color_after: str = "coral",
    alpha: float = 0.8,
    linewidth: float = 2,
    show_difference: bool = True,
    title: Optional[str] = None,
    figure_size: Tuple[float, float] = (14, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> Figure:
    """
    Compare before and after a single preprocessing step.

    Parameters
    ----------
    wavenumbers : np.ndarray
        Wavenumber values (1D)
    before_spectrum : np.ndarray
        Spectrum before preprocessing
    after_spectrum : np.ndarray
        Spectrum after preprocessing
    preprocessing_name : str
        Name of preprocessing step (e.g., "Baseline Correction")
    zoom_regions : Optional[List[Tuple[float, float]]]
        Zoom regions as (min_wv, max_wv) tuples
    color_before : str
        Color for before spectrum
    color_after : str
        Color for after spectrum
    alpha : float
        Line transparency
    linewidth : float
        Line width
    show_difference : bool
        Whether to show difference subplot
    title : Optional[str]
        Plot title
    figure_size : Tuple[float, float]
        Figure size
    save_path : Optional[Path]
        Save path for PNG
    dpi : int
        DPI for PNG

    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    # Validate inputs
    _validate_wavenumbers(wavenumbers)

    if not isinstance(before_spectrum, np.ndarray):
        before_spectrum = np.asarray(before_spectrum)
    if not isinstance(after_spectrum, np.ndarray):
        after_spectrum = np.asarray(after_spectrum)

    if len(before_spectrum) != len(wavenumbers):
        raise ValueError("before_spectrum length doesn't match wavenumbers")
    if len(after_spectrum) != len(wavenumbers):
        raise ValueError("after_spectrum length doesn't match wavenumbers")

    if title is None:
        title = f"Preprocessing Comparison: {preprocessing_name}"

    # Create figure
    if show_difference:
        fig, axes = plt.subplots(2, 1, figsize=figure_size, dpi=dpi)
    else:
        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
        axes = [ax]

    # Main comparison plot
    ax = axes[0]
    ax.plot(
        wavenumbers,
        before_spectrum,
        label="Before",
        color=color_before,
        alpha=alpha,
        linewidth=linewidth
    )
    ax.plot(
        wavenumbers,
        after_spectrum,
        label="After",
        color=color_after,
        alpha=alpha,
        linewidth=linewidth
    )
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Intensity", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Difference plot
    if show_difference:
        ax = axes[1]
        difference = after_spectrum - before_spectrum
        ax.fill_between(wavenumbers, difference, alpha=0.5, color="green", label="Difference")
        ax.plot(wavenumbers, difference, color="darkgreen", linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Δ Intensity", fontsize=11, fontweight="bold")
        ax.set_title("Change Applied", fontsize=11, fontweight="bold")
        ax.legend(loc="best", framealpha=0.95, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def get_processing_statistics(
    stages_data: Dict[str, np.ndarray],
    wavenumbers: Optional[np.ndarray] = None
) -> Dict:
    """
    Extract statistics for each preprocessing stage.

    Parameters
    ----------
    stages_data : Dict[str, np.ndarray]
        Dictionary of stage name to spectrum
    wavenumbers : Optional[np.ndarray]
        Wavenumber array (for region-specific statistics)

    Returns
    -------
    Dict
        Statistics dictionary with keys for each stage
    """
    stats = {}

    for stage_name, spectrum in stages_data.items():
        if not isinstance(spectrum, np.ndarray):
            spectrum = np.asarray(spectrum)

        stats[stage_name] = {
            "mean": float(np.mean(spectrum)),
            "std": float(np.std(spectrum)),
            "min": float(np.min(spectrum)),
            "max": float(np.max(spectrum)),
            "median": float(np.median(spectrum)),
            "q25": float(np.percentile(spectrum, 25)),
            "q75": float(np.percentile(spectrum, 75)),
            "range": float(np.max(spectrum) - np.min(spectrum)),
        }

    return stats
