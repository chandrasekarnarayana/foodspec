from __future__ import annotations

"""
Paper-ready figure presets for publication-quality matplotlib figures.

Provides standardized styling for JOSS, IEEE, Elsevier, and Nature publications
with deterministic figure sizes, fonts, line widths, and spacing. No hardcoded
colors—presets only affect layout and typography.

Examples
--------
Apply preset globally::

    from foodspec.viz.paper import FigurePreset, apply_figure_preset
    apply_figure_preset(FigurePreset.JOSS)
    # All figures now use JOSS styling

Use context manager for temporary preset::

    from foodspec.viz.paper import figure_context
    import matplotlib.pyplot as plt

    with figure_context(FigurePreset.IEEE):
        fig, ax = plt.subplots()
        # IEEE styling applied
        ax.plot([1, 2, 3], [1, 4, 9])
        save_figure(fig, "plot.png", dpi=300)
    # Original styling restored
"""


import contextlib
import copy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator

import matplotlib.pyplot as plt


class FigurePreset(Enum):
    """Publication venue figure presets.

    Defines layout standards (figure size, fonts, line widths) for different
    journals. No hardcoded colors; presets focus on typography and spacing.
    """

    JOSS = "joss"
    IEEE = "ieee"
    ELSEVIER = "elsevier"
    NATURE = "nature"


# Preset configurations: each preset defines rcParams overrides
_PRESET_CONFIGS: Dict[FigurePreset, Dict[str, Any]] = {
    # JOSS: Single-column figures (3-4 inches wide)
    FigurePreset.JOSS: {
        "figure.figsize": (3.5, 2.8),  # Single column, 87mm
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "patch.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
    # IEEE: Single-column (3.5 inches) or double-column (7 inches)
    FigurePreset.IEEE: {
        "figure.figsize": (3.5, 2.5),  # Default single-column
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "patch.linewidth": 0.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
    # ELSEVIER: Single-column (3.5 inches) or double-column (7.5 inches)
    FigurePreset.ELSEVIER: {
        "figure.figsize": (3.5, 2.6),  # Single column
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5.5,
        "patch.linewidth": 0.6,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
    # NATURE: Small, dense figures (89mm single-column, 183mm double)
    FigurePreset.NATURE: {
        "figure.figsize": (3.5, 2.8),  # 89mm single column
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "patch.linewidth": 0.4,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.25,
        "ytick.minor.size": 1.25,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
}


def apply_figure_preset(preset: FigurePreset | str) -> None:
    """Apply figure preset to matplotlib rcParams globally.

    Changes affect all subsequently created figures. Use `figure_context()`
    for temporary application.

    Parameters
    ----------
    preset : FigurePreset or str
        Preset enum or name: 'joss', 'ieee', 'elsevier', 'nature'.

    Examples
    --------
    Apply JOSS preset globally::

        apply_figure_preset(FigurePreset.JOSS)
        fig, ax = plt.subplots()  # Uses JOSS styling
        ax.plot([1, 2, 3])

    Apply by string::

        apply_figure_preset("ieee")
    """
    if isinstance(preset, str):
        preset_str = preset.lower()
        try:
            preset = FigurePreset(preset_str)
        except ValueError:
            valid = ", ".join(p.value for p in FigurePreset)
            raise ValueError(
                f"Invalid preset '{preset}'. Choose: {valid}"
            ) from None

    config = _PRESET_CONFIGS[preset]
    plt.rcParams.update(config)


def get_figure_preset_config(preset: FigurePreset | str) -> Dict[str, Any]:
    """Get rcParams config dict for a preset without applying it.

    Parameters
    ----------
    preset : FigurePreset or str
        Preset enum or name.

    Returns
    -------
    dict
        rcParams overrides for this preset.
    """
    if isinstance(preset, str):
        preset_str = preset.lower()
        try:
            preset = FigurePreset(preset_str)
        except ValueError:
            valid = ", ".join(p.value for p in FigurePreset)
            raise ValueError(
                f"Invalid preset '{preset}'. Choose: {valid}"
            ) from None

    return copy.deepcopy(_PRESET_CONFIGS[preset])


@contextlib.contextmanager
def figure_context(
    preset: FigurePreset | str,
) -> Generator[None, None, None]:
    """Context manager to temporarily apply a figure preset.

    Restores original rcParams after exiting the context.

    Parameters
    ----------
    preset : FigurePreset or str
        Preset enum or name.

    Yields
    ------
    None

    Examples
    --------
    Create IEEE figures temporarily::

        with figure_context(FigurePreset.IEEE):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            save_figure(fig, "ieee_plot.png")
        # Original rcParams restored
    """
    # Save original rcParams
    original_params = copy.deepcopy(plt.rcParams)

    try:
        # Apply preset
        apply_figure_preset(preset)
        yield
    finally:
        # Restore original rcParams
        plt.rcParams.update(original_params)


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 300,
    transparent: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.1,
) -> Path:
    """Save figure with publication-ready defaults.

    Handles DPI, transparency, and tight bounding box automatically.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Output file path. Extension determines format (.png, .pdf, .svg, etc.).
    dpi : int, default 300
        Resolution for raster formats (PNG, JPG).
    transparent : bool, default False
        Save with transparent background (useful for PDFs over colored backgrounds).
    bbox_inches : str, default 'tight'
        Bounding box strategy. 'tight' removes excess whitespace.
    pad_inches : float, default 0.1
        Padding in inches around the figure when using bbox_inches='tight'.

    Returns
    -------
    Path
        Absolute path where figure was saved.

    Examples
    --------
    Save with defaults (300 DPI, tight bbox)::

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        save_figure(fig, "plot.png")

    Save as PDF with transparency::

        save_figure(fig, "plot.pdf", transparent=True)

    Save high-resolution PNG::

        save_figure(fig, "hires.png", dpi=600)
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )

    return path


def list_presets() -> Dict[str, str]:
    """List available figure presets with descriptions.

    Returns
    -------
    dict
        Preset names and brief descriptions.
    """
    descriptions = {
        "joss": "Journal of Open Source Software (3.5 inch single column)",
        "ieee": "IEEE (3.5 inch single column, 7 inch double column)",
        "elsevier": "Elsevier (3.5 inch single column, 7.5 inch double column)",
        "nature": "Nature (3.5 inch single column, 7 inch double column)",
    }
    return descriptions
