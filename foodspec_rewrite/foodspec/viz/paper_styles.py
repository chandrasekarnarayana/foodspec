"""
Paper-ready matplotlib style presets for common venues.

Provides deterministic rcParams and sizing helpers so figures are reproducible
across reports and exports. Styles favor high-DPI outputs and predictable
font/line weights. Usage:

    from foodspec.viz.paper_styles import apply_paper_style
    apply_paper_style("joss")
"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt

# Base DPI for exports; callers can override but we keep a minimum.
DEFAULT_DPI = 300


PAPER_STYLES: Dict[str, Dict[str, object]] = {
    "joss": {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6.5, 4.0),
        "savefig.format": "png",
    },
    "ieee": {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "lines.linewidth": 1.0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (3.5, 2.5),
        "savefig.format": "png",
    },
    "elsevier": {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "lines.linewidth": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (7.0, 4.5),
        "savefig.format": "png",
    },
    "nature": {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "lines.linewidth": 1.0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (3.3, 2.8),
        "savefig.format": "png",
    },
}


def apply_paper_style(style: str = "joss", dpi: int = DEFAULT_DPI) -> Dict[str, object]:
    """Apply a paper-ready rcParam set and return the params used.

    Parameters
    ----------
    style : str
        One of PAPER_STYLES keys (e.g., "joss", "ieee", "elsevier", "nature").
    dpi : int
        DPI to enforce for saved figures. Minimum DEFAULT_DPI.
    """

    style_key = style.lower()
    if style_key not in PAPER_STYLES:
        raise ValueError(f"Unknown paper style '{style}'. Available: {sorted(PAPER_STYLES)}")

    params = dict(PAPER_STYLES[style_key])
    params["savefig.dpi"] = max(dpi, DEFAULT_DPI)
    plt.rcParams.update(params)
    return params


__all__ = ["apply_paper_style", "PAPER_STYLES", "DEFAULT_DPI"]
