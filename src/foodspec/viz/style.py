"""Shared plotting styles for FoodSpec visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import matplotlib as mpl

try:
    from foodspec.reporting.figures import FigureStyle
except Exception:  # pragma: no cover
    FigureStyle = None  # type: ignore[assignment]


_PAPER_FONTS: Dict[str, Iterable[str]] = {
    "nature": ["Times New Roman", "Times", "DejaVu Serif"],
    "science": ["Helvetica", "Arial", "DejaVu Sans"],
    "elsevier": ["Arial", "Helvetica", "DejaVu Sans"],
    "ieee": ["Times New Roman", "Times", "DejaVu Serif"],
    "joss": ["DejaVu Sans", "Arial", "Helvetica"],
}


@dataclass
class StyleConfig:
    """Style configuration for plots."""

    style: str = "default"
    dpi: int = 300
    font_size: int = 9


def apply_style(style: str | None = None, *, dpi: int = 300, font_size: int = 9) -> None:
    """Apply a shared matplotlib style.

    Parameters
    ----------
    style : str or None
        Style name ("default" or paper styles).
    dpi : int
        DPI for rendering.
    font_size : int
        Base font size.
    """
    style_name = (style or "default").lower()
    fonts = ["DejaVu Sans", "Arial", "Helvetica"]
    if FigureStyle is not None and isinstance(style, FigureStyle):  # type: ignore[arg-type]
        style_name = style.value
    if style_name in _PAPER_FONTS:
        fonts = list(_PAPER_FONTS[style_name])

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fonts,
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )


__all__ = ["StyleConfig", "apply_style"]
