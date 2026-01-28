"""Paper-ready figure exporters and style presets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class FigureStyle(str, Enum):
    """Supported paper-ready figure styles."""

    NATURE = "nature"
    SCIENCE = "science"
    ELSEVIER = "elsevier"
    IEEE = "ieee"
    JOSS = "joss"


_STYLE_FONTS = {
    FigureStyle.NATURE: ["Times New Roman", "Times", "DejaVu Serif"],
    FigureStyle.SCIENCE: ["Helvetica", "Arial", "DejaVu Sans"],
    FigureStyle.ELSEVIER: ["Arial", "Helvetica", "DejaVu Sans"],
    FigureStyle.IEEE: ["Times New Roman", "Times", "DejaVu Serif"],
    FigureStyle.JOSS: ["DejaVu Sans", "Arial", "Helvetica"],
}

_SIZE_PRESETS = {
    "single": (3.5, 2.6),
    "double": (7.0, 2.8),
    "full": (7.2, 4.5),
}


def _apply_style(style: FigureStyle, font_size: int, dpi: int) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _STYLE_FONTS[style],
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )


@dataclass
class FigureExporter:
    """Export matplotlib figures with publication-ready presets."""

    style: FigureStyle = FigureStyle.JOSS
    size_preset: str = "single"
    dpi: int = 300
    formats: Tuple[str, ...] = ("png", "svg", "pdf")

    def export(self, fig, out_dir: Path | str, name: str) -> List[Path]:
        """Export figure in multiple formats with deterministic sizing."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _apply_style(self.style, font_size=9, dpi=self.dpi)
        size = _SIZE_PRESETS.get(self.size_preset, _SIZE_PRESETS["single"])
        fig.set_size_inches(size[0], size[1])
        outputs: List[Path] = []
        for fmt in self.formats:
            target = out_dir / f"{name}.{fmt}"
            fig.savefig(target, bbox_inches="tight")
            outputs.append(target)
        return outputs


def radar_plot(
    labels: Iterable[str],
    values: Iterable[float],
    *,
    title: str,
    seed: int = 0,
) -> Figure:
    """Create a deterministic radar plot."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    vals = np.asarray(list(values), dtype=float)
    if np.isnan(vals).any():
        vals = np.nan_to_num(vals, nan=0.0)
    if vals.size == 0:
        vals = rng.random(5)
    labels_list = list(labels) or [f"m{i + 1}" for i in range(len(vals))]
    angles = np.linspace(0, 2 * np.pi, len(vals), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    vals = np.concatenate([vals, vals[:1]])

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, color="#2a6fdb", linewidth=2)
    ax.fill(angles, vals, color="#2a6fdb", alpha=0.15)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels_list)
    ax.set_title(title)
    ax.set_ylim(0.0, max(1.0, float(vals.max())))
    return fig


__all__ = ["FigureExporter", "FigureStyle", "radar_plot"]
