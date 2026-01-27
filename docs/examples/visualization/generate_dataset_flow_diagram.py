"""Generate a simple dataset flow diagram for docs/assets/dataset_flow.png.

This script draws a schematic of the ingestion pipeline:
raw files -> read_spectra -> FoodSpectrumSet -> create_library -> HDF5 -> workflows.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _box(ax, xy, text, color="#e0e0e0"):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y), 1.8, 0.6, boxstyle="round,pad=0.1", fc=color, ec="k", lw=1
    )
    ax.add_patch(rect)
    ax.text(x + 0.9, y + 0.3, text, ha="center", va="center", fontsize=10)


def main():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    _box(ax, (0.2, 1.2), "Raw files\n(CSV/TXT/vendor)")
    _box(ax, (2.2, 1.2), "read_spectra\n+ metadata\n(cm⁻¹ axis)")
    _box(ax, (4.2, 1.2), "FoodSpectrumSet")
    _box(ax, (6.2, 1.2), "create_library")
    _box(ax, (8.2, 1.2), "HDF5 library\n(metadata + spectra)")
    _box(ax, (10.2, 1.2), "Workflows\n(preprocess →\nfeatures → models)")

    xs = [1.0, 3.0, 5.0, 7.0, 9.0]
    for x in xs:
        ax.annotate(
            "",
            xy=(x + 0.2, 1.5),
            xytext=(x - 0.2, 1.5),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

    ax.set_xlim(-0.2, 12)
    ax.set_ylim(0.5, 2.2)

    out_path = Path(__file__).parent.parent.parent / "assets" / "dataset_flow.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
