"""
Generate publication-quality workflow diagram for JOSS paper.
Clean, minimalistic design suitable for journal publication.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# JOSS-style color palette: minimal, professional - SINGLE ACCENT COLOR
COLOR_PRIMARY = "#2E86AB"  # Teal blue - ONLY accent color
COLOR_GRAY = "#6C757D"  # Medium gray for text
COLOR_LIGHT_GRAY = "#E9ECEF"  # Light gray for badges
COLOR_BLACK = "#212529"  # Near-black for titles


def create_workflow_diagram():
    """Generate FoodSpec workflow diagram for JOSS paper."""

    fig, ax = plt.subplots(figsize=(16, 7), dpi=300, facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Title at top - exact casing from spec
    ax.text(
        8,
        6.5,
        "FoodSpec workflow: protocol-driven Raman/FTIR analysis",
        ha="center",
        va="top",
        fontsize=16,
        weight="bold",
        color=COLOR_BLACK,
        family="sans-serif",
    )

    # Define 6 pipeline stages - exact wording from spec
    stages = [
        {
            "x": 0.5,
            "y": 3.8,
            "width": 2.2,
            "height": 1.8,
            "title": "Data ingestion",
            "subtitle": "Import Raman/FTIR spectra",
            "details": ["Sample metadata", "Acquisition descriptors"],
            "badge": None,
        },
        {
            "x": 3.0,
            "y": 3.8,
            "width": 2.2,
            "height": 1.8,
            "title": "QC & Cleaning",
            "subtitle": "Quality checks",
            "details": ["Outliers, noise flags", "Replicate consistency"],
            "badge": None,
        },
        {
            "x": 5.5,
            "y": 3.8,
            "width": 2.2,
            "height": 1.8,
            "title": "Preprocessing",
            "subtitle": "Protocol-driven",
            "details": ["Baseline correction", "Smoothing, normalization", "Trimming, spike removal"],
            "badge": "YAML protocol",
        },
        {
            "x": 8.0,
            "y": 3.8,
            "width": 2.2,
            "height": 1.8,
            "title": "Chemometrics",
            "subtitle": "",
            "details": ["PCA, PLS", "Clustering", "Outlier detection"],
            "badge": None,
        },
        {
            "x": 10.5,
            "y": 3.8,
            "width": 2.2,
            "height": 1.8,
            "title": "Machine learning",
            "subtitle": "",
            "details": ["Classification/regression", "Nested CV", "Calibration diagnostics"],
            "badge": "Leakage-aware",
        },
        {
            "x": 13.0,
            "y": 3.8,
            "width": 2.5,
            "height": 1.8,
            "title": "Reporting & Export",
            "subtitle": "",
            "details": ["Metrics, plots, artifacts", "HDF5 + metadata"],
            "badge": "FAIR-aligned",
        },
    ]

    # Draw stage boxes
    for stage in stages:
        # Main box
        box = FancyBboxPatch(
            (stage["x"], stage["y"]),
            stage["width"],
            stage["height"],
            boxstyle="round,pad=0.1",
            edgecolor=COLOR_PRIMARY,
            facecolor="white",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(box)

        # Title
        ax.text(
            stage["x"] + stage["width"] / 2,
            stage["y"] + stage["height"] - 0.25,
            stage["title"],
            ha="center",
            va="top",
            fontsize=11,
            weight="bold",
            color=COLOR_BLACK,
            family="sans-serif",
        )

        # Subtitle
        ax.text(
            stage["x"] + stage["width"] / 2,
            stage["y"] + stage["height"] - 0.55,
            stage["subtitle"],
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color=COLOR_GRAY,
            family="sans-serif",
        )

        # Details
        y_pos = stage["y"] + stage["height"] - 0.9
        for detail in stage["details"]:
            ax.text(
                stage["x"] + stage["width"] / 2,
                y_pos,
                detail,
                ha="center",
                va="top",
                fontsize=7.5,
                color=COLOR_GRAY,
                family="sans-serif",
            )
            y_pos -= 0.25

        # Badge if present - use PRIMARY color only
        if stage["badge"]:
            badge_box = FancyBboxPatch(
                (stage["x"] + 0.15, stage["y"] + 0.08),
                stage["width"] - 0.3,
                0.35,
                boxstyle="round,pad=0.05",
                edgecolor=COLOR_PRIMARY,
                facecolor=COLOR_LIGHT_GRAY,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(badge_box)
            ax.text(
                stage["x"] + stage["width"] / 2,
                stage["y"] + 0.25,
                stage["badge"],
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
                color=COLOR_PRIMARY,
                family="sans-serif",
            )

    # Draw arrows between stages
    arrow_y = 4.7
    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + stages[i]["width"]
        x_end = stages[i + 1]["x"]
        arrow = FancyArrowPatch(
            (x_start + 0.05, arrow_y),
            (x_end - 0.05, arrow_y),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2,
            color=COLOR_PRIMARY,
            zorder=1,
        )
        ax.add_patch(arrow)

    # Reproducibility band at bottom - exact wording from spec
    repro_y = 2.8
    repro_box = FancyBboxPatch(
        (0.5, repro_y),
        15,
        0.6,
        boxstyle="round,pad=0.08",
        edgecolor=COLOR_PRIMARY,
        facecolor=COLOR_LIGHT_GRAY,
        linewidth=1.5,
        alpha=0.7,
        zorder=1,
    )
    ax.add_patch(repro_box)

    ax.text(
        8,
        repro_y + 0.3,
        "Reproducibility",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color=COLOR_PRIMARY,
        family="sans-serif",
    )

    ax.text(
        8,
        repro_y + 0.05,
        "configuration snapshot • random seeds • version stamping • run artifacts",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=COLOR_GRAY,
        family="sans-serif",
    )

    # Interface labels at bottom - clean, minimal
    ax.text(
        1.5,
        2.2,
        "Python API",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_LIGHT_GRAY, edgecolor=COLOR_PRIMARY, linewidth=1.5),
        color=COLOR_PRIMARY,
        family="sans-serif",
    )

    ax.text(
        3.5,
        2.2,
        "CLI",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_LIGHT_GRAY, edgecolor=COLOR_PRIMARY, linewidth=1.5),
        color=COLOR_PRIMARY,
        family="sans-serif",
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Generating FoodSpec workflow diagram for JOSS paper...")
    fig = create_workflow_diagram()

    output_path = "../figures/workflow.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ Workflow diagram saved to: {output_path}")
    print("  Resolution: 300 DPI")
    print("  Format: PNG with white background")
    print("  Optimized for JOSS publication standards")

    plt.close()
