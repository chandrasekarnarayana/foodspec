"""Data lineage visualization for showing data flow and transformations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _hash_summary(hash_value: Optional[str], length: int = 8) -> str:
    """Get abbreviated hash display.
    
    Parameters
    ----------
    hash_value : str, optional
        Full hash value
    length : int
        Number of characters to display
    
    Returns
    -------
    str
        Abbreviated hash or placeholder
    """
    if not hash_value:
        return "—"
    return hash_value[:length] if len(hash_value) > length else hash_value


def _format_timestamp(ts: Optional[str]) -> str:
    """Format timestamp for display.
    
    Parameters
    ----------
    ts : str, optional
        ISO format timestamp
    
    Returns
    -------
    str
        Formatted timestamp
    """
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(ts)[:19]


def _extract_lineage_from_manifest(manifest: Any) -> Dict[str, Any]:
    """Extract lineage information from manifest object.
    
    Parameters
    ----------
    manifest : object
        Manifest object with data and processing history
    
    Returns
    -------
    dict
        Lineage structure with inputs, processing, outputs
    """
    lineage = {
        "inputs": [],
        "preprocessing": [],
        "processing": [],
        "outputs": [],
    }
    
    # Extract inputs
    if hasattr(manifest, "inputs"):
        inputs = manifest.inputs
        if isinstance(inputs, list):
            for inp in inputs:
                if isinstance(inp, dict):
                    lineage["inputs"].append(inp)
                elif hasattr(inp, "path"):
                    lineage["inputs"].append({
                        "path": getattr(inp, "path", "unknown"),
                        "hash": getattr(inp, "hash", None),
                        "timestamp": getattr(inp, "timestamp", None),
                    })
    
    # Extract preprocessing steps
    if hasattr(manifest, "preprocessing"):
        preprocessing = manifest.preprocessing
        if isinstance(preprocessing, list):
            lineage["preprocessing"] = preprocessing
        elif hasattr(preprocessing, "steps"):
            lineage["preprocessing"] = preprocessing.steps or []
    
    # Extract processing steps
    if hasattr(manifest, "processing"):
        processing = manifest.processing
        if isinstance(processing, list):
            lineage["processing"] = processing
        elif hasattr(processing, "steps"):
            lineage["processing"] = processing.steps or []
    
    # Extract outputs
    if hasattr(manifest, "outputs"):
        outputs = manifest.outputs
        if isinstance(outputs, list):
            for out in outputs:
                if isinstance(out, dict):
                    lineage["outputs"].append(out)
                elif hasattr(out, "path"):
                    lineage["outputs"].append({
                        "path": getattr(out, "path", "unknown"),
                        "hash": getattr(out, "hash", None),
                        "timestamp": getattr(out, "timestamp", None),
                    })
    
    return lineage


def plot_data_lineage(
    manifest: Any,
    save_path: Optional[Path] = None,
    figure_size: Tuple[float, float] = (16, 10),
    dpi: int = 300,
) -> plt.Figure:
    """Plot data lineage as horizontal flow chart.
    
    Parameters
    ----------
    manifest : object
        Manifest object with data lineage information
    save_path : Path, optional
        Directory to save PNG and JSON files
    figure_size : tuple
        Figure size in inches (width, height)
    dpi : int
        Resolution for PNG export
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    lineage = _extract_lineage_from_manifest(manifest)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Color scheme
    input_color = "#E3F2FD"
    preprocessing_color = "#F3E5F5"
    processing_color = "#FFF3E0"
    output_color = "#E8F5E9"
    arrow_color = "#666666"
    text_color = "#333333"
    
    # Layout parameters
    y_spacing = 0.15
    x_spacing = 0.20
    box_width = 0.12
    box_height = 0.08
    
    current_y = 0.85
    
    # Helper function to draw stage
    def draw_stage(title: str, items: List[Dict], y_pos: float, color: str, x_start: float = 0.05):
        nonlocal current_y
        
        # Stage title
        ax.text(x_start, y_pos, title, fontsize=12, fontweight="bold",
               transform=ax.transAxes, color=text_color)
        y_pos -= 0.05
        
        x_pos = x_start
        for i, item in enumerate(items[:3]):  # Limit to 3 items per stage
            # Item box
            rect = FancyBboxPatch(
                (x_pos, y_pos - box_height), box_width, box_height,
                boxstyle="round,pad=0.01", transform=ax.transAxes,
                facecolor=color, edgecolor=text_color, linewidth=1.5
            )
            ax.add_patch(rect)
            
            # Item content
            if isinstance(item, dict):
                path = item.get("path", "data")
                if "/" in str(path):
                    path = str(path).split("/")[-1]
                
                hash_val = item.get("hash", "")
                timestamp = item.get("timestamp", "")
            else:
                path = str(item)[:20]
                hash_val = ""
                timestamp = ""
            
            # Display text
            path_text = str(path)[:15] + "..." if len(str(path)) > 15 else str(path)
            ax.text(x_pos + 0.01, y_pos - 0.02, path_text, fontsize=8,
                   transform=ax.transAxes, family="monospace",
                   verticalalignment="center")
            
            # Hash and timestamp (small text below)
            if hash_val or timestamp:
                info_text = f"{_hash_summary(hash_val)} | {_format_timestamp(timestamp)}"
                ax.text(x_pos + 0.01, y_pos - 0.055, info_text, fontsize=7,
                       transform=ax.transAxes, family="monospace",
                       verticalalignment="top", style="italic", color="gray")
            
            x_pos += box_width + 0.01
        
        # Show overflow count
        if len(items) > 3:
            ax.text(x_pos, y_pos - 0.04, f"+{len(items) - 3}", fontsize=9,
                   transform=ax.transAxes, style="italic", color="gray")
        
        return y_pos - y_spacing
    
    # Draw stages
    stages = [
        ("Inputs", lineage["inputs"], input_color),
        ("Preprocessing", lineage["preprocessing"], preprocessing_color),
        ("Processing", lineage["processing"], processing_color),
        ("Outputs", lineage["outputs"], output_color),
    ]
    
    y_positions = {}
    current_y = 0.80
    
    for title, items, color in stages:
        y_positions[title] = current_y
        current_y = draw_stage(title, items, current_y, color)
    
    # Draw arrows between stages
    arrow_y = 0.75
    x_pos = 0.17
    arrow_length = 0.025
    
    for _ in range(3):  # 3 arrows for 4 stages
        arrow = FancyArrowPatch(
            (x_pos, arrow_y), (x_pos + arrow_length, arrow_y),
            transform=ax.transAxes, arrowstyle="->",
            mutation_scale=20, linewidth=2, color=arrow_color
        )
        ax.add_patch(arrow)
        x_pos += 0.20
    
    # Add legend
    legend_y = 0.08
    legend_items = [
        ("Input files", input_color),
        ("Preprocessing", preprocessing_color),
        ("Processing", processing_color),
        ("Outputs", output_color),
    ]
    
    legend_x = 0.05
    for label, color in legend_items:
        rect = FancyBboxPatch(
            (legend_x, legend_y), 0.02, 0.02,
            boxstyle="round,pad=0.002", transform=ax.transAxes,
            facecolor=color, edgecolor=text_color, linewidth=0.5
        )
        ax.add_patch(rect)
        ax.text(legend_x + 0.03, legend_y + 0.01, label, fontsize=8,
               transform=ax.transAxes, verticalalignment="center")
        legend_x += 0.20
    
    # Configure axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Data Lineage: Input → Processing → Output", fontsize=16,
                fontweight="bold", pad=20)
    
    # Add statistics
    total_items = sum(len(stage[1]) for stage in stages)
    stats_text = f"Total items: {total_items} | Stages: 4"
    ax.text(0.98, 0.02, stats_text, fontsize=9, transform=ax.transAxes,
           ha="right", style="italic", color="gray")
    
    plt.tight_layout()
    
    # Save outputs
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PNG
        png_path = save_path / "data_lineage.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        
        # Save JSON snapshot
        json_data = {
            "lineage": lineage,
            "summary": {
                "inputs": len(lineage["inputs"]),
                "preprocessing_steps": len(lineage["preprocessing"]),
                "processing_steps": len(lineage["processing"]),
                "outputs": len(lineage["outputs"]),
                "total_items": sum(
                    len(lineage[k]) for k in ["inputs", "preprocessing", "processing", "outputs"]
                ),
            }
        }
        json_path = save_path / "data_lineage.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
    
    return fig


def get_lineage_summary(manifest: Any) -> Dict[str, Any]:
    """Extract lineage summary statistics.
    
    Parameters
    ----------
    manifest : object
        Manifest object
    
    Returns
    -------
    dict
        Summary with stage counts and details
    """
    lineage = _extract_lineage_from_manifest(manifest)
    
    return {
        "input_count": len(lineage["inputs"]),
        "preprocessing_steps": len(lineage["preprocessing"]),
        "processing_steps": len(lineage["processing"]),
        "output_count": len(lineage["outputs"]),
        "total_items": sum(
            len(lineage[k]) for k in ["inputs", "preprocessing", "processing", "outputs"]
        ),
        "lineage": lineage,
    }
