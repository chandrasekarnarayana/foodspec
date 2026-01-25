"""Reproducibility badge generator for FoodSpec workflows."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch


def _extract_reproducibility_info(manifest: Any) -> Dict[str, Optional[str]]:
    """Extract reproducibility information from manifest.
    
    Parameters
    ----------
    manifest : object
        Manifest object with reproducibility metadata
    
    Returns
    -------
    dict
        {seed, protocol_hash, data_hash, env_hash}
    """
    info = {
        "seed": None,
        "protocol_hash": None,
        "data_hash": None,
        "env_hash": None,
    }
    
    # Extract seed
    if hasattr(manifest, "seed"):
        info["seed"] = manifest.seed
    elif hasattr(manifest, "config") and hasattr(manifest.config, "seed"):
        info["seed"] = manifest.config.seed
    
    # Extract protocol hash
    if hasattr(manifest, "protocol_hash"):
        info["protocol_hash"] = manifest.protocol_hash
    elif hasattr(manifest, "hashes") and hasattr(manifest.hashes, "protocol"):
        info["protocol_hash"] = manifest.hashes.protocol
    
    # Extract data hash
    if hasattr(manifest, "data_hash"):
        info["data_hash"] = manifest.data_hash
    elif hasattr(manifest, "hashes") and hasattr(manifest.hashes, "data"):
        info["data_hash"] = manifest.hashes.data
    
    # Extract env hash
    if hasattr(manifest, "env_hash"):
        info["env_hash"] = manifest.env_hash
    elif hasattr(manifest, "environment") and hasattr(manifest.environment, "hash"):
        info["env_hash"] = manifest.environment.hash
    
    return info


def _determine_badge_level(info: Dict[str, Optional[str]]) -> Tuple[str, str, str]:
    """Determine badge level and color based on reproducibility info.
    
    Parameters
    ----------
    info : dict
        Reproducibility information
    
    Returns
    -------
    tuple
        (level, color, status_text)
        level: "green", "yellow", "red"
        color: hex color code
        status_text: human-readable status
    """
    has_seed = info["seed"] is not None
    has_protocol = info["protocol_hash"] is not None
    has_data = info["data_hash"] is not None
    has_env = info["env_hash"] is not None
    
    # Critical items: seed, protocol_hash, data_hash
    has_critical = has_seed and has_protocol and has_data
    
    if has_critical and has_env:
        # All present - GREEN
        return "green", "#4CAF50", "Fully Reproducible"
    elif has_critical and not has_env:
        # Missing only env - YELLOW
        return "yellow", "#FFC107", "Partially Reproducible"
    else:
        # Missing critical items - RED
        return "red", "#F44336", "Not Reproducible"


def plot_reproducibility_badge(
    manifest: Any,
    save_path: Optional[Path] = None,
    figure_size: Tuple[float, float] = (4, 2),
    dpi: int = 150,
) -> plt.Figure:
    """Generate reproducibility badge showing workflow reproducibility status.
    
    Creates a visual badge indicating reproducibility level based on:
    - Seed presence
    - Protocol hash
    - Data hash
    - Environment hash
    
    Badge colors:
    - Green: All items present (fully reproducible)
    - Yellow: Missing environment hash only (partially reproducible)
    - Red: Missing critical items like seed or hashes (not reproducible)
    
    Parameters
    ----------
    manifest : object
        Manifest object with reproducibility metadata
    save_path : Path, optional
        Directory to save badge PNG
    figure_size : tuple
        Figure size in inches (width, height), default (4, 2)
    dpi : int
        Resolution for PNG export, default 150
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> fig = plot_reproducibility_badge(manifest, save_path="output/")
    >>> # Generates: output/reproducibility_badge.png
    """
    # Extract info and determine level
    info = _extract_reproducibility_info(manifest)
    level, color, status_text = _determine_badge_level(info)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Draw badge background
    badge_rect = FancyBboxPatch(
        (0.05, 0.2), 0.9, 0.6,
        boxstyle="round,pad=0.05",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="#333333",
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(badge_rect)
    
    # Add status text (main label)
    ax.text(0.5, 0.65, "REPRODUCIBILITY", fontsize=10, fontweight="bold",
           transform=ax.transAxes, ha="center", va="center",
           color="white", family="sans-serif")
    
    ax.text(0.5, 0.35, status_text, fontsize=12, fontweight="bold",
           transform=ax.transAxes, ha="center", va="center",
           color="white", family="sans-serif")
    
    # Add checkmarks/crosses for each component
    components = [
        ("Seed", info["seed"] is not None, 0.15),
        ("Protocol", info["protocol_hash"] is not None, 0.35),
        ("Data", info["data_hash"] is not None, 0.55),
        ("Env", info["env_hash"] is not None, 0.75),
    ]
    
    y_offset = 0.08
    for label, present, x_pos in components:
        symbol = "✓" if present else "✗"
        symbol_color = "#2E7D32" if present else "#C62828"
        
        # Draw small indicator circle
        circle = Circle((x_pos, y_offset), 0.025, transform=ax.transAxes,
                       facecolor="white", edgecolor="#333333", linewidth=1)
        ax.add_patch(circle)
        
        # Add checkmark/cross
        ax.text(x_pos, y_offset, symbol, fontsize=8, fontweight="bold",
               transform=ax.transAxes, ha="center", va="center",
               color=symbol_color)
        
        # Add label below
        ax.text(x_pos, y_offset - 0.04, label, fontsize=6,
               transform=ax.transAxes, ha="center", va="top",
               color="#666666")
    
    # Configure axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    plt.tight_layout()
    
    # Save output
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        png_path = save_path / "reproducibility_badge.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", 
                   facecolor="white", edgecolor="none")
    
    return fig


def get_reproducibility_status(manifest: Any) -> Dict[str, Any]:
    """Get reproducibility status information.
    
    Parameters
    ----------
    manifest : object
        Manifest object with reproducibility metadata
    
    Returns
    -------
    dict
        Status with level, components present, and details
    """
    info = _extract_reproducibility_info(manifest)
    level, color, status_text = _determine_badge_level(info)
    
    components_present = sum(1 for v in info.values() if v is not None)
    
    return {
        "level": level,
        "status": status_text,
        "color": color,
        "components": info,
        "components_present": components_present,
        "total_components": 4,
        "is_fully_reproducible": level == "green",
        "is_partially_reproducible": level == "yellow",
        "missing_components": [k for k, v in info.items() if v is None],
    }
