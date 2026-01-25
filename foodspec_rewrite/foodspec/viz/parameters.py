"""Parameter map visualization for protocol configuration."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


def _get_default_values() -> Dict[str, Any]:
    """Get default parameter values for comparison."""
    return {
        "data.input": None,
        "data.format": "csv",
        "preprocess.recipe": None,
        "preprocess.steps": [],
        "qc.thresholds": {},
        "qc.metrics": [],
        "features.modules": [],
        "features.strategy": "auto",
        "model.estimator": None,
        "model.hyperparameters": None,
        "uncertainty.conformal": {},
        "interpretability.methods": [],
        "interpretability.marker_panel": None,
        "reporting.format": None,
        "reporting.sections": [],
        "export.bundle": False,
    }


def _flatten_protocol(protocol: Any) -> Dict[str, Any]:
    """Flatten protocol into dot-notation dictionary.
    
    Parameters
    ----------
    protocol : object
        Protocol object with nested attributes
    
    Returns
    -------
    dict
        Flattened protocol as {path: value}
    """
    result = {}
    
    # Map of attribute paths to extract
    paths = [
        ("data.input", lambda p: getattr(p.data, "input", None)),
        ("data.format", lambda p: getattr(p.data, "format", "csv")),
        ("preprocess.recipe", lambda p: getattr(p.preprocess, "recipe", None)),
        ("preprocess.steps", lambda p: getattr(p.preprocess, "steps", [])),
        ("qc.thresholds", lambda p: getattr(p.qc, "thresholds", {})),
        ("qc.metrics", lambda p: getattr(p.qc, "metrics", [])),
        ("features.modules", lambda p: getattr(p.features, "modules", [])),
        ("features.strategy", lambda p: getattr(p.features, "strategy", "auto")),
        ("model.estimator", lambda p: getattr(p.model, "estimator", None)),
        ("model.hyperparameters", lambda p: getattr(p.model, "hyperparameters", None)),
        ("uncertainty.conformal", lambda p: getattr(p.uncertainty, "conformal", {})),
        ("interpretability.methods", lambda p: getattr(p.interpretability, "methods", [])),
        ("interpretability.marker_panel", lambda p: getattr(p.interpretability, "marker_panel", None)),
        ("reporting.format", lambda p: getattr(p.reporting, "format", None)),
        ("reporting.sections", lambda p: getattr(p.reporting, "sections", [])),
        ("export.bundle", lambda p: getattr(p.export, "bundle", False)),
    ]
    
    for path, extractor in paths:
        try:
            value = extractor(protocol)
            result[path] = value
        except (AttributeError, TypeError):
            result[path] = None
    
    return result


def _identify_non_defaults(protocol: Any) -> Dict[str, bool]:
    """Identify which parameters differ from defaults.
    
    Parameters
    ----------
    protocol : object
        Protocol object
    
    Returns
    -------
    dict
        {parameter_path: is_non_default}
    """
    flattened = _flatten_protocol(protocol)
    defaults = _get_default_values()
    
    non_defaults = {}
    for key, value in flattened.items():
        default = defaults.get(key)
        is_non_default = value != default
        non_defaults[key] = is_non_default
    
    return non_defaults


def _build_hierarchy(flattened: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat dict to nested hierarchy.
    
    Parameters
    ----------
    flattened : dict
        Flat {path: value} dictionary
    
    Returns
    -------
    dict
        Nested hierarchy structure
    """
    hierarchy = {}
    
    for path, value in flattened.items():
        parts = path.split(".")
        current = hierarchy
        
        # Navigate/create hierarchy
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set final value
        current[parts[-1]] = value
    
    return hierarchy


def _format_value(value: Any) -> str:
    """Format value for display.
    
    Parameters
    ----------
    value : any
        Value to format
    
    Returns
    -------
    str
        Formatted string
    """
    if isinstance(value, bool):
        return "✓" if value else "✗"
    elif isinstance(value, list):
        if not value:
            return "[]"
        return f"[{len(value)} items]"
    elif isinstance(value, dict):
        if not value:
            return "{}"
        return f"{{{len(value)} keys}}"
    elif value is None:
        return "—"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        s = str(value)
        return s[:40] + "..." if len(s) > 40 else s


def plot_parameter_map(
    protocol: Any,
    save_path: Optional[Path] = None,
    figure_size: Tuple[float, float] = (14, 10),
    dpi: int = 300,
) -> plt.Figure:
    """Plot protocol parameters as hierarchical tree visualization.
    
    Parameters
    ----------
    protocol : object
        Protocol object with configuration parameters
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
    # Flatten and identify non-defaults
    flattened = _flatten_protocol(protocol)
    non_defaults = _identify_non_defaults(protocol)
    hierarchy = _build_hierarchy(flattened)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Color scheme
    default_color = "#E8E8E8"
    non_default_color = "#FFD700"
    text_color = "#333333"
    
    # Render hierarchy
    y_pos = 0.95
    y_step = 0.04
    
    def render_tree(node: Dict, depth: int = 0):
        nonlocal y_pos
        
        for key in sorted(node.keys()):
            value = node[key]
            indent = "  " * depth
            
            if isinstance(value, dict):
                # Category header
                text = f"{indent}▼ {key}"
                ax.text(0.02, y_pos, text, fontsize=11, fontweight="bold",
                       transform=ax.transAxes, family="monospace")
                y_pos -= y_step
                
                # Render subcategories
                render_tree(value, depth + 1)
            else:
                # Parameter line
                param_path = ".".join([k for k in node.keys()])
                full_path = ".".join([p for p in [k for k in range(depth)][::-1]])
                
                # Find full parameter path
                for flat_key in flattened.keys():
                    if flat_key.endswith(key):
                        full_path = flat_key
                        break
                
                is_non_default = non_defaults.get(full_path, False)
                bg_color = non_default_color if is_non_default else default_color
                
                # Format parameter display
                formatted_value = _format_value(value)
                param_text = f"{indent}{key}: {formatted_value}"
                
                # Draw background box
                bbox = dict(boxstyle="round,pad=0.5", facecolor=bg_color,
                          edgecolor="none", alpha=0.3)
                ax.text(0.02, y_pos, param_text, fontsize=10,
                       transform=ax.transAxes, family="monospace",
                       bbox=bbox, verticalalignment="top")
                
                y_pos -= y_step
    
    # Render from flattened view for clarity
    lines = []
    for path in sorted(flattened.keys()):
        value = flattened[path]
        is_non_default = non_defaults[path]
        
        bg_color = non_default_color if is_non_default else default_color
        formatted_value = _format_value(value)
        
        # Parse path into hierarchy levels
        parts = path.split(".")
        if len(parts) > 1:
            indent = "  "
            category = parts[0]
            param = parts[1]
            text = f"{indent}{param}: {formatted_value}"
        else:
            text = f"{path}: {formatted_value}"
        
        bbox = dict(boxstyle="round,pad=0.3", facecolor=bg_color,
                   edgecolor="none", alpha=0.3)
        ax.text(0.02, y_pos, text, fontsize=9.5,
               transform=ax.transAxes, family="monospace",
               bbox=bbox, verticalalignment="top")
        y_pos -= y_step
    
    # Add legend
    ax.text(0.02, 0.02, "🟨 Non-default parameter   ⬜ Default parameter",
           fontsize=9, transform=ax.transAxes, style="italic")
    
    # Configure axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Protocol Parameter Map", fontsize=16, fontweight="bold",
                pad=20)
    
    # Add summary stats
    non_default_count = sum(1 for v in non_defaults.values() if v)
    total_count = len(non_defaults)
    stats_text = f"Parameters: {non_default_count}/{total_count} non-default ({100*non_default_count//total_count}%)"
    ax.text(0.98, 0.02, stats_text, fontsize=9, transform=ax.transAxes,
           ha="right", style="italic", color="gray")
    
    plt.tight_layout()
    
    # Save outputs
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PNG
        png_path = save_path / "parameter_map.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        
        # Save JSON snapshot
        json_data = {
            "parameters": flattened,
            "non_defaults": {k: v for k, v in non_defaults.items() if v},
            "summary": {
                "total_parameters": total_count,
                "non_default_count": non_default_count,
                "non_default_percentage": 100 * non_default_count // total_count,
            }
        }
        json_path = save_path / "parameter_map.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
    
    return fig


def get_parameter_summary(protocol: Any) -> Dict[str, Any]:
    """Extract parameter summary statistics.
    
    Parameters
    ----------
    protocol : object
        Protocol object
    
    Returns
    -------
    dict
        Summary with counts and non-default parameters
    """
    flattened = _flatten_protocol(protocol)
    non_defaults = _identify_non_defaults(protocol)
    
    non_default_params = {k: flattened[k] for k, v in non_defaults.items() if v}
    
    return {
        "total_parameters": len(flattened),
        "non_default_parameters": len(non_default_params),
        "non_default_percentage": 100 * len(non_default_params) // len(flattened),
        "non_defaults": non_default_params,
        "all_parameters": flattened,
    }
