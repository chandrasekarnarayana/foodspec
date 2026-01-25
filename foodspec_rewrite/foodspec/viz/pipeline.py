"""
FoodSpec workflow DAG visualizer.

Generates deterministic, annotated pipeline visualizations showing:
  Data → Preprocess → QC → Features → Model → Calibration → Conformal → Report → Bundle

Supports SVG and PNG export with reproducible layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from foodspec.core.protocol import ProtocolV2


def _build_pipeline_graph(protocol: ProtocolV2) -> Tuple[nx.DiGraph, Dict[str, Dict[str, Any]]]:
    """Build a directed graph representing the FoodSpec pipeline.
    
    Parameters
    ----------
    protocol : ProtocolV2
        Protocol defining pipeline stages and parameters
    
    Returns
    -------
    graph : nx.DiGraph
        Directed graph with nodes for each stage
    node_attrs : dict
        Attributes for each node (parameters, enabled status)
    """
    graph = nx.DiGraph()
    node_attrs = {}
    
    # Define pipeline stages in order
    stages = [
        ("Data", "load"),
        ("Preprocess", "preprocess"),
        ("QC", "qc"),
        ("Features", "features"),
        ("Model", "model"),
        ("Calibration", "uncertainty"),
        ("Conformal", "uncertainty"),
        ("Interpret", "interpretability"),
        ("Report", "reporting"),
        ("Bundle", "export"),
    ]
    
    # Add nodes with stage-specific attributes
    for i, (stage_name, stage_key) in enumerate(stages):
        graph.add_node(i, label=stage_name, stage_key=stage_key)
        
        # Extract parameters for this stage
        params = {}
        enabled = True
        
        if stage_key == "load":
            params = {
                "input": str(protocol.data.input),
                "format": protocol.data.format or "raw",
            }
        elif stage_key == "preprocess":
            enabled = bool(protocol.preprocess.recipe or protocol.preprocess.steps)
            if protocol.preprocess.recipe:
                params["recipe"] = protocol.preprocess.recipe
            if protocol.preprocess.steps:
                params["steps"] = len(protocol.preprocess.steps)
        elif stage_key == "qc":
            enabled = bool(protocol.qc.thresholds or protocol.qc.metrics)
            if protocol.qc.metrics:
                params["metrics"] = len(protocol.qc.metrics)
        elif stage_key == "features":
            enabled = bool(protocol.features.modules or protocol.features.strategy not in {"auto", ""})
            if protocol.features.strategy:
                params["strategy"] = protocol.features.strategy
            if protocol.features.modules:
                params["modules"] = len(protocol.features.modules)
        elif stage_key == "model":
            enabled = protocol.model.estimator not in {"", None}
            params["estimator"] = protocol.model.estimator or "baseline"
            if protocol.model.hyperparameters:
                params["tuning"] = "grid" if protocol.model.hyperparameters else "none"
        elif stage_key == "uncertainty":
            if stage_name == "Calibration":
                enabled = bool(protocol.uncertainty.conformal.get("calibration"))
                if enabled:
                    cal_method = protocol.uncertainty.conformal.get("calibration", {})
                    if isinstance(cal_method, dict):
                        params["method"] = cal_method.get("method", "platt")
            elif stage_name == "Conformal":
                enabled = bool(protocol.uncertainty.conformal.get("conformal"))
                if enabled:
                    conf_method = protocol.uncertainty.conformal.get("conformal", {})
                    if isinstance(conf_method, dict):
                        params["method"] = conf_method.get("method", "mondrian")
                        alpha = conf_method.get("alpha")
                        if alpha:
                            params["alpha"] = f"{alpha:.2f}"
        elif stage_key == "interpretability":
            enabled = bool(protocol.interpretability.methods or protocol.interpretability.marker_panel)
            if protocol.interpretability.methods:
                params["methods"] = len(protocol.interpretability.methods)
        elif stage_key == "reporting":
            enabled = protocol.reporting.format != ""
            params["format"] = protocol.reporting.format or "markdown"
            if protocol.reporting.sections:
                params["sections"] = len(protocol.reporting.sections)
        elif stage_key == "export":
            enabled = bool(protocol.export.bundle)
            if protocol.export.bundle:
                params["bundle"] = "enabled"
        
        node_attrs[stage_name] = {
            "params": params,
            "enabled": enabled,
            "stage_key": stage_key,
        }
    
    # Add edges connecting stages
    for i in range(len(stages) - 1):
        graph.add_edge(i, i + 1)
    
    return graph, node_attrs


def _compute_deterministic_layout(
    graph: nx.DiGraph,
    seed: int = 42,
    k: float = 2.0,
    iterations: int = 50,
) -> Dict[int, Tuple[float, float]]:
    """Compute deterministic node positions using spring layout with seed.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Graph to layout
    seed : int
        Random seed for reproducibility
    k : float
        Optimal distance between nodes
    iterations : int
        Number of iterations
    
    Returns
    -------
    pos : dict
        Node positions {node_id: (x, y)}
    """
    # Use spring layout with fixed seed for determinism
    np.random.seed(seed)
    pos = nx.spring_layout(
        graph,
        k=k,
        iterations=iterations,
        seed=seed,
    )
    return pos


def plot_pipeline_dag(
    protocol: ProtocolV2,
    save_path: Optional[Path] = None,
    seed: int = 42,
    figure_size: Tuple[float, float] = (16, 10),
    dpi: int = 300,
) -> plt.Figure:
    """Plot the FoodSpec workflow DAG with deterministic layout.
    
    Generates a directed graph showing pipeline stages and annotated parameters.
    Exports to SVG and PNG if save_path provided.
    
    Parameters
    ----------
    protocol : ProtocolV2
        Protocol defining pipeline stages and parameters
    save_path : Path, optional
        Directory to save SVG and PNG files. If None, no export.
    seed : int
        Random seed for deterministic layout (default: 42)
    figure_size : tuple
        Figure dimensions in inches (default: 16x10)
    dpi : int
        DPI for PNG export (default: 300)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> protocol = ProtocolV2.load("pipeline.yaml")
    >>> fig = plot_pipeline_dag(protocol, save_path=Path("/tmp/output"))
    # Saves to /tmp/output/pipeline_dag.svg and .png
    """
    # Build graph and get node attributes
    graph, node_attrs = _build_pipeline_graph(protocol)
    
    # Compute deterministic layout
    pos = _compute_deterministic_layout(graph, seed=seed)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Separate nodes by enabled status
    enabled_nodes = [i for i, (stage_name, _) in enumerate(
        [("Data", "load"), ("Preprocess", "preprocess"), ("QC", "qc"), 
         ("Features", "features"), ("Model", "model"), ("Calibration", "uncertainty"), 
         ("Conformal", "uncertainty"), ("Interpret", "interpretability"), 
         ("Report", "reporting"), ("Bundle", "export")]
    ) if node_attrs[stage_name]["enabled"]]
    
    disabled_nodes = [i for i in range(len(node_attrs)) if i not in enabled_nodes]
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        width=2,
        connectionstyle="arc3,rad=0.1",
    )
    
    # Draw enabled nodes (green)
    if enabled_nodes:
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=enabled_nodes,
            node_color="#90EE90",
            node_size=3000,
            ax=ax,
        )
    
    # Draw disabled nodes (light gray)
    if disabled_nodes:
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=disabled_nodes,
            node_color="#D3D3D3",
            node_size=3000,
            ax=ax,
        )
    
    # Draw node labels
    labels = {i: node_attrs[stage_name]["params"].get("label", stage_name)
              for i, stage_name in enumerate(
                  ["Data", "Preprocess", "QC", "Features", "Model", 
                   "Calibration", "Conformal", "Interpret", "Report", "Bundle"])}
    
    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=10,
        font_weight="bold",
        ax=ax,
    )
    
    # Add parameter annotations below nodes
    for i, (stage_name, _) in enumerate(
        [("Data", "load"), ("Preprocess", "preprocess"), ("QC", "qc"), 
         ("Features", "features"), ("Model", "model"), ("Calibration", "uncertainty"), 
         ("Conformal", "uncertainty"), ("Interpret", "interpretability"), 
         ("Report", "reporting"), ("Bundle", "export")]
    ):
        attrs = node_attrs[stage_name]
        params = attrs["params"]
        
        # Create parameter text
        param_text = stage_name
        if params:
            param_str = "\n".join([f"{k}: {v}" for k, v in list(params.items())[:2]])
            param_text = f"{stage_name}\n({param_str})"
        
        # Position text below node
        x, y = pos[i]
        ax.text(
            x, y - 0.15,
            param_text,
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="yellow" if not attrs["enabled"] else "white",
                alpha=0.7,
                edgecolor="gray",
            ),
        )
    
    # Add title and legend
    ax.set_title(
        "FoodSpec Pipeline DAG\n(Green: enabled, Gray: disabled)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    
    ax.axis("off")
    fig.tight_layout()
    
    # Export if save_path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save SVG (vector format)
        svg_path = save_path / "pipeline_dag.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=dpi)
        
        # Save PNG (raster format)
        png_path = save_path / "pipeline_dag.png"
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=dpi)
    
    return fig


def get_pipeline_stats(protocol: ProtocolV2) -> Dict[str, Any]:
    """Extract pipeline statistics from protocol.
    
    Parameters
    ----------
    protocol : ProtocolV2
        Protocol to analyze
    
    Returns
    -------
    stats : dict
        Pipeline statistics including:
        - total_stages: number of pipeline stages
        - enabled_stages: number of enabled stages
        - disabled_stages: number of disabled stages
        - stage_details: per-stage information
    
    Examples
    --------
    >>> protocol = ProtocolV2.load("pipeline.yaml")
    >>> stats = get_pipeline_stats(protocol)
    >>> print(f"Enabled stages: {stats['enabled_stages']}")
    """
    graph, node_attrs = _build_pipeline_graph(protocol)
    
    enabled = sum(1 for attrs in node_attrs.values() if attrs["enabled"])
    disabled = sum(1 for attrs in node_attrs.values() if not attrs["enabled"])
    
    return {
        "total_stages": len(node_attrs),
        "enabled_stages": enabled,
        "disabled_stages": disabled,
        "stage_details": {
            stage_name: {
                "enabled": attrs["enabled"],
                "params": attrs["params"],
            }
            for stage_name, attrs in node_attrs.items()
        },
    }


__all__ = ["plot_pipeline_dag", "get_pipeline_stats"]
