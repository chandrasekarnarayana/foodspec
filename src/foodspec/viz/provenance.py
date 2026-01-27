"""Provenance and pipeline visualization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import matplotlib.pyplot as plt

from foodspec.reporting.schema import RunBundle
from foodspec.viz.style import apply_style


def _flatten_dict(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in payload.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            items.update(_flatten_dict(value, new_key))
        else:
            items[new_key] = value
    return items


def _resolve_protocol_steps(bundle: RunBundle | Mapping[str, Any]) -> List[str]:
    if isinstance(bundle, RunBundle):
        snapshot = bundle.manifest.get("protocol_snapshot", {}) or {}
    else:
        snapshot = bundle.get("protocol_snapshot", {}) if isinstance(bundle, Mapping) else {}
    steps = snapshot.get("steps", [])
    names = []
    for step in steps:
        if isinstance(step, Mapping):
            names.append(str(step.get("name") or step.get("type") or "step"))
        else:
            names.append(str(step))
    return names


def plot_workflow_dag(data_bundle: RunBundle | Mapping[str, Any], *, seed: int = 0):
    """Plot a simple workflow DAG using protocol steps."""
    apply_style()
    steps = _resolve_protocol_steps(data_bundle)
    if not steps:
        steps = ["load", "preprocess", "model", "report"]

    fig, ax = plt.subplots(figsize=(max(6, len(steps) * 1.2), 2.5))
    ax.axis("off")
    y = 0.5
    x_positions = list(range(len(steps)))
    for idx, step in enumerate(steps):
        ax.scatter([x_positions[idx]], [y], s=400, color="#2a6fdb")
        ax.text(x_positions[idx], y, step, ha="center", va="center", color="white", fontsize=9)
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.3, y),
                xytext=(x_positions[idx] + 0.3, y),
                arrowprops={"arrowstyle": "->", "color": "#4a4a4a", "lw": 1.5},
            )
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_title("Workflow DAG")
    return fig


def plot_parameter_map(data_bundle: RunBundle | Mapping[str, Any], *, seed: int = 0):
    """Plot flattened configuration map as a text table."""
    apply_style()
    if isinstance(data_bundle, RunBundle):
        payload = data_bundle.manifest.get("protocol_snapshot", {}) or data_bundle.manifest.get("config", {})
    else:
        payload = data_bundle.get("config", {}) if isinstance(data_bundle, Mapping) else {}
    flat = _flatten_dict(payload)
    items = list(flat.items())[:40]
    fig, ax = plt.subplots(figsize=(6, max(3, len(items) * 0.25)))
    ax.axis("off")
    text_lines = [f"{k}: {v}" for k, v in items]
    if len(flat) > len(items):
        text_lines.append(f"... ({len(flat) - len(items)} more)")
    ax.text(0.01, 0.99, "\n".join(text_lines), va="top", ha="left", fontsize=8, family="monospace")
    ax.set_title("Parameter Map")
    return fig


def plot_data_lineage(data_bundle: RunBundle | Mapping[str, Any], *, seed: int = 0):
    """Plot a data lineage summary."""
    apply_style()
    if isinstance(data_bundle, RunBundle):
        manifest = data_bundle.manifest
    else:
        manifest = data_bundle if isinstance(data_bundle, Mapping) else {}
    inputs = manifest.get("inputs", [])
    lines = ["Data Lineage", ""]
    if not inputs:
        lines.append("No inputs recorded.")
    else:
        for entry in inputs:
            if isinstance(entry, Mapping):
                path = entry.get("path", "unknown")
                sha = entry.get("sha256", "n/a")
                lines.append(f"- {path} (sha256={sha})")
            else:
                lines.append(f"- {entry}")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9)
    ax.set_title("Data Lineage")
    return fig


def plot_reproducibility_badge(data_bundle: RunBundle | Mapping[str, Any], *, seed: int = 0):
    """Plot a reproducibility badge with run id and commit."""
    apply_style()
    if isinstance(data_bundle, RunBundle):
        manifest = data_bundle.manifest
        run_id = data_bundle.run_id
    else:
        manifest = data_bundle if isinstance(data_bundle, Mapping) else {}
        run_id = str(manifest.get("run_id", "run"))
    commit = manifest.get("git_commit", "unknown")
    date = manifest.get("timestamp", "unknown")

    fig, ax = plt.subplots(figsize=(4, 1.6))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.15), 0.96, 0.7, color="#1f2428", alpha=0.9))
    ax.text(0.5, 0.65, f"Run {run_id}", color="white", ha="center", va="center", fontsize=10)
    ax.text(0.5, 0.35, f"{commit} | {date}", color="#d0d0d0", ha="center", va="center", fontsize=7)
    ax.set_title("Reproducibility Badge", fontsize=9, pad=6)
    return fig


__all__ = [
    "plot_workflow_dag",
    "plot_parameter_map",
    "plot_data_lineage",
    "plot_reproducibility_badge",
]
