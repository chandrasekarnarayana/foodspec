"""Reporting utilities for CLI workflows."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

__all__ = [
    "create_run_dir",
    "create_report_folder",
    "write_summary_json",
    "write_metrics_csv",
    "save_figure",
    "write_json",
    "write_markdown_report",
    "summarize_metrics_for_markdown",
]


def create_run_dir(base_dir: pathlib.Path | str, prefix: str) -> pathlib.Path:
    """Create a timestamped run directory under base_dir using UTC."""

    base = pathlib.Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_report_folder(base_out: str | pathlib.Path, workflow_name: str) -> pathlib.Path:
    """Backward-compatible wrapper to create a run directory."""

    return create_run_dir(base_out, workflow_name.upper())


def write_json(path: pathlib.Path | str, data: Dict) -> pathlib.Path:
    """Write JSON with UTF-8 and indentation."""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_summary_json(report_dir: pathlib.Path, summary: dict) -> pathlib.Path:
    """Write summary.json to the report directory."""

    report_dir.mkdir(parents=True, exist_ok=True)
    return write_json(report_dir / "summary.json", summary)


def write_metrics_csv(report_dir: pathlib.Path, name: str, df: pd.DataFrame) -> pathlib.Path:
    """Write a metrics DataFrame to CSV with the given base name."""

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{name}.csv"
    df.to_csv(path, index=True)
    return path


def save_figure(report_dir: pathlib.Path, name: str, fig) -> pathlib.Path:
    """Save a matplotlib figure to the report directory as PNG."""

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def summarize_metrics_for_markdown(metrics: Dict[str, Any]) -> str:
    """Convert a nested metrics dict into a Markdown bullet list."""

    lines = []
    for key, val in metrics.items():
        if isinstance(val, dict):
            lines.append(f"- **{key}**:")
            for k2, v2 in val.items():
                lines.append(f"  - {k2}: {v2}")
        else:
            lines.append(f"- **{key}**: {val}")
    return "\n".join(lines)


def write_markdown_report(path: pathlib.Path | str, title: str, sections: Dict[str, str]) -> pathlib.Path:
    """Write a simple Markdown report."""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}"]
    for sec_title, body in sections.items():
        parts.append(f"## {sec_title}")
        parts.append(body)
    path.write_text("\n\n".join(parts), encoding="utf-8")
    return path
