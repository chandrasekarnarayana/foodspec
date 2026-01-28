"""Multi-run comparison helpers and leaderboard generation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from foodspec.reporting.figures import FigureExporter, FigureStyle, radar_plot
from foodspec.reporting.schema import RunBundle


def _metric_from_row(row: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_score(bundle: RunBundle) -> float:
    if bundle.metrics:
        row = bundle.metrics[0]
        for key in ("macro_f1", "accuracy", "balanced_accuracy", "auroc"):
            value = _metric_from_row(row, [key])
            if value is not None:
                return value
    return 0.0


@dataclass
class CompareResult:
    run_ids: List[str]
    leaderboard_path: Path
    dashboard_path: Path
    radar_path: Path


def compare_runs(
    run_dirs: List[Path | str],
    out_dir: Path | str,
    *,
    style: FigureStyle = FigureStyle.JOSS,
    seed: int = 0,
) -> CompareResult:
    """Compare multiple run directories and build leaderboard + dashboard."""
    out_dir = Path(out_dir)
    compare_dir = out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    bundles = [RunBundle.from_run_dir(Path(run)) for run in run_dirs]
    run_ids = [bundle.run_id for bundle in bundles]

    leaderboard_path = compare_dir / "leaderboard.csv"
    rows: List[Dict[str, Any]] = []
    baseline = bundles[0] if bundles else None
    baseline_score = _extract_score(baseline) if baseline else 0.0

    for bundle in bundles:
        score = _extract_score(bundle)
        rows.append(
            {
                "run_id": bundle.run_id,
                "score": score,
                "delta_vs_baseline": score - baseline_score,
                "path": str(bundle.run_dir),
            }
        )

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    with leaderboard_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    metrics_labels = [r["run_id"] for r in rows] if rows else ["baseline"]
    metrics_values = [r["score"] for r in rows] if rows else [0.0]
    fig = radar_plot(metrics_labels, metrics_values, title="Run Comparison", seed=seed)
    exporter = FigureExporter(style=style, size_preset="single")
    outputs = exporter.export(fig, compare_dir, "radar")
    png_path = next((p for p in outputs if p.suffix == ".png"), compare_dir / "radar.png")

    dashboard_path = compare_dir / "dashboard.html"
    dashboard_path.write_text(
        _render_dashboard_html(rows, png_path),
        encoding="utf-8",
    )

    return CompareResult(
        run_ids=run_ids,
        leaderboard_path=leaderboard_path,
        dashboard_path=dashboard_path,
        radar_path=png_path,
    )


def _render_dashboard_html(rows: List[Dict[str, Any]], radar_path: Path) -> str:
    table_rows = ""
    for row in rows:
        table_rows += "<tr>" + "".join([f"<td>{row[k]}</td>" for k in row]) + "</tr>"
    headers = "".join([f"<th>{k}</th>" for k in (rows[0].keys() if rows else [])])
    radar_rel = radar_path.name
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Run Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #fbf8f2; padding: 20px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; }}
    img {{ max-width: 480px; margin-top: 12px; }}
  </style>
</head>
<body>
  <h1>Run Comparison Dashboard</h1>
  <img src="{radar_rel}" alt="Radar plot">
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""


__all__ = ["compare_runs", "CompareResult"]
