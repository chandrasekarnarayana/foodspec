"""Minimal marker panel selection and export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from foodspec.features.schema import FeatureInfo, normalize_assignment
from foodspec.utils.run_artifacts import safe_json_dump


def _feature_info_map(feature_info: Optional[Iterable[Any]]) -> dict[str, dict[str, Any]]:
    if not feature_info:
        return {}
    info_map: dict[str, dict[str, Any]] = {}
    for entry in feature_info:
        if isinstance(entry, FeatureInfo):
            info_map[entry.name] = entry.to_dict()
        elif isinstance(entry, dict) and "name" in entry:
            info_map[str(entry["name"])] = dict(entry)
    return info_map


def build_marker_panel(
    stability: pd.Series | pd.DataFrame,
    *,
    k: int = 8,
    performance: Optional[pd.Series] = None,
    feature_info: Optional[Iterable[Any]] = None,
    weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> pd.DataFrame:
    """Select a minimal marker panel using stability, performance, and interpretability."""

    if isinstance(stability, pd.DataFrame):
        if "feature" not in stability.columns or "frequency" not in stability.columns:
            raise ValueError("stability DataFrame must include feature and frequency columns.")
        stability_series = stability.set_index("feature")["frequency"]
    else:
        stability_series = stability

    if stability_series.empty:
        raise ValueError("stability scores are empty.")

    info_map = _feature_info_map(feature_info)
    perf = performance if performance is not None else pd.Series(0.0, index=stability_series.index)
    perf = perf.reindex(stability_series.index).fillna(0.0)

    perf_min, perf_max = float(perf.min()), float(perf.max())
    if perf_max > perf_min:
        perf_norm = (perf - perf_min) / (perf_max - perf_min)
    else:
        perf_norm = pd.Series(0.0, index=perf.index)

    interp_scores = []
    assignments = []
    descriptions = []
    for feat in stability_series.index:
        meta = info_map.get(str(feat), {})
        assignment = normalize_assignment(meta.get("assignment"))
        assignments.append(assignment)
        interp_scores.append(1.0 if assignment != "unassigned" else 0.0)
        desc = meta.get("description") or f"Feature {feat} (assignment: {assignment})."
        descriptions.append(desc)

    w_stab, w_perf, w_interp = weights
    score = (
        w_stab * stability_series
        + w_perf * perf_norm
        + w_interp * pd.Series(interp_scores, index=stability_series.index)
    )

    panel = pd.DataFrame(
        {
            "feature": stability_series.index,
            "stability": stability_series.values,
            "performance": perf_norm.values,
            "interpretability": interp_scores,
            "score": score.values,
            "assignment": assignments,
            "explanation": descriptions,
        }
    )
    panel = panel.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    panel.insert(0, "rank", range(1, len(panel) + 1))
    return panel


def export_marker_panel(panel: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Export marker panel to JSON and CSV."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "marker_panel.json"
    csv_path = out_dir / "marker_panel.csv"

    safe_json_dump(json_path, {"panel": panel.to_dict(orient="records")})
    panel.to_csv(csv_path, index=False)
    return json_path, csv_path


__all__ = ["build_marker_panel", "export_marker_panel"]
