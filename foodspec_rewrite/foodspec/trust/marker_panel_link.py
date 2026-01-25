"""
Helper to align marker panel selections with interpretability outputs.

Merges stability selection marker panel artifacts (marker_panel.json)
with coefficient tables and permutation importance results, filtering to
selected markers when available and exporting a consolidated CSV for reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from foodspec.core.artifacts import ArtifactRegistry


def link_marker_panel_explanations(
    registry: ArtifactRegistry,
    coefficients: pd.DataFrame,
    permutation_importances: pd.DataFrame,
    marker_panel_path: Optional[Path] = None,
    output_filename: str = "marker_panel_explanations.csv",
) -> pd.DataFrame:
    """
    Merge marker panel selections with interpretability outputs and write CSV.

    The helper checks for a marker_panel.json (or a provided path). When present,
    coefficients and permutation importance tables are filtered to the selected
    markers, merged on the "feature" column, enriched with selection frequencies,
    and written to ``trust/marker_panel_explanations.csv`` under the run's
    artifact directory.

    Parameters
    ----------
    registry : ArtifactRegistry
        Registry pointing to the run artifact root.
    coefficients : pd.DataFrame
        Coefficient table containing a "feature" column (e.g., output of
        ``extract_linear_coefficients``).
    permutation_importances : pd.DataFrame
        Permutation importance table with a "feature" column (e.g., output of
        ``permutation_importance`` or ``permutation_importance_with_names``).
    marker_panel_path : Path, optional
        Override path to marker_panel.json. Defaults to ``registry.root / "marker_panel.json"``.
    output_filename : str, default="marker_panel_explanations.csv"
        Output filename placed under ``registry.trust_dir``.

    Returns
    -------
    pd.DataFrame
        Merged explanation table aligned with the marker panel (when present).
    """

    registry.ensure_layout()
    panel_path = Path(marker_panel_path) if marker_panel_path else registry.root / "marker_panel.json"

    selected_names: Optional[set] = None
    selected_indices: Optional[set[int]] = None
    freq_map: dict = {}

    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
        names = panel.get("selected_feature_names") or panel.get("selected_names") or []
        indices_list = panel.get("selected_indices") or []
        freqs = panel.get("selection_frequencies") or []

        if names:
            selected_names = set(names)
        if indices_list:
            selected_indices = {int(i) for i in indices_list}

        # Build frequency map keyed by names when available, otherwise by index.
        if freqs and indices_list:
            if names and len(names) == len(indices_list):
                for name, idx in zip(names, indices_list):
                    if 0 <= idx < len(freqs):
                        freq_map[name] = freqs[idx]
            else:
                for idx in indices_list:
                    if 0 <= idx < len(freqs):
                        freq_map[int(idx)] = freqs[idx]

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if selected_names:
            out = out[out["feature"].isin(selected_names)]
        elif selected_indices is not None:
            # Match numeric feature identifiers
            def _to_int(val):
                try:
                    return int(val)
                except Exception:
                    return None
            numeric_features = out["feature"].map(_to_int)
            out = out[numeric_features.isin(selected_indices)]
        return out

    filtered_coef = _filter(coefficients)
    filtered_perm = _filter(permutation_importances)

    merged = pd.merge(filtered_coef, filtered_perm, on="feature", how="outer")

    if freq_map:
        def _lookup_frequency(val):
            if val in freq_map:
                return freq_map[val]
            try:
                val_int = int(val)
                return freq_map.get(val_int)
            except Exception:
                return None
        merged["selection_frequency"] = merged["feature"].map(_lookup_frequency)

    # Keep an interpretable ordering if available
    if "abs_coefficient" in merged.columns:
        merged = merged.sort_values("abs_coefficient", ascending=False, ignore_index=True)
    elif "importance_mean" in merged.columns:
        merged = merged.sort_values("importance_mean", ascending=False, ignore_index=True)
    else:
        merged = merged.reset_index(drop=True)

    output_path = registry.trust_dir / output_filename
    registry.write_csv(output_path, merged.to_dict(orient="records"))
    return merged


__all__ = ["link_marker_panel_explanations"]
