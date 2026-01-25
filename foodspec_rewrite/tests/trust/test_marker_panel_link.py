"""
Tests for marker panel and interpretability alignment helper.
"""

import json

import pandas as pd
import pytest

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.trust.marker_panel_link import link_marker_panel_explanations


def test_link_filters_by_marker_names(tmp_path):
    registry = ArtifactRegistry(tmp_path)

    coef_df = pd.DataFrame(
        {
            "feature": ["f0", "f1", "f2"],
            "coefficient": [0.5, 0.1, -0.3],
            "abs_coefficient": [0.5, 0.1, 0.3],
        }
    )
    perm_df = pd.DataFrame(
        {
            "feature": ["f0", "f1", "f2"],
            "importance_mean": [0.2, 0.05, 0.1],
            "importance_std": [0.01, 0.02, 0.03],
            "baseline_metric": [0.9, 0.9, 0.9],
        }
    )

    panel = {
        "selected_feature_names": ["f0", "f2"],
        "selected_indices": [0, 2],
        "selection_frequencies": [0.8, 0.1, 0.7],
    }
    (tmp_path / "marker_panel.json").write_text(json.dumps(panel))

    merged = link_marker_panel_explanations(registry, coef_df, perm_df)

    out_path = registry.trust_dir / "marker_panel_explanations.csv"
    assert out_path.exists()

    loaded = pd.read_csv(out_path)
    assert set(loaded["feature"]) == {"f0", "f2"}
    assert pytest.approx(0.8) == loaded.loc[loaded["feature"] == "f0", "selection_frequency"].iloc[0]
    assert pytest.approx(0.7) == loaded.loc[loaded["feature"] == "f2", "selection_frequency"].iloc[0]
    # Columns from both sources are preserved
    assert "coefficient" in merged.columns
    assert "importance_mean" in merged.columns


def test_link_filters_by_indices_only(tmp_path):
    registry = ArtifactRegistry(tmp_path)

    coef_df = pd.DataFrame(
        {
            "feature": [0, 1, 2],
            "coefficient": [1.0, 0.2, -0.4],
            "abs_coefficient": [1.0, 0.2, 0.4],
        }
    )
    perm_df = pd.DataFrame(
        {
            "feature": [0, 1, 2],
            "importance_mean": [0.3, 0.15, 0.25],
            "importance_std": [0.01, 0.02, 0.02],
            "baseline_metric": [0.92, 0.92, 0.92],
        }
    )

    panel = {
        "selected_indices": [1, 2],
        "selection_frequencies": [0.05, 0.4, 0.6],
    }
    (tmp_path / "marker_panel.json").write_text(json.dumps(panel))

    merged = link_marker_panel_explanations(registry, coef_df, perm_df)

    assert set(merged["feature"]) == {1, 2}
    assert pytest.approx(0.4) == merged.loc[merged["feature"] == 1, "selection_frequency"].iloc[0]
    assert pytest.approx(0.6) == merged.loc[merged["feature"] == 2, "selection_frequency"].iloc[0]


def test_link_no_marker_panel_pass_through(tmp_path):
    registry = ArtifactRegistry(tmp_path)

    coef_df = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "coefficient": [0.4, -0.2],
            "abs_coefficient": [0.4, 0.2],
        }
    )
    perm_df = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance_mean": [0.12, 0.08],
            "importance_std": [0.01, 0.01],
            "baseline_metric": [0.95, 0.95],
        }
    )

    merged = link_marker_panel_explanations(registry, coef_df, perm_df)

    assert set(merged["feature"]) == {"a", "b"}
    out_path = registry.trust_dir / "marker_panel_explanations.csv"
    assert out_path.exists()
    saved = pd.read_csv(out_path)
    assert set(saved["feature"]) == {"a", "b"}
