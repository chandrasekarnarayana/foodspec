"""
Tests for trust artifact registry paths and writers.
"""

import pandas as pd
import pytest

from foodspec.core.artifacts import ArtifactRegistry


def test_artifact_registry_trust_paths(tmp_path):
    """Verify all trust artifact paths are accessible."""
    registry = ArtifactRegistry(tmp_path)

    assert registry.calibration_metrics_path == tmp_path / "trust" / "calibration_metrics.csv"
    assert registry.conformal_coverage_path == tmp_path / "trust" / "conformal_coverage.csv"
    assert registry.conformal_sets_path == tmp_path / "trust" / "conformal_sets.csv"
    assert registry.abstention_summary_path == tmp_path / "trust" / "abstention_summary.csv"
    assert registry.coefficients_path == tmp_path / "trust" / "coefficients.csv"
    assert registry.permutation_importance_path == tmp_path / "trust" / "permutation_importance.csv"
    assert registry.marker_panel_explanations_path == tmp_path / "trust" / "marker_panel_explanations.csv"


def test_write_trust_calibration_metrics(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_layout()

    metrics = {"ece": 0.05, "mce": 0.12}
    registry.write_trust_calibration_metrics(metrics)

    assert registry.calibration_metrics_path.exists()
    df = pd.read_csv(registry.calibration_metrics_path)
    assert df["ece"].iloc[0] == 0.05
    assert df["mce"].iloc[0] == 0.12


def test_write_trust_coverage(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_layout()

    coverage_df = pd.DataFrame(
        {
            "group": ["all", "train"],
            "coverage": [0.90, 0.88],
            "n_samples": [100, 60],
        }
    )
    registry.write_trust_coverage(coverage_df)

    assert registry.conformal_coverage_path.exists()
    loaded = pd.read_csv(registry.conformal_coverage_path)
    assert len(loaded) == 2
    assert list(loaded["group"]) == ["all", "train"]


def test_write_trust_conformal_sets(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_layout()

    conformal_df = pd.DataFrame(
        {
            "set_size": [2, 1, 3],
            "set_members": ["[0, 1]", "[1]", "[0, 1, 2]"],
            "covered": [1, 1, 0],
        }
    )
    registry.write_trust_conformal_sets(conformal_df)

    assert registry.conformal_sets_path.exists()
    loaded = pd.read_csv(registry.conformal_sets_path)
    assert len(loaded) == 3


def test_write_trust_abstention_summary(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_layout()

    summary = {"abstain_rate": 0.15, "accuracy_on_answered": 0.92}
    registry.write_trust_abstention_summary(summary)

    assert registry.abstention_summary_path.exists()
    loaded = pd.read_csv(registry.abstention_summary_path)
    assert loaded["abstain_rate"].iloc[0] == 0.15


def test_write_trust_coefficients(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_layout()

    coef_df = pd.DataFrame(
        {
            "feature": ["f0", "f1", "f2"],
            "coefficient": [0.5, 0.1, -0.3],
            "abs_coefficient": [0.5, 0.1, 0.3],
        }
    )
    registry.write_trust_coefficients(coef_df)

    assert registry.coefficients_path.exists()
    loaded = pd.read_csv(registry.coefficients_path)
    assert len(loaded) == 3


def test_write_trust_permutation_importance(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_layout()

    importance_df = pd.DataFrame(
        {
            "feature": ["f0", "f1"],
            "importance_mean": [0.2, 0.05],
            "importance_std": [0.01, 0.02],
        }
    )
    registry.write_trust_permutation_importance(importance_df)

    assert registry.permutation_importance_path.exists()
    loaded = pd.read_csv(registry.permutation_importance_path)
    assert len(loaded) == 2


def test_write_trust_marker_panel_explanations(tmp_path):
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_layout()

    explanations_df = pd.DataFrame(
        {
            "feature": ["f0", "f2"],
            "coefficient": [0.5, -0.3],
            "importance_mean": [0.2, 0.1],
            "selection_frequency": [0.8, 0.7],
        }
    )
    registry.write_trust_marker_panel_explanations(explanations_df)

    assert registry.marker_panel_explanations_path.exists()
    loaded = pd.read_csv(registry.marker_panel_explanations_path)
    assert len(loaded) == 2
