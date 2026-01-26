from __future__ import annotations

from pathlib import Path

import numpy as np

from foodspec.viz import api as viz_api


def _assert_outputs(tmp_path: Path, name: str) -> None:
    fig_path = tmp_path / "figures" / f"{name}.png"
    meta_path = tmp_path / "figures" / f"{name}.meta.json"
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0
    assert meta_path.exists()
    assert meta_path.stat().st_size > 0


def test_viz_api_smoke(tmp_path: Path) -> None:
    wn = np.linspace(400, 1800, 50)
    spectrum = np.sin(wn / 200.0)
    payload = {
        "wavenumbers": wn,
        "raw": spectrum,
        "processed": spectrum * 0.9,
    }
    viz_api.plot_raw_processed_overlay(payload, outdir=tmp_path, name="overlay")
    _assert_outputs(tmp_path, "overlay")

    viz_api.plot_spectra_heatmap(
        {"matrix": np.random.default_rng(0).normal(size=(10, 20))},
        outdir=tmp_path,
        name="spectra_heatmap",
    )
    _assert_outputs(tmp_path, "spectra_heatmap")

    viz_api.plot_correlation_heatmap(
        {"matrix": np.random.default_rng(0).normal(size=(10, 6))},
        outdir=tmp_path,
        name="correlation",
    )
    _assert_outputs(tmp_path, "correlation")

    viz_api.plot_pca_scatter(
        {"X": np.random.default_rng(0).normal(size=(20, 5)), "labels": ["a"] * 10 + ["b"] * 10},
        outdir=tmp_path,
        name="pca",
    )
    _assert_outputs(tmp_path, "pca")

    viz_api.plot_umap_scatter(
        {"X": np.random.default_rng(1).normal(size=(20, 5)), "labels": ["a"] * 10 + ["b"] * 10},
        outdir=tmp_path,
        name="umap",
    )
    _assert_outputs(tmp_path, "umap")

    viz_api.plot_confusion_matrix(
        {"y_true": [0, 0, 1, 1], "y_pred": [0, 1, 1, 1]},
        outdir=tmp_path,
        name="confusion",
    )
    _assert_outputs(tmp_path, "confusion_counts")
    _assert_outputs(tmp_path, "confusion_normalized")

    viz_api.plot_reliability_diagram(
        {"y_true": [0, 1, 0, 1], "y_prob": [0.1, 0.9, 0.4, 0.8]},
        outdir=tmp_path,
        name="reliability",
    )
    _assert_outputs(tmp_path, "reliability")

    viz_api.plot_workflow_dag(
        {"protocol_snapshot": {"steps": [{"type": "load"}, {"type": "preprocess"}, {"type": "model"}]}},
        outdir=tmp_path,
        name="workflow",
    )
    _assert_outputs(tmp_path, "workflow")

    viz_api.plot_parameter_map(
        {"config": {"preprocess": {"baseline": "als"}, "model": {"name": "svm"}}},
        outdir=tmp_path,
        name="params",
    )
    _assert_outputs(tmp_path, "params")

    viz_api.plot_data_lineage(
        {"inputs": [{"path": "data.csv", "sha256": "abc"}]},
        outdir=tmp_path,
        name="lineage",
    )
    _assert_outputs(tmp_path, "lineage")

    viz_api.plot_reproducibility_badge(
        {"run_id": "demo", "git_commit": "deadbeef", "timestamp": "2025-01-01T00:00:00Z"},
        outdir=tmp_path,
        name="badge",
    )
    _assert_outputs(tmp_path, "badge")

    viz_api.plot_batch_drift(
        {"batch_names": ["b1", "b2"], "drift_scores": [0.1, 0.3]},
        outdir=tmp_path,
        name="batch_drift",
    )
    _assert_outputs(tmp_path, "batch_drift")

    viz_api.plot_stage_difference_spectra(
        {"wavenumbers": wn, "stage_spectra": {"base": spectrum, "stage1": spectrum + 0.1}},
        outdir=tmp_path,
        name="stage_diff",
    )
    _assert_outputs(tmp_path, "stage_diff")

    viz_api.plot_replicate_similarity(
        {"similarity_matrix": np.eye(3)},
        outdir=tmp_path,
        name="replicate_similarity",
    )
    _assert_outputs(tmp_path, "replicate_similarity")

    viz_api.plot_temporal_drift(
        {"time_points": [1, 2, 3], "drift_values": [0.1, 0.2, 0.25]},
        outdir=tmp_path,
        name="temporal_drift",
    )
    _assert_outputs(tmp_path, "temporal_drift")

    viz_api.plot_importance_overlay(
        {"wavenumbers": wn, "spectrum": spectrum, "importance": np.abs(np.random.default_rng(0).normal(size=wn.size))},
        outdir=tmp_path,
        name="importance_overlay",
    )
    _assert_outputs(tmp_path, "importance_overlay")

    viz_api.plot_marker_bands(
        {"wavenumbers": wn, "bands": [(600, 650), (1200, 1300)]},
        outdir=tmp_path,
        name="marker_bands",
    )
    _assert_outputs(tmp_path, "marker_bands")

    coef = np.random.default_rng(0).normal(size=(2, 4))
    viz_api.plot_coefficient_heatmap(
        {"coefficients": coef, "feature_names": ["f1", "f2", "f3", "f4"]},
        outdir=tmp_path,
        name="coef_heatmap",
    )
    _assert_outputs(tmp_path, "coef_heatmap")

    stability = np.random.default_rng(0).random(size=(3, 4))
    viz_api.plot_feature_stability(
        {"stability_matrix": stability, "feature_names": ["f1", "f2", "f3", "f4"]},
        outdir=tmp_path,
        name="feature_stability",
    )
    _assert_outputs(tmp_path, "feature_stability")

    viz_api.plot_confidence_map(
        {"confidences": [0.2, 0.7, 0.9]},
        outdir=tmp_path,
        name="confidence",
    )
    _assert_outputs(tmp_path, "confidence")

    viz_api.plot_conformal_set_sizes(
        {"set_sizes": [1, 2, 2, 3]},
        outdir=tmp_path,
        name="conformal",
    )
    _assert_outputs(tmp_path, "conformal")

    viz_api.plot_coverage_efficiency(
        {"alphas": [0.05, 0.1], "coverages": [0.95, 0.9], "avg_sizes": [1.2, 1.5]},
        outdir=tmp_path,
        name="coverage_efficiency",
    )
    _assert_outputs(tmp_path, "coverage_efficiency")

    viz_api.plot_abstention_distribution(
        {"abstain_flags": [0, 1, 0, 1]},
        outdir=tmp_path,
        name="abstention",
    )
    _assert_outputs(tmp_path, "abstention")
