import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.viz.visualization_manager import VisualizationManager


def _dummy_protocol() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(input="data.csv", format="csv"),
        preprocess=SimpleNamespace(recipe="basic", steps=["smooth"]),
        qc=SimpleNamespace(thresholds={"snr": 1.0}, metrics=["snr"]),
        features=SimpleNamespace(modules=["pca"], strategy="manual"),
        model=SimpleNamespace(estimator="svm", hyperparameters={"C": 1.0}),
        uncertainty=SimpleNamespace(
            conformal={"calibration": {"method": "platt"}, "conformal": {"method": "mondrian", "alpha": 0.1}}
        ),
        interpretability=SimpleNamespace(methods=["coefficients"], marker_panel=["band1"]),
        reporting=SimpleNamespace(format="html", sections=["summary"]),
        export=SimpleNamespace(bundle=True),
    )


def test_visualization_manager_end_to_end(tmp_path, monkeypatch):
    registry = ArtifactRegistry(tmp_path)
    registry.ensure_layout()

    protocol = _dummy_protocol()

    wavenumbers = np.linspace(1000, 1100, 40)
    spectra = np.stack(
        [
            np.linspace(0.1, 1.1, wavenumbers.size),
            np.linspace(0.2, 1.2, wavenumbers.size),
            np.linspace(0.15, 1.15, wavenumbers.size),
            np.linspace(0.05, 1.05, wavenumbers.size),
            np.linspace(0.12, 1.12, wavenumbers.size),
        ]
    )
    meta = {"batch": np.array(["A", "A", "A", "B", "B"])}

    spectrum = np.linspace(0.0, 1.0, wavenumbers.size)
    importance = np.linspace(0.2, 0.8, wavenumbers.size)
    confidences = np.linspace(0.2, 0.9, 5)
    sample_labels = [f"sample-{i}" for i in range(len(confidences))]

    data_store = {
        "protocol": protocol,
        "spectra": spectra,
        "meta": meta,
        "wavenumbers": wavenumbers,
        "spectrum": spectrum,
        "importance": importance,
        "confidences": confidences,
        "sample_labels": sample_labels,
    }

    visualizations = [
        {
            "name": "pipeline_dag",
            "type": "pipeline_dag",
            "params": {"save_path": registry.viz_pipeline_dir},
            "data_keys": {"protocol": "protocol"},
        },
        {
            "name": "batch_drift",
            "type": "batch_drift",
            "params": {
                "save_path": registry.viz_drift_dir,
                "batch_key": "batch",
                "wavenumbers": wavenumbers,
            },
            "data_keys": {"spectra": "spectra", "meta": "meta"},
        },
        {
            "name": "importance_overlay",
            "type": "importance_overlay",
            "params": {
                "save_path": registry.viz_interpretability_dir / "importance_overlay.png",
                "title": "Importance Overlay - batch A",
                "wavenumbers": wavenumbers,
            },
            "data_keys": {"spectrum": "spectrum", "importance": "importance"},
        },
        {
            "name": "confidence_map",
            "type": "confidence_map",
            "params": {
                "save_path": registry.viz_uncertainty_dir / "confidence_map.png",
                "title": "Confidence Map - batch A",
            },
            "data_keys": {"confidences": "confidences", "sample_labels": "sample_labels"},
        },
    ]

    closed_figures = []
    original_close = plt.close

    def _track_close(fig=None):
        if fig is not None:
            closed_figures.append(fig)
        return original_close(fig)

    monkeypatch.setattr(plt, "close", _track_close)

    manager = VisualizationManager(
        protocol={},
        manifest={"visualizations": visualizations},
        data_store=data_store,
        output_dir=registry.viz_dir,
    )

    results = manager.run_all()

    assert len(results) == 4
    assert {r.status for r in results} == {"success"}
    assert all(r.save_path is not None for r in results)

    pipeline_png = registry.viz_pipeline_dir / "pipeline_dag.png"
    drift_png = registry.viz_drift_dir / "batch_drift.png"
    importance_png = registry.viz_interpretability_dir / "importance_overlay.png"
    confidence_png = registry.viz_uncertainty_dir / "confidence_map.png"

    for path in [pipeline_png, drift_png, importance_png, confidence_png]:
        assert path.exists()
        assert path.stat().st_size > 0

    name_to_path = {result.name: result.save_path for result in results}
    assert name_to_path["pipeline_dag"] == registry.viz_pipeline_dir
    assert name_to_path["batch_drift"] == registry.viz_drift_dir
    assert name_to_path["importance_overlay"] == importance_png
    assert name_to_path["confidence_map"] == confidence_png

    titles = [ax.get_title() for fig in closed_figures for ax in fig.axes]
    assert any("Reference Batch 'A'" in title for title in titles)
    assert any("Confidence Map - batch A" in title for title in titles)
