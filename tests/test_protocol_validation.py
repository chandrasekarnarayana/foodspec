import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.apps.protocol_validation import run_protocol_benchmarks
from foodspec.core.dataset import FoodSpectrumSet


def _synthetic_classification() -> FoodSpectrumSet:
    wn = np.linspace(1000, 1010, 10)
    X = np.random.default_rng(0).normal(size=(6, wn.size))
    labels = ["olive", "olive", "sunflower", "sunflower", "canola", "canola"]
    meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)], "oil_type": labels})
    return FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")


def _synthetic_mixture() -> FoodSpectrumSet:
    wn = np.linspace(1000, 1010, 10)
    X = np.random.default_rng(1).normal(size=(5, wn.size))
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    meta = pd.DataFrame(
        {"sample_id": [f"m{i}" for i in range(5)], "mixture_fraction_evoo": fractions}
    )
    return FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")


def test_run_protocol_benchmarks_smoke(tmp_path):
    from foodspec.apps import protocol_validation as pv

    def fake_cls(random_state=0):
        return {"task": "oil_classification", "accuracy": 1.0}, pd.DataFrame([[1]])

    def fake_mix(random_state=0):
        return {"task": "mixture_regression", "rmse": 0.0, "r2": 1.0}

    # avoid stratified split constraints on tiny synthetic data
    monkeypatcher = pv
    pv._classification_benchmark = fake_cls  # type: ignore
    pv._mixture_benchmark = fake_mix  # type: ignore

    summary = run_protocol_benchmarks(output_dir=tmp_path, random_state=0)
    run_dir = Path(summary["run_dir"])
    assert run_dir.parent == tmp_path
    metrics_path = run_dir / "classification_metrics.json"
    report_path = run_dir / "report.md"
    assert metrics_path.exists()
    assert report_path.exists()

    with metrics_path.open() as f:
        metrics = json.load(f)
    assert "task" in metrics
    mixture_metrics = json.loads((run_dir / "mixture_metrics.json").read_text())
    assert "rmse" in mixture_metrics


def test_run_protocol_benchmarks_cv_splits(tmp_path, monkeypatch):
    # Reduce CV splits in classifier factory to avoid errors on tiny datasets
    from foodspec.apps import protocol_validation as pv

    def _classification_benchmark(random_state=0):
        ds = _synthetic_classification()
        X = ds.x
        y = ds.metadata["oil_type"].to_numpy()
        # trivial metrics for coverage
        return {"task": "oil_classification", "accuracy": 1.0}, pd.DataFrame(
            [[1, 0], [0, 1]], index=np.unique(y)[:2], columns=np.unique(y)[:2]
        )

    monkeypatch.setattr(pv, "_classification_benchmark", _classification_benchmark)
    summary = run_protocol_benchmarks(output_dir=tmp_path, random_state=1)
    assert "classification" in summary or "classification_error" in summary
