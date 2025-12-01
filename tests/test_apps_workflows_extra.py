import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.apps.heating import run_heating_degradation_analysis
from foodspec.apps.methodsx_reproduction import run_methodsx_reproduction
from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.apps.protocol_validation import run_protocol_benchmarks
from foodspec.apps.qc import apply_qc_model, train_qc_model
from foodspec.core.dataset import FoodSpectrumSet


def _make_dataset(n_samples: int, n_features: int, labels: list[str], modality: str = "raman") -> FoodSpectrumSet:
    x = np.linspace(0, 1, n_samples * n_features, dtype=float).reshape(n_samples, n_features)
    wn = np.linspace(500, 1900, n_features, dtype=float)
    meta = pd.DataFrame({"label": labels, "sample_id": [f"s{i}" for i in range(n_samples)]})
    return FoodSpectrumSet(x=x, wavenumbers=wn, metadata=meta, modality=modality)


def test_run_oil_authentication_workflow_smoke():
    labels = ["olive"] * 5 + ["sunflower"] * 5
    fs = _make_dataset(10, 50, labels)
    fs.metadata["oil_type"] = labels
    result = run_oil_authentication_workflow(fs, label_column="oil_type", classifier_name="rf", cv_splits=5)
    assert result.confusion_matrix.shape[0] == len(result.class_labels)
    assert not result.cv_metrics.empty


def test_run_heating_degradation_workflow_smoke():
    labels = ["olive", "olive", "sunflower", "sunflower"]
    wn = np.linspace(500, 1900, 80)
    peaks1 = np.exp(-0.5 * ((wn - 1655) / 10) ** 2)
    peaks2 = np.exp(-0.5 * ((wn - 1742) / 10) ** 2)
    base = peaks1 + 0.5 * peaks2
    x = np.vstack([base * (1 + i * 0.1) for i in range(4)])
    meta = pd.DataFrame({"label": labels, "heating_time": [0, 10, 20, 30]})
    fs = FoodSpectrumSet(x=x, wavenumbers=wn, metadata=meta, modality="raman")
    res = run_heating_degradation_analysis(fs, time_column="heating_time")
    assert not res.key_ratios.empty
    if res.anova_results is not None:
        assert "pvalue" in res.anova_results.columns


def test_qc_train_apply_smoke():
    labels = ["ok"] * 4 + ["suspect"] * 2
    fs = _make_dataset(6, 30, labels)
    model = train_qc_model(fs, model_type="oneclass_svm")
    qc_res = apply_qc_model(fs, model=model)
    assert len(qc_res.scores) == len(fs)
    assert set(qc_res.labels_pred.unique()) <= {"authentic", "suspect"}


def test_protocol_benchmarks_success(monkeypatch, tmp_path):
    # Patch loaders to small synthetic datasets
    def fake_oils():
        labels = ["olive"] * 4 + ["sunflower"] * 4
        fs = _make_dataset(8, 12, labels)
        fs.metadata["oil_type"] = labels
        return fs

    def fake_mix():
        labels = ["mix"] * 12
        fs = _make_dataset(12, 12, labels)
        fs.metadata["mixture_fraction_evoo"] = np.linspace(0, 1, 12)
        return fs

    monkeypatch.setattr("foodspec.apps.protocol_validation.load_public_mendeley_oils", fake_oils)
    monkeypatch.setattr("foodspec.apps.protocol_validation.load_public_evoo_sunflower_raman", fake_mix)
    summary = run_protocol_benchmarks(output_dir=tmp_path, random_state=0)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "classification_metrics.json").exists()
    assert (run_dir / "mixture_metrics.json").exists()
    metrics = json.loads((run_dir / "classification_metrics.json").read_text())
    assert "accuracy" in metrics


def test_protocol_benchmarks_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "foodspec.apps.protocol_validation.load_public_mendeley_oils",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing oils")),
    )
    monkeypatch.setattr(
        "foodspec.apps.protocol_validation.load_public_evoo_sunflower_raman",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing mix")),
    )
    summary = run_protocol_benchmarks(output_dir=tmp_path)
    assert "classification_error" in summary
    assert "mixture_error" in summary


def test_methodsx_reproduction_smoke(monkeypatch, tmp_path):
    labels = ["olive", "olive", "sunflower", "sunflower"]

    def fake_oils():
        local_labels = ["olive"] * 3 + ["sunflower"] * 3 + ["canola"] * 3
        fs = _make_dataset(9, 20, local_labels)
        fs.metadata["oil_type"] = local_labels
        return fs

    def fake_mix():
        fs = _make_dataset(5, 12, labels[:5] if len(labels) >= 5 else labels + ["olive"])
        fs.metadata["mixture_fraction_evoo"] = np.linspace(0, 1, len(fs))
        return fs

    monkeypatch.setattr("foodspec.apps.methodsx_reproduction.load_public_mendeley_oils", fake_oils)
    monkeypatch.setattr("foodspec.apps.methodsx_reproduction.load_public_evoo_sunflower_raman", fake_mix)
    metrics = run_methodsx_reproduction(output_dir=tmp_path)
    run_dir = Path(metrics["run_dir"])
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "report.md").exists()
