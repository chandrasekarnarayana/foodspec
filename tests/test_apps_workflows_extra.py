import json
from pathlib import Path

import numpy as np
import pandas as pd

from foodspec.apps.heating import (
    run_heating_degradation_analysis,
    run_heating_quality_workflow,
)
from foodspec.apps.oils import (
    run_oil_authentication_quickstart,
    run_oil_authentication_workflow,
)
from foodspec.apps.protocol_validation import run_protocol_benchmarks
from foodspec.apps.qc import apply_qc_model, run_qc_workflow, train_qc_model
from foodspec.chemometrics.mixture import run_mixture_analysis_workflow
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


def test_run_oil_authentication_quickstart_smoke():
    labels = ["olive"] * 5 + ["sunflower"] * 5
    fs = _make_dataset(10, 50, labels)
    fs.metadata["oil_type"] = labels
    result = run_oil_authentication_quickstart(fs, label_column="oil_type")
    assert result.confusion_matrix.shape[0] == len(result.class_labels)


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


def test_heating_quality_wrapper_smoke():
    labels = ["olive", "olive", "sunflower", "sunflower"]
    wn = np.linspace(500, 1900, 80)
    peaks1 = np.exp(-0.5 * ((wn - 1655) / 10) ** 2)
    peaks2 = np.exp(-0.5 * ((wn - 1742) / 10) ** 2)
    base = peaks1 + 0.5 * peaks2
    x = np.vstack([base * (1 + i * 0.1) for i in range(4)])
    meta = pd.DataFrame({"label": labels, "heating_time": [0, 10, 20, 30]})
    fs = FoodSpectrumSet(x=x, wavenumbers=wn, metadata=meta, modality="raman")
    res = run_heating_quality_workflow(fs, time_column="heating_time")
    assert not res.key_ratios.empty


def test_qc_train_apply_smoke():
    labels = ["ok"] * 4 + ["suspect"] * 2
    fs = _make_dataset(6, 30, labels)
    model = train_qc_model(fs, model_type="oneclass_svm")
    qc_res = apply_qc_model(fs, model=model)
    assert len(qc_res.scores) == len(fs)
    assert set(qc_res.labels_pred.unique()) <= {"authentic", "suspect"}


def test_qc_workflow_wrapper_smoke():
    labels = ["ok"] * 4 + ["suspect"] * 2
    fs = _make_dataset(6, 30, labels)
    train_mask = pd.Series([True, True, True, True, False, False])
    qc_res = run_qc_workflow(fs, train_mask=train_mask, model_type="oneclass_svm")
    assert len(qc_res.scores) == len(fs)


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



def test_mixture_workflow_nnls_smoke():
    wn = np.linspace(600, 800, 20)
    s1 = np.exp(-0.5 * ((wn - 650) / 5) ** 2)
    s2 = np.exp(-0.5 * ((wn - 750) / 5) ** 2)
    pure = np.vstack([s1, s2])
    mix = 0.7 * s1 + 0.3 * s2
    res = run_mixture_analysis_workflow(mixtures=np.vstack([mix]), pure_spectra=pure, mode="nnls")
    assert "coefficients" in res and res["coefficients"].shape[0] == 1
