import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.preprocess.engine import (
    AutoPreprocess,
    AlignmentStep,
    BaselineStep,
    DerivativeStep,
    NormalizationStep,
    PreprocessPipeline,
    SmoothingStep,
)


def _toy_dataset(n_samples: int = 4, n_points: int = 50) -> FoodSpectrumSet:
    x = np.linspace(0, 10, n_points)
    spectra = []
    for i in range(n_samples):
        spectra.append(np.sin(x) + 0.1 * np.random.randn(n_points) + 0.2 * i)
    spectra = np.vstack(spectra)
    metadata = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n_samples)]})
    return FoodSpectrumSet(x=spectra, wavenumbers=x, metadata=metadata, modality="raman")


def test_pipeline_steps_run_and_produce_metrics():
    ds = _toy_dataset()
    pipe = PreprocessPipeline([
        BaselineStep("rubberband"),
        SmoothingStep("moving_average", window=3),
        NormalizationStep("vector"),
        DerivativeStep(order=1, window_length=7, polyorder=2),
    ])
    transformed, metrics = pipe.transform(ds)
    assert transformed.x.shape == ds.x.shape
    assert "baseline" in metrics
    assert "smoothing" in metrics
    assert "normalization" in metrics


def test_auto_preprocess_returns_best_pipeline():
    ds = _toy_dataset()
    auto = AutoPreprocess(
        baselines=[{"method": "rubberband"}],
        smoothers=[{"method": "moving_average", "window": 3}],
        aligners=[{"method": "none"}, {"method": "cow", "max_shift": 4}],
        normalizers=[{"method": "vector"}],
        derivatives=[{"order": 0}],
    )
    result = auto.search(ds, max_candidates=4)
    assert result.pipeline.steps, "Pipeline should not be empty"
    transformed, _ = result.pipeline.transform(ds)
    assert transformed.x.shape == ds.x.shape


def test_alignment_step_reduces_shift_error():
    n_points = 80
    x = np.linspace(0, 2 * np.pi, n_points)
    ref = np.sin(x)
    shifted = np.roll(ref, 5)
    spectra = np.vstack([ref, shifted])
    ds = FoodSpectrumSet(x=spectra, wavenumbers=x, metadata=pd.DataFrame({"sample_id": ["r", "s"]}), modality="raman")

    step = AlignmentStep(method="cow", max_shift=8, reference="first")
    pipe = PreprocessPipeline([step])
    aligned, metrics = pipe.transform(ds)

    before_mse = np.mean((ds.x - ref) ** 2)
    after_mse = np.mean((aligned.x - ref) ** 2)
    assert after_mse < before_mse
    assert "alignment" in metrics
    assert metrics["alignment"]["alignment_improvement"] > 0
