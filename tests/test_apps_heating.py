import numpy as np
import pandas as pd

from foodspec.apps.heating import HeatingAnalysisResult, run_heating_degradation_analysis
from foodspec.core.dataset import FoodSpectrumSet


def test_heating_stub_returns_result():
    wn = np.linspace(800, 1800, 200)
    base = np.sin(wn / 200) * 0.05 + 1.0
    peak = np.exp(-0.5 * ((wn - 1655) / 15) ** 2)
    spectra = []
    times = []
    for t in [0, 10, 20, 30]:
        spectra.append(base + (1 + t / 100) * peak)
        times.append(t)
    X = np.vstack(spectra)
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(times))],
            "heating_time": times,
            "oil_type": ["A", "A", "B", "B"],
        }
    )
    ds = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=metadata, modality="ftir")

    result = run_heating_degradation_analysis(ds, time_column="heating_time")
    assert isinstance(result, HeatingAnalysisResult)
    assert result.key_ratios.shape[0] == len(ds)
    assert (result.time_variable.values == np.array(times)).all()
    assert isinstance(result.trend_models, dict)
    assert result.trend_models
    ratio_name = list(result.trend_models.keys())[0]
    model = result.trend_models[ratio_name]
    assert hasattr(model, "coef_") and model.coef_[0] > 0  # slope positive
    assert result.anova_results is not None
    assert len(result.anova_results) >= 1
