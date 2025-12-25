import pandas as pd

from foodspec.features.rq import PeakDefinition, RatioDefinition, RatioQualityEngine, RQConfig


def test_rq_guardrails_and_qc_warnings():
    # Tiny dataset with many features to trigger warnings
    data = {
        "oil_type": ["A", "B"],
        "matrix": ["oil", "oil"],
        "heating_stage": [0, 1],
    }
    # Add many dummy features
    for i in range(20):
        data[f"I_{i}"] = [1.0, 2.0]
    df = pd.DataFrame(data)
    peaks = [PeakDefinition(name=f"I_{i}", column=f"I_{i}", mode="max") for i in range(5)]
    ratios = [RatioDefinition(name=f"r{i}", numerator=f"I_{i}", denominator="I_0") for i in range(1, 5)]
    cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage", max_features=10, n_splits=5)
    engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg)
    res = engine.run_all(df)
    assert any("High feature-to-sample ratio" in w for w in res.warnings)
    assert "Samples:" in res.text_report
