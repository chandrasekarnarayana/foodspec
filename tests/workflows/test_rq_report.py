import pandas as pd

from foodspec.features.rq import PeakDefinition, RatioDefinition, RatioQualityEngine, RQConfig


def test_rq_report_contains_sections():
    df = pd.DataFrame(
        {
            "oil_type": ["A", "A", "B", "B"],
            "matrix": ["oil"] * 4,
            "heating_stage": [0, 1, 0, 1],
            "I_1000": [10, 11, 5, 6],
            "I_1100": [8, 9, 4, 5],
        }
    )
    peaks = [
        PeakDefinition(name="I_1000", column="I_1000", wavenumber=1000),
        PeakDefinition(name="I_1100", column="I_1100", wavenumber=1100),
    ]
    ratios = [RatioDefinition(name="1000/1100", numerator="I_1000", denominator="I_1100")]
    engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=RQConfig(oil_col="oil_type", matrix_col="matrix"))
    res = engine.run_all(df)
    text = res.text_report
    assert "RQ1" in text
    assert "RQ2" in text
    assert "RQ3" in text
    assert res.stability_summary is not None
