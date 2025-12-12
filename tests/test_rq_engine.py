import numpy as np
import pandas as pd

from foodspec.rq import (
    PeakDefinition,
    RatioDefinition,
    RQConfig,
    RatioQualityEngine,
)


def make_engine():
    peaks = [
        PeakDefinition(name="I_1742", column="I_1742"),
        PeakDefinition(name="I_1652", column="I_1652"),
        PeakDefinition(name="I_2720", column="I_2720"),
    ]
    ratios = [
        RatioDefinition(name="1742/2720", numerator="I_1742", denominator="I_2720"),
        RatioDefinition(name="1652/2720", numerator="I_1652", denominator="I_2720"),
    ]
    cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage")
    return RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg)


def test_ratio_computation():
    df = pd.DataFrame({"I_1742": [10, 20], "I_1652": [5, 10], "I_2720": [2, 4]})
    engine = make_engine()
    out = engine.compute_ratios(df)
    np.testing.assert_allclose(out["1742/2720"], [5.0, 5.0])
    np.testing.assert_allclose(out["1652/2720"], [2.5, 2.5])


def test_stability_cv():
    df = pd.DataFrame(
        {
            "oil_type": ["A", "A", "A"],
            "I_1742": [10.0, 10.0, 10.0],
            "I_1652": [5.0, 6.0, 7.0],
            "I_2720": [2.0, 2.0, 2.0],
            "1742/2720": [5.0, 5.0, 5.0],
            "1652/2720": [2.5, 3.0, 3.5],
        }
    )
    engine = make_engine()
    stability = engine.compute_stability(df)
    cv_1742 = stability[(stability["feature"] == "I_1742") & (stability["level"] == "overall")]["cv_percent"].iloc[0]
    assert cv_1742 == 0.0
    cv_ratio = stability[(stability["feature"] == "1652/2720") & (stability["level"] == "overall")]["cv_percent"].iloc[0]
    assert cv_ratio > 0.0


def test_discriminative_test():
    df = pd.DataFrame(
        {
            "oil_type": ["A"] * 5 + ["B"] * 5,
            "I_1742": [10, 10, 10, 10, 10, 20, 20, 20, 20, 20],
            "I_1652": [5] * 10,
            "I_2720": [2] * 10,
            "1742/2720": [5] * 5 + [10] * 5,
            "1652/2720": [2.5] * 10,
        }
    )
    engine = make_engine()
    discrim, _ = engine.compute_discriminative_power(df)
    sig = discrim[discrim["feature"] == "1742/2720"]["p_value"].iloc[0]
    assert sig < 0.01


def test_heating_trend_detection():
    df = pd.DataFrame(
        {
            "heating_stage": [0, 1, 2, 3, 4],
            "I_1742": [10, 9, 8, 7, 6],
            "I_1652": [1, 1, 1, 1, 1],
            "I_2720": [2, 2, 2, 2, 2],
            "1742/2720": [5, 4.5, 4, 3.5, 3],
            "1652/2720": [0.5] * 5,
        }
    )
    engine = make_engine()
    trends = engine.compute_heating_trends(df)
    slope = trends[trends["feature"] == "1742/2720"]["slope"].iloc[0]
    assert slope < 0
    pval = trends[trends["feature"] == "1742/2720"]["p_value"].iloc[0]
    assert pval < 0.1
