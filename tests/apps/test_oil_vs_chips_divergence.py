import pandas as pd

from foodspec.features.rq import (
    PeakDefinition,
    RatioDefinition,
    RatioQualityEngine,
    RQConfig,
)


def test_oil_vs_chips_divergence_detects_mean_shift():
    df = pd.DataFrame(
        {
            "oil_type": ["A"] * 4 + ["A"] * 4,
            "matrix": ["oil"] * 4 + ["chips"] * 4,
            "Heating_Stage": [0, 1, 2, 3] * 2,
            "I_1742": [10, 10, 10, 10, 20, 20, 20, 20],
            "I_2720": [10] * 8,
        }
    )
    peaks = [PeakDefinition("I_1742", "I_1742"), PeakDefinition("I_2720", "I_2720")]
    ratios = [RatioDefinition("1742/2720", "I_1742", "I_2720")]
    cfg = RQConfig(
        oil_col="oil_type",
        matrix_col="matrix",
        heating_col="Heating_Stage",
        adjust_p_values=True,
    )
    engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg)
    res = engine.compare_oil_vs_chips(engine.compute_ratios(df))
    assert "p_mean_adj" in res.columns
    # mean difference should be significant
    assert (res["p_mean_adj"] < 0.1).any()
    assert res.loc[res.index[0], "diverges"] is True
