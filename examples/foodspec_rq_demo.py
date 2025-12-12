"""
Minimal demo: run the RQ engine on a small in-memory dataset.
"""
import pandas as pd
from foodspec.rq import PeakDefinition, RatioDefinition, RQConfig, RatioQualityEngine


def main():
    df = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4],
            "oil_type": ["A", "A", "B", "B"],
            "matrix": ["oil"] * 4,
            "heating_stage": [0, 1, 0, 1],
            "I_1742": [10, 9, 6, 5],
            "I_1652": [4, 3, 8, 7],
            "I_2720": [5, 5, 5, 5],
        }
    )

    peaks = [
        PeakDefinition(name="I_1742", column="I_1742", wavenumber=1742),
        PeakDefinition(name="I_1652", column="I_1652", wavenumber=1652),
        PeakDefinition(name="I_2720", column="I_2720", wavenumber=2720),
    ]
    ratios = [
        RatioDefinition(name="1742/2720", numerator="I_1742", denominator="I_2720"),
        RatioDefinition(name="1652/2720", numerator="I_1652", denominator="I_2720"),
    ]
    cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage")

    engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg)
    res = engine.run_all(df)

    print("=== Executive summary ===")
    print("\n".join(res.text_report.splitlines()[:12]))


if __name__ == "__main__":
    main()
