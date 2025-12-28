# FoodSpec RQ Engine (backend)

This document covers the Ratio-Quality Engine in FoodSpec. It is a pure-Python, DataFrame-first API for analyzing Raman/FTIR peak ratios across oils/chips with stability, discrimination, heating trends, and matrix comparisons.

## Inputs
- **DataFrame** with metadata columns and peak intensities.
  - Typical metadata: `sample_id`, `oil_type`, `matrix` (oil/chips), `heating_stage`, `replicate_id`.
  - Peak columns: `I_1742`, `I_1652`, etc. (floats).
- **Peak definitions**: list of `PeakDefinition(name, column, wavenumber, window?)`.
- **Ratio definitions**: list of `RatioDefinition(name, numerator, denominator)`.
- **RQConfig**: column names (oil/matrix/heating/replicate/sample), options for feature importance, CV folds, random seed.

## Outputs
`RatioQualityResult`:
- `ratio_table`: input DF with added ratio columns.
- `stability_summary`: CV/MAD per feature (overall and per-oil).
- `discriminative_summary`: ANOVA/Kruskal p-values per feature.
- `feature_importance`: optional RF/LR importances and CV accuracy.
- `heating_trend_summary`: slope, p-value, monotonic label per feature vs heating_stage.
- `oil_vs_chips_summary`: CVs and trend comparison between matrices; `diverges` flag.
- `text_report`: multi-section RQ report.

## Minimal example
```python
import pandas as pd
from foodspec.features.rq import (
    PeakDefinition, RatioDefinition, RQConfig, RatioQualityEngine
)

# Toy data
df = pd.DataFrame({
    "sample_id": [1, 2, 3, 4],
    "oil_type": ["A", "A", "B", "B"],
    "matrix": ["oil"] * 4,
    "heating_stage": [0, 1, 0, 1],
    "I_1742": [10, 9, 6, 5],
    "I_1652": [4, 3, 8, 7],
    "I_2720": [5, 5, 5, 5],
})

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

print(res.text_report)
```

## Raw spectra → peaks → RQ (full preprocessing)
```python
import pandas as pd
from foodspec.preprocessing_pipeline import PreprocessingConfig, run_full_preprocessing
from foodspec.features.rq import PeakDefinition, RatioDefinition, RQConfig, RatioQualityEngine

# Wide-format spectra (wavenumber columns)
df_raw = pd.read_csv("raman_raw.csv")  # columns: sample_id, oil_type, matrix, heating_stage, 500.0, 501.0, ...

peaks = [
    PeakDefinition(name="I_1742", column="I_1742", wavenumber=1742),
    PeakDefinition(name="I_1652", column="I_1652", wavenumber=1652),
    PeakDefinition(name="I_2720", column="I_2720", wavenumber=2720),
]
ratios = [
    RatioDefinition(name="1742/2720", numerator="I_1742", denominator="I_2720"),
    RatioDefinition(name="1652/2720", numerator="I_1652", denominator="I_2720"),
]

pp_cfg = PreprocessingConfig(
    baseline_lambda=1e5,
    baseline_p=0.01,
    smooth_window=9,
    smooth_polyorder=3,
    normalization="reference",
    reference_wavenumber=2720.0,
    peak_definitions=peaks,
)

df_peaks = run_full_preprocessing(df_raw, pp_cfg)  # adds I_XXXX columns
cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage")
res = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg).run_all(df_peaks)
print(res.text_report)
```

## RQ sections (scientific meaning)
- **Stability (CV/MAD)**: how reproducible each peak/ratio is across replicates/oils; lower CV means more robust markers.
- **Discriminative power (ANOVA/Kruskal + feature importance)**: which peaks/ratios differ significantly between oils; high RF importance or low p-values indicate strong markers.
- **Heating trends**: linear trends vs heating stage; slopes/p-values and monotonic labels (“increases/decreases”) suggest thermal degradation or processing markers.
- **Oil vs chips comparison**: contrasts behavior of the same features in pure oil vs chips matrices; flags divergences when significance or slope direction differs.
- **Text report**: human-readable summary suitable for protocol notes or manuscripts.

---

## When Results Cannot Be Trusted

RQ engine results are unreliable when:

1. **Insufficient replicates** (< 3 per condition) – CV estimates become unstable; increase replicates or flag as exploratory
2. **Zero or near-zero denominators in ratios** – Division by small values amplifies noise; verify denominator peak is well-defined and non-zero
3. **High CV (> 0.3) in reference peaks** – Internal normalization fails if reference peak is unstable; choose different reference or use vector normalization
4. **Non-monotonic heating trends with high p-value** – Heating stage may not correlate with actual thermal history; verify heating conditions and measurement timing
5. **Contradictory matrix effects** – If oil and chips show opposite trends but neither is significant, matrix effect is confounded; collect more data or simplify design

**How to detect:**
- Check `stability_summary['cv']` values in output
- Review `heating_trend_summary['p_value']` and `monotonic` flags
- Inspect `oil_vs_chips_summary['diverges']` column
- Plot raw peak intensities vs replicate_id to identify outliers

**What to do:**
- Filter features with CV > 0.3 before downstream analysis
- Use robust statistics (MAD instead of CV) for non-normal distributions
- Increase replicates per condition (aim for ≥5)
- Verify peak extraction windows capture entire peak without baseline interference

---

## Cross-References

For theoretical background see [RQ Engine (detailed)](rq_engine_detailed.md).  
For practical examples see [heating_quality_monitoring.md](../workflows/quality-monitoring/heating_quality_monitoring.md).
