# Heating degradation workflow

Questions this page answers
- How do heating time/temperature affect oil spectra?
- How do I run the heating workflow (Python and CLI)?
- How do I interpret ratio trends, slopes, and ANOVA?
- How should I report these results?

## Experiment setup
- Input: spectra with metadata column (e.g., `heating_time` or `temperature`); optional `oil_type`.
- Goal: track degradation via ratios (e.g., 1655/1742) over time/cycles.

## Running the workflow
CLI:
```bash
foodspec heating libraries/heating.h5 \
  --time-column heating_time \
  --output-dir runs/heating_demo
```
Outputs: ratios.csv, ratio_vs_time.png, optional anova.csv, report.md.

Python:
```python
from foodspec.data import load_library
from foodspec.apps.heating import run_heating_degradation_analysis
fs = load_library("libraries/heating.h5")
res = run_heating_degradation_analysis(fs, time_column="heating_time")
print(res.key_ratios.head())
```

## Interpreting results
- Ratios vs time: slope sign/magnitude indicates increasing/decreasing features (e.g., unsaturation loss).  
- Trend model: simple linear fit \( r = a + b t \); slope \( b \) is key metric.  
- ANOVA (if group labels): p-value tests differences between groups (e.g., oil types).
- Look for consistent trends across batches; inspect variance.

## Reporting
- Main figures: ratio_vs_time plot with fitted line; note slope and any significant p-values.  
- Main text: preprocessing summary, ratio definition, slope/correlation, ANOVA if applicable.  
- Supplementary: full ratios table, ANOVA table, spectra before/after heating, run metadata/configs.
- Supporting tests (conceptual): peroxide/anisidine values, GC–MS, sensory panel for corroboration.

## Optional: testing degradation trends
Test whether a ratio changes significantly over time using simple linear regression.
```python
import pandas as pd
from scipy.stats import linregress

# df_ratios has columns: ratio_1655_1745, heating_time
slope, intercept, r, p, stderr = linregress(df_ratios["heating_time"], df_ratios["ratio_1655_1745"])
print(f"slope={slope:.3f}, R²={r**2:.3f}, p={p:.3g}")
```
Interpretation: slope indicates direction/magnitude of change; p tests if slope differs from zero; R² shows how much of the ratio variance is explained by time. Include these in MethodsX-style reports alongside plots.

### Minimal workflow example (Python)
```python
import numpy as np
from foodspec.data import load_example_oils
from foodspec.apps.heating import run_heating_degradation_analysis

fs = load_example_oils()
fs.metadata["heating_time"] = np.linspace(0, 60, len(fs))
result = run_heating_degradation_analysis(fs, time_column="heating_time")
print(result.key_ratios.head())
```

Recommended plots: ratio vs time with regression line; ANOVA boxplots; residuals. See [plotting](visualization/plotting_with_foodspec.md) and [metrics](metrics/metrics_and_evaluation.md).  
Preprocessing links: [Baseline](preprocessing/baseline_correction.md), [Normalization](preprocessing/normalization_smoothing.md).  
Stats links: [ANOVA/hypothesis testing](stats/hypothesis_testing_in_food_spectroscopy.md), [Nonparametric](stats/nonparametric_methods_and_robustness.md).  
Reproducibility: [Checklist](protocols/reproducibility_checklist.md), [Reporting](reporting_guidelines.md).

See also
- [Metrics & evaluation](metrics/metrics_and_evaluation.md)
- [reporting_guidelines.md](reporting_guidelines.md)
- [keyword_index.md](keyword_index.md)
- [ftir_raman_preprocessing.md](ftir_raman_preprocessing.md)
