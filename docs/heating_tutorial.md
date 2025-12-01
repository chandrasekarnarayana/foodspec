# Heating Degradation Analysis

Use `run_heating_degradation_analysis` to assess heating-induced spectral changes.

```python
from foodspec.apps.heating import run_heating_degradation_analysis
from foodspec.data.loader import load_example_oils  # replace with your dataset
from foodspec.viz.heating import plot_ratio_vs_time
import matplotlib.pyplot as plt

spectra = load_example_oils()
# spectra.metadata should include a time column, e.g., "heating_time"
result = run_heating_degradation_analysis(spectra, time_column="heating_time")

print(result.key_ratios.head())        # peak ratios vs time
print(result.trend_models)             # fitted LinearRegression models
print(result.anova_results)            # optional ANOVA across groups (if present)

# Plot ratio vs time with fitted trend
ratio_name = "ratio_1655_1742"
model = result.trend_models[ratio_name]
plot_ratio_vs_time(result.time_variable, result.key_ratios[ratio_name], model=model)
plt.show()
```

Outputs:
- Preprocessed spectra and cropped wavenumbers.
- Peak ratios (e.g., 1655/1742) vs the heating time variable.
- Trend models (`LinearRegression`) per ratio, and optionally per `oil_type` group.
- ANOVA summary if at least two groups are available.
# Heating & degradation workflow

## Scientific question
“How do heating time and temperature affect oil composition and quality?” Spectra track chemical changes (oxidation, polymerization) over time or across frying cycles.

### Required metadata
- A time/temperature variable (e.g., `heating_time`, `cycle`, or `temperature`).
- Optional group labels (e.g., `oil_type`, treatment vs control).

## How FoodSpec computes trends
1. Preprocess spectra (ALS baseline, Savitzky–Golay smoothing, Vector/MSC normalization, fingerprint cropping).  
2. Extract peak ratios indicative of oxidation: e.g., 1655/1742 cm⁻¹ (C=C/C=O balance).  
3. Fit simple trend models: linear regression of ratio vs time.  
4. Optional ANOVA: compare ratio distributions across groups (e.g., oil types) using one-way ANOVA on end-point or aggregated values.

### Mathematics (high level)
- Trend model: for each ratio \( r \), fit \( r = a + b \cdot t \) (t = time/temperature). Slope \( b \) indicates increase/decrease over heating.  
- ANOVA: tests whether group means differ significantly; reports F statistic and p-value (p < 0.05 suggests group differences).

## Metrics and interpretation
- **Slope**: sign and magnitude reflect degradation direction/rate.  
- **R (corr)** or **R²** (if reported): strength of linear trend.  
- **ANOVA p-value**: evidence of group differences; small p suggests statistically distinct behaviors.  
- **Ratios table**: key_ratios per sample; inspect variability.

## Workflow usage (CLI)
```bash
foodspec heating \
  libraries/heating_demo.h5 \
  --time-column heating_time \
  --output-dir runs/heating_demo
```
Outputs: ratios.csv, trend summaries, optional anova.csv, ratio_vs_time.png, report.md.

## Reporting guidance (MethodsX style)
- **Main figures**: ratio vs time plot with fitted trend line; annotate slope and R or p-value.  
- **Main text**: describe preprocessing choices and key ratio(s) used, with slope/p-values.  
- **Supplementary**: full ratios table; ANOVA table; spectra before/after heating; per-time-point stats.  
- **Supporting tests** (conceptual): peroxide value, anisidine value, sensory panel, or GC–MS profiling for corroboration.
