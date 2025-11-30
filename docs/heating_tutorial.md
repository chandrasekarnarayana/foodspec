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
