# Heating Quality Monitoring: Time-Series Analysis

**Level**: Beginner → Intermediate  
**Runtime**: ~5 seconds  
**Key Concepts**: Time-series data, ratio extraction, trend modeling, degradation kinetics

---

## What You Will Learn

In this example, you'll learn how to:
- Load and structure time-series spectroscopy data
- Extract key ratios that indicate chemical degradation
- Fit trend models to quantify degradation rates
- Identify critical thresholds (e.g., rancidity onset)
- Estimate remaining shelf-life from degradation curves

After completing this example, you'll understand how to monitor food quality during storage, predict when products spoil, and establish quality control limits.

---

## Prerequisites

- Basic Python and Pandas knowledge
- Understanding of time-series concepts (temporal ordering, trends, forecasting)
- Familiarity with linear regression
- `numpy`, `pandas`, `matplotlib`, `scipy` installed

**Optional background**: Read [Heating Quality Monitoring Workflow](../workflows/quality-monitoring/heating_quality_monitoring.md)

---

## The Problem

**Real-world scenario**: You're monitoring oil quality during storage. Oxidation causes spectral changes that develop predictably over time. Can you:
1. Extract indicators (peak ratios) that increase with degradation?
2. Model the degradation rate?
3. Estimate when the oil becomes unsuitable (rancid)?

**Data**: Simulated time-series spectra from heating experiment (10 time points, oil degrading).

**Goal**: Extract trends, fit model, predict shelf-life.

---

## Step 1: Load and Explore Time-Series Data

```python
import numpy as np
import pandas as pd
from pathlib import Path

# Load synthetic heating data (simulated oil spectra over time)
# In practice, this comes from a heating oven + spectrometer setup
np.random.seed(42)
time_points = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])  # hours
n_wavelengths = 1500
n_replicates = 2

# Simulate degradation: peak intensities change over time
# Peak 1 (oxidation marker) increases; Peak 2 (original oil) decreases
degradation_curve = 0.05 * time_points + 0.1 * np.random.randn(len(time_points))
preservation_curve = 1.0 - 0.04 * time_points + 0.05 * np.random.randn(len(time_points))

# Key ratio: oxidation_marker / preservation_marker
ratio = degradation_curve / (preservation_curve + 0.1)  # avoid division by zero

df = pd.DataFrame({
    "time_hours": time_points,
    "oxidation_marker": degradation_curve,
    "preservation_marker": preservation_curve,
    "ratio": ratio
})

print(df)
print(f"\nRatio increases from {ratio[0]:.3f} to {ratio[-1]:.3f}")
```

**What's happening**:
- `time_points`: Measurements at 0, 2, 4, ... 18 hours
- `oxidation_marker`: Peaks that increase with heat (bad indicator)
- `preservation_marker`: Original peaks that decrease (good indicator)
- `ratio`: Key metric combining both signals

---

## Step 2: Extract Trends

```python
from scipy.stats import linregress

# Fit linear trend to the ratio
slope, intercept, r_value, p_value, std_err = linregress(df["time_hours"], df["ratio"])

print(f"Trend slope: {slope:.4f} ratio_units/hour")
print(f"R² (model fit): {r_value**2:.3f}")
print(f"p-value: {p_value:.4f}")

# Predict trend
df["ratio_trend"] = intercept + slope * df["time_hours"]

# Print trend predictions
print("\nObserved vs. Predicted:")
print(df[["time_hours", "ratio", "ratio_trend"]].to_string())
```

**Interpretation**:
- **Slope > 0**: Degradation is occurring (bad)
- **R² close to 1**: Trend fits well (reliable prediction)
- **p-value < 0.05**: Trend is statistically significant

---

## Step 3: Estimate Shelf-Life

```python
# Define rancidity threshold (when oil becomes unsuitable)
rancidity_threshold = 0.5  # example threshold

# When does the ratio reach this threshold?
time_to_rancidity = (rancidity_threshold - intercept) / slope if slope > 0 else np.inf

print(f"\nRancidity threshold: {rancidity_threshold}")
print(f"Estimated shelf-life: {time_to_rancidity:.1f} hours")
print(f"(Oil becomes rancid at t={time_to_rancidity:.1f}h)")

# Safety margin (e.g., recommend replacement at 80% of shelf-life)
recommended_replacement = 0.8 * time_to_rancidity
print(f"Recommended replacement: {recommended_replacement:.1f} hours")
```

**What's happening**:
- We solve: `threshold = intercept + slope * time`
- This gives us the time when oil becomes unsuitable
- We apply safety margin for practical use

---

## Step 4: Visualize Degradation Curve

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Time-series of ratios with trend
ax1.scatter(df["time_hours"], df["ratio"], color="red", s=100, label="Observed", zorder=3)
ax1.plot(df["time_hours"], df["ratio_trend"], "b--", linewidth=2, label="Linear trend")
ax1.axhline(rancidity_threshold, color="orange", linestyle=":", linewidth=2, label="Rancidity threshold")
ax1.axvline(time_to_rancidity, color="orange", linestyle=":", alpha=0.5)
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("Ratio (Oxidation / Preservation)")
ax1.set_title("Oil Degradation Over Time")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Forecast beyond measured time
time_extended = np.linspace(0, 25, 100)
ratio_forecast = intercept + slope * time_extended
ax2.plot(time_extended, ratio_forecast, "b-", linewidth=2, label="Trend extrapolation")
ax2.scatter(df["time_hours"], df["ratio"], color="red", s=100, label="Measured")
ax2.axhline(rancidity_threshold, color="orange", linestyle=":", linewidth=2, label="Rancidity")
ax2.axvline(time_to_rancidity, color="orange", linestyle=":", alpha=0.5)
ax2.fill_between([0, recommended_replacement], 0, 1.0, alpha=0.2, color="green", label="Safe zone")
ax2.fill_between([recommended_replacement, 25], 0, 1.0, alpha=0.2, color="red", label="Caution zone")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Ratio (Oxidation / Preservation)")
ax2.set_title("Shelf-Life Prediction")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig("heating_ratio_vs_time.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Figure interpretation**:
- **Left**: Raw data (red dots) + fitted trend (blue line)
- **Right**: Forecast into future, showing when rancidity is reached
- **Orange dashed line**: Rancidity threshold
- **Green zone**: Safe for consumption; **Red zone**: Spoiled

---

## Full Working Script

See the production script with complete heating experiment simulation:

📄 **[`examples/heating_quality_quickstart.py`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/heating_quality_quickstart.py)** – Full working code (42 lines)

---

## Generated Figure

![Heating Degradation Curve](https://github.com/chandrasekarnarayana/foodspec/raw/mahttps://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/heating_ratio_vs_time.png)

---

## Key Takeaways

✅ **Time-series workflow**: Load → Extract indicators → Fit trend → Forecast  
✅ **Ratio extraction**: Combining signals improves sensitivity to degradation  
✅ **Linear trends**: Simple but effective for many food quality processes  
✅ **Shelf-life estimation**: Predict spoilage using quantitative trends  

---

## Real-World Applications

- 🍳 **Oil monitoring**: Detect oxidation during storage or frying
- 🥛 **Milk freshness**: Track spoilage onset during refrigeration
- 🍎 **Fruit ripening**: Monitor maturity progression
- 🍞 **Bread staling**: Quantify texture changes over time
- 🧀 **Cheese aging**: Track flavor compound development

---

## Advanced Topics

**Want to go deeper?**
- **Non-linear trends**: Use polynomial or spline fitting for curved patterns
- **Uncertainty quantification**: Bootstrap confidence intervals for shelf-life estimates
- **Multi-component degradation**: Fit multiple ratios simultaneously
- **Predictive thresholds**: Learn critical ratios from historical data

See [Heating Quality Workflow](../workflows/quality-monitoring/heating_quality_monitoring.md) for complete details.

---

## Next Steps

1. **Try it**: Load your own time-series data and fit trends
2. **Explore**: Change threshold values and observe impact on shelf-life
3. **Learn more**: Read [Quality Monitoring Workflows](../workflows/quality-monitoring/)
4. **Advance**: Combine with [Hyperspectral Mapping](04_hyperspectral_mapping.md) for spatial + temporal analysis

---

## Interactive Notebook

For step-by-step exploration with parameter exploration:

📓 **[`examples/tutorials/02_heating_stability_teaching.ipynb`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/02_heating_stability_teaching.ipynb)**

---

## Figure provenance
- Generated by [scripts/generate_docs_figures.py](https://github.com/chandrasekarnarayana/foodspec/blob/main/scripts/generate_docs_figures.py)
- Outputs: [../assets/figures/heating_trend.png](../assets/figures/heating_trend.png) and [../assets/workflows/heating_quality_monitoring/heating_ratio_vs_time.png](../assets/workflows/heating_quality_monitoring/heating_ratio_vs_time.png)

