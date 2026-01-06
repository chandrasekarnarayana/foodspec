# Workflow: Heating & Quality Monitoring

## 📋 Standard Header

**Purpose:** Quantify spectral degradation markers in oils over heating time/temperature to monitor oxidation and quality decline.

**When to Use:**
- Track frying oil degradation across time/temperature cycles
- Identify safe heating limits for regulatory compliance
- Study oxidation kinetics in accelerated aging experiments
- Monitor batch-to-batch thermal stability differences
- Validate antioxidant effectiveness in thermal stress tests

**Inputs:**
- Format: HDF5 spectral library or CSV with wavenumber columns
- Required metadata: `heating_time` (hours) OR `temperature` + `time`
- Optional metadata: `oil_type`, `replicate_id`, `batch`, `treatment` (antioxidant, etc.)
- Wavenumber range: 600–1800 cm⁻¹ (focus on C=O 1742, C=C 1655 cm⁻¹)
- Min samples: 5–20 time points × 3+ replicates per time (15–60 spectra)

**Outputs:**
- ratio_vs_time.png — Trend plot with fitted regression line and confidence bands
- ratio_table.csv — Calculated ratios (e.g., 1655/1742) at each time point
- trend_models.json — Slope, intercept, R², p-value for each ratio
- anova_results.csv — (Optional) Group-wise comparison if multiple oil types
- report.md — Narrative with interpretation and quality recommendations

**Assumptions:**
- Temperature controlled or monitored (consistent heating conditions)
- Samples independent (not repeated scans of same oil; 3+ distinct replicates)
- Baseline and normalization applied consistently across all time points
- No confounding factors (moisture, oxygen, light) varying systematically with time

---

## 🔬 Minimal Reproducible Example (MRE)

### Option A: Bundled Synthetic Data

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from foodspec.apps.heating import run_heating_quality_workflow
from foodspec.demo import synthetic_heating_dataset
from foodspec.viz.heating import plot_ratio_vs_time

# Generate synthetic heating data (0-8 hours, oxidation trend)
fs = synthetic_heating_dataset()
print(f"Loaded: {fs.x.shape[0]} spectra across {fs.metadata['heating_time'].nunique()} time points")
print(f"Time range: {fs.metadata['heating_time'].min()}-{fs.metadata['heating_time'].max()} hours")

# Run complete workflow
result = run_heating_quality_workflow(fs, time_column="heating_time")

# Extract key ratio (unsaturation/carbonyl)
ratio_name = result.key_ratios.columns[0]
ratio_values = result.key_ratios[ratio_name]
time_values = fs.metadata["heating_time"]

# Display trend model
model = result.trend_models.get(ratio_name)
print(f"\n{ratio_name} Trend Model:")
print(f"  Slope: {model.slope:.4f} per hour")
print(f"  R²: {model.r_squared:.3f}")
print(f"  p-value: {model.p_value:.1e}")

# Plot ratio vs time with fitted line
fig, ax = plt.subplots(figsize=(8, 6))
plot_ratio_vs_time(time_values, ratio_values, model=model, ax=ax)
ax.set_title(f"Oil Degradation: {ratio_name}")
ax.set_xlabel("Heating Time (hours)")
ax.set_ylabel(f"{ratio_name} Ratio")
ax.grid(alpha=0.3)
plt.tight_layout()
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "heating_ratio_vs_time.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/heating_ratio_vs_time.png")

# If groups present (e.g., multiple oil types), run ANOVA
if 'oil_type' in fs.metadata.columns:
    from foodspec.stats import run_one_way_anova
    anova_res = run_one_way_anova(
        ratio_values,
        groups=fs.metadata['oil_type']
    )
    print(f"\nANOVA (ratio ~ oil_type): F={anova_res.f_stat:.2f}, p={anova_res.p_value:.1e}")
```

![Heating ratio over time](../../assets/workflows/heating_quality_monitoring/heating_ratio_vs_time.png)
```yaml
Loaded: 24 spectra across 8 time points
Time range: 0-8 hours

ratio_1655_1742 Trend Model:
  Slope: -0.0125 per hour
  R²: 0.892
  p-value: 1.2e-06

Saved: heating_ratio_vs_time.png
```

### Option B: Custom Synthetic Generator

```python
import numpy as np
import pandas as pd
from foodspec import SpectralDataset

def generate_synthetic_heating(n_times=8, n_replicates=3, random_state=42):
    """Generate synthetic oil spectra showing oxidation over heating time."""
    np.random.seed(random_state)
    
    wavenumbers = np.linspace(600, 1800, 400)
    time_points = np.linspace(0, 8, n_times)  # 0-8 hours
    
    spectra = []
    metadata = []
    
    for t in time_points:
        for rep in range(n_replicates):
            # Base spectrum with characteristic peaks
            spectrum = np.zeros(len(wavenumbers))
            
            # C=C stretch (1655 cm⁻¹) - decreases with heating
            unsaturation = 1.8 * (1 - 0.1 * t) * np.exp(-((wavenumbers - 1655) ** 2) / 2000)
            
            # C=O stretch (1742 cm⁻¹) - increases with oxidation
            carbonyl = (1.2 + 0.08 * t) * np.exp(-((wavenumbers - 1742) ** 2) / 1800)
            
            # CH2 bending (1450 cm⁻¹) - relatively stable
            ch2_bend = 1.5 * np.exp(-((wavenumbers - 1450) ** 2) / 1500)
            
            spectrum = unsaturation + carbonyl + ch2_bend
            
            # Add noise and batch variability
            noise = np.random.normal(0, 0.05, len(wavenumbers))
            batch_effect = np.random.normal(0, 0.02)
            spectrum = spectrum + noise + batch_effect
            
            spectra.append(spectrum)
            metadata.append({
                'heating_time': t,
                'replicate_id': f"rep{rep+1}",
                'batch': 'A'
            })
    
    # Create DataFrame
    df = pd.DataFrame(
        np.array(spectra),
        columns=[f"{w:.1f}" for w in wavenumbers]
    )
    for col, values in pd.DataFrame(metadata).items():
        df.insert(len(metadata[0]) - list(metadata[0].keys()).index(col) - 1, col, values)
    
    # Convert to SpectralDataset
    dataset = SpectralDataset.from_dataframe(
        df,
        metadata_columns=list(metadata[0].keys()),
        intensity_columns=[f"{w:.1f}" for w in wavenumbers],
        wavenumber=wavenumbers
    )
    
    return dataset

# Generate and use
fs_heating = generate_synthetic_heating(n_times=8, n_replicates=3)
print(f"Generated: {fs_heating.x.shape[0]} synthetic heating spectra")
```

---

## ✅ Validation & Sanity Checks

### Success Indicators

**Trend Plot (Ratio vs Time):**
- ✅ Clear monotonic trend (increasing or decreasing)
- ✅ Confidence bands narrow around fitted line (R² > 0.70)
- ✅ Replicate scatter modest (CV < 15% at each time point)

**Statistical Significance:**
- ✅ p-value < 0.05 for trend slope (significant degradation)
- ✅ R² > 0.70 (trend explains most variability)
- ✅ Residuals normally distributed (Q-Q plot linear)

**Chemical Plausibility:**
- ✅ Unsaturation ratio (1655/1742) decreases with heating (oxidation expected)
- ✅ Carbonyl peak (1742) increases (oxidation products form)
- ✅ Slope magnitude matches literature (e.g., -0.01 to -0.02 per hour for typical oils)

**Replication:**
- ✅ 3+ replicates per time point show consistent values (error bars < 10% of mean)
- ✅ No outliers more than 3 SD from group mean
- ✅ Technical replicates (same oil) averaged before analysis

### Failure Indicators

**⚠️ Warning Signs:**

1. **Trend non-monotonic (ratio increases, then decreases, or vice versa)**
   - Problem: Confounding factor (temperature spikes, contamination) or wrong ratio direction
   - Fix: Check temperature logs; verify ratio definition (numerator/denominator correct); inspect raw spectra

2. **High scatter, low R² (< 0.50)**
   - Problem: Biological variability too large; baseline/normalization issues; insufficient replication
   - Fix: Increase replicates; check preprocessing consistency; stratify by oil source

3. **p-value > 0.05 but visual trend obvious**
   - Problem: Underpowered (too few samples); high within-group variability
   - Fix: Increase time points or replicates; check for outliers inflating variance

4. **Slope sign opposite to expectation (unsaturation increases with heating)**
   - Problem: Ratio inverted; preprocessing artifact; wrong peak assignment
   - Fix: Verify peak positions (plot raw spectra); check ratio numerator/denominator; confirm baseline correction applied

5. **All time points identical (ratio flat, slope ≈ 0)**
   - Problem: Heating had no effect (experiment failed); ratio insensitive to oxidation; wrong spectral region
   - Fix: Verify heating occurred (temperature records); try alternative ratios (1742/1450, 1655/1450); check if oil pre-oxidized

6. **Confidence bands very wide (span > 50% of mean ratio)**
   - Problem: High within-group variability; too few replicates
   - Fix: Increase n per time; remove outliers; check instrument drift

### Quality Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Trend R² | 0.50 | 0.75 | 0.90 |
| Trend p-value | < 0.05 | < 0.01 | < 0.001 |
| Within-Time CV | < 20% | < 10% | < 5% |
| Replicates per Time | 2 | 3 | 5+ |
| Residuals Normality (Shapiro p) | > 0.05 | > 0.10 | > 0.20 |

---

## ⚙️ Parameters You Must Justify

### Critical Parameters (Report in Methods)

**1. Ratio Definition**
- **Parameter:** Numerator/denominator wavenumbers
- **Default:** `ratio_1655_1742` (unsaturation/carbonyl)
- **When to adjust:**
  - Use 1742/1450 (carbonyl/CH2) if interested in oxidation products only
  - Use 1655/1450 (unsaturation/CH2) if carbonyl varies too much
- **Justification template:**
  > "The ratio of peak heights at 1655 cm⁻¹ (C=C stretch) to 1742 cm⁻¹ (C=O stretch) was used as an oxidation marker, as unsaturation decreases and carbonyl increases with thermal degradation (Guillen & Cabo, 1997)."

**2. Baseline Correction (ALS)**
- **Parameter:** `lam` (smoothness), `p` (asymmetry)
- **Default:** lam=1e4, p=0.01
- **When to adjust:**
  - Increase `lam` (1e5) if background curvature strong
  - Decrease `p` (0.001) if fluorescence dominates
- **Justification template:**
  > "Asymmetric Least Squares baseline correction (λ=1e4, p=0.01) removed background curvature consistently across all time points."

**3. Smoothing (Savitzky-Golay)**
- **Parameter:** `window_length`, `polyorder`
- **Default:** window=21, polyorder=3
- **When to adjust:**
  - Increase window (31) if very noisy
  - Decrease window (11) if peaks narrow
- **Justification template:**
  > "Savitzky-Golay smoothing (window=21, polynomial order=3) reduced high-frequency noise while preserving peak positions."

**4. Normalization**
- **Parameter:** Method (SNV, L2, minmax)
- **Default:** L2 (unit vector)
- **When to adjust:**
  - Use SNV if baseline variability persists
  - Use minmax if absolute peak heights needed for ratios
- **Justification template:**
  > "Spectra were normalized to unit L2 norm to remove intensity scaling artifacts while preserving relative peak heights."

**5. Trend Model Type**
- **Parameter:** Linear, polynomial, exponential
- **Default:** Linear regression
- **When to adjust:**
  - Use polynomial (degree=2) if degradation plateaus at long times
  - Use exponential if first-order kinetics expected
- **Justification template:**
  > "Linear regression was fit to ratio vs heating time to quantify degradation rate (slope) and significance (p-value)."

**6. Statistical Test**
- **Parameter:** Pearson correlation, linear model p-value, ANOVA
- **Default:** Linear model p-value (slope ≠ 0)
- **When to adjust:**
  - Use Pearson correlation if only testing association (not causation)
  - Use ANOVA if comparing grouped stages (early/mid/late)
- **Justification template:**
  > "Significance of the trend was assessed via p-value for the regression slope (H₀: slope = 0); p < 0.05 indicated significant degradation."

### Optional Parameters (Mention if Changed)

**Replication Strategy:**
- Number of independent samples per time point (3+ recommended)
- Technical replicates (averaged before analysis)

**Time Range:**
- Start time (0 hours = fresh oil)
- End time (when to stop heating; QA limit)

**ANOVA (if groups present):**
- Grouping variable (oil_type, treatment)
- Post-hoc test (Tukey, Bonferroni) if ANOVA significant

---
flowchart LR
  subgraph Data
    A[Raw spectra] --> A2[Heating metadata (time/temp)]
  end
  subgraph Preprocess
    B[Baseline + smoothing + norm + crop]
  end
  subgraph Features
    C[Peak ratios (e.g., 1655/1742) ± PCA]
  end
  subgraph Model/Stats
    D[Trend models (linear/ANCOVA) + ANOVA]
    E[Metrics: slope, R², p-values; plots]
  end
  subgraph Report
    F[Ratio vs time + stats tables + report.md]
  end
  A --> B --> C --> D --> E --> F
  A2 --> D
```yaml

## 1. Problem and dataset
- **Why labs care:** Regulatory/QA limits on frying lifetime; detecting off-spec batches; studying oxidation kinetics.
- **Inputs:** Spectra with metadata column for `heating_time` or stage/temperature. Wavenumbers typically cropped to 600–1800 cm⁻¹.
- **Typical size:** Time series across 5–20 points; multiple replicates per time to assess variability.

## 2. Pipeline (default)
- **Preprocessing:** ALS baseline → Savitzky–Golay → L2 normalization → crop to 600–1800 cm⁻¹.
- **Features:** Key ratio `ratio_1655_1742` (unsaturation vs carbonyl band). Additional ratios can be added for specific matrices.
- **Models:** Linear regression of ratio vs time; optional group-wise models if `oil_type` present; ANOVA across groups for end-point differences.
- **Outputs:** Ratio table, fitted slopes/intercepts, optional ANOVA p-values.

## 3. Python example (synthetic)
```
from foodspec.apps.heating import run_heating_quality_workflow
from foodspec.viz.heating import plot_ratio_vs_time
import matplotlib.pyplot as plt

# See examples/heating_quality_quickstart.py for full synthetic data creation
from examples.heating_quality_quickstart import _synthetic_heating_dataset

fs = _synthetic_heating_dataset()
res = run_heating_quality_workflow(fs, time_column="heating_time")
ratio_name = res.key_ratios.columns[0]
model = res.trend_models.get(ratio_name)

fig, ax = plt.subplots()
plot_ratio_vs_time(fs.metadata["heating_time"], res.key_ratios[ratio_name], model=model, ax=ax)
fig.savefig("heating_ratio_vs_time.png", dpi=150)
```yaml

## 4. CLI example (with config)
Create `examples/configs/heating_quality_quickstart.yml`:
```
input_hdf5: libraries/oils_heating.h5
time_column: heating_time
output_dir: runs/heating_demo
```bash
Run:
```
foodspec heating --config examples/configs/heating_quality_quickstart.yml
```yaml
Outputs: ratio CSV, optional ANOVA CSV, ratio_vs_time.png, report.md.

## 5. Interpretation
- Report slope and confidence (p-value or R²) for key ratios; note direction (e.g., decreasing unsaturation ratio indicates oxidation).
- If groups (oil types) exist, compare trends or ANOVA at endpoints.
- Main figure: ratio vs time with fitted line. Supplement: ANOVA table, spectra snapshots.

### Qualitative & quantitative interpretation
- **Qualitative:** Ratio vs time plots reveal whether degradation markers rise/fall; optional PCA scores can show separation of early vs late stages.
- **Quantitative:** Report slope/p-value and R² from trend models; ANOVA/ANCOVA p-values and effect sizes for grouped stages (see [ANOVA/MANOVA](../../methods/statistics/anova_and_manova.md)); silhouette on PCA (if used) for stage structure.
- **Reviewer phrasing:** “The unsaturation ratio decreases with heating time (slope = …, p < …); grouped ANOVA confirms stage differences (p < …); PCA shows partial separation of early vs late stages (silhouette ≈ …).”

## Summary
- Track unsaturation/oxidation markers via ratios over time/temperature.
- Use simple linear models; verify significance and direction of trends.
- Provide plots and statistics to support quality decisions or reporting.

## Statistical analysis
- **Why:** Test whether degradation markers change with heating; quantify slope significance.
- **Example (correlation/linear fit):**
```
from foodspec.stats import compute_correlations
from foodspec.apps.heating import run_heating_quality_workflow
from examples.heating_quality_quickstart import _synthetic_heating_dataset

fs = _synthetic_heating_dataset()
res = run_heating_quality_workflow(fs, time_column="heating_time")
ratio = res.key_ratios.iloc[:, 0]
corr = compute_correlations(
    pd.DataFrame({"ratio": ratio, "time": fs.metadata["heating_time"]}),
    ("ratio", "time"),
    method="pearson",
)
print(corr)
```
- **Interpretation:** Significant negative/positive correlation implies the ratio changes with time (degradation/oxidation). Report slope and p-value from the trend model; use ANOVA across grouped stages if discretized.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for heating quality monitoring workflow:**

1. **Heating experiment conducted without temperature monitoring (assuming oven temperature is constant)**
   - Temperature variation causes spectral changes independent of chemical degradation
   - Can't distinguish heating effects from temperature effects
   - **Fix:** Monitor oven/oil temperature throughout experiment; report actual temperature profile

2. **Single oil sample heated repeatedly, spectra treated as independent replicates**
   - Repeated scans of same sample are autocorrelated, not independent
   - Statistical tests assuming independence produce inflated significance
   - **Fix:** Include ≥3 distinct oil samples; average technical replicates before analysis

3. **Ratios used without baseline correction or normalization (peak heights compared directly)**
   - Baseline shifts can create apparent ratio changes
   - Normalization differences between time points affect interpretation
   - **Fix:** Apply consistent baseline correction and normalization to all spectra; use corrected peaks/ratios

4. **No control for natural oil variability (all oils from same source/variety)**
   - Inter-source variability in unheated oils unknown
   - Can't distinguish heating changes from source differences
   - **Fix:** Include oils from different sources; quantify baseline variability before heating

5. **Heating trend extrapolated beyond measured times (model trained on 0–2 hours, predicting 10-hour stability)**
   - Extrapolation assumes trend continues linearly; may plateau, accelerate, or reverse
   - Real degradation kinetics may be non-monotonic
   - **Fix:** Only infer within measured time range; test extended heating if predictions needed

6. **No moisture/oxygen control (heating in open vs sealed container, humidity varies)**
   - Oxygen availability affects oxidation rates; moisture affects hydrolysis
   - Confounding factors dominate spectrum changes
   - **Fix:** Control atmosphere (sealed, N₂ atmosphere, or open with defined airflow); document conditions

7. **Statistical significance mistaken for practical quality change (p < 0.05 ratio change, but <1% magnitude)**
   - Tiny changes can be statistically significant with enough replication
   - Practically, oil may still be acceptable
   - **Fix:** Report effect sizes alongside p-values; define actionable quality thresholds independent of statistics

8. **No replication or confidence intervals on trend (reporting mean ratio at each time, no variability bands)**
   - Variability across samples unknown; trend appears more certain than it is
   - Can't assess whether trend is consistent or noisy
   - **Fix:** Include error bars (± SD) or confidence bands; report n per timepoint; fit trend with CI

## Further reading
- [Baseline correction](../../methods/preprocessing/baseline_correction.md)
- [Normalization & smoothing](../../methods/preprocessing/normalization_smoothing.md)
- [Derivatives & feature enhancement](../../methods/preprocessing/derivatives_and_feature_enhancement.md)
- [Model evaluation](../../methods/chemometrics/model_evaluation_and_validation.md)
