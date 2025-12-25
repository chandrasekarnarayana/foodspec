# Workflow: Heating & Quality Monitoring

> New to workflow design? See [Designing & reporting workflows](workflow_design_and_reporting.md).
> For model and evaluation choices, see [ML & DL models](../ml/models_and_best_practices.md) and [Metrics & evaluation](../../metrics/metrics_and_evaluation/).

Heating/frying alters oil composition (oxidation, polymerization, loss of unsaturation). This workflow quantifies spectral markers over time/temperature to monitor quality and degradation.

Suggested visuals: ratio vs time with fit, box/violin plots by stage, and correlation plots. See [Plots guidance](workflow_design_and_reporting.md#plots-visualizations).
For troubleshooting (e.g., low SNR, metadata gaps), see [Common problems & solutions](../troubleshooting/common_problems_and_solutions.md).

```mermaid
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
```

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
```python
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
```

## 4. CLI example (with config)
Create `examples/configs/heating_quality_quickstart.yml`:
```yaml
input_hdf5: libraries/oils_heating.h5
time_column: heating_time
output_dir: runs/heating_demo
```
Run:
```bash
foodspec heating --config examples/configs/heating_quality_quickstart.yml
```
Outputs: ratio CSV, optional ANOVA CSV, ratio_vs_time.png, report.md.

## 5. Interpretation
- Report slope and confidence (p-value or R²) for key ratios; note direction (e.g., decreasing unsaturation ratio indicates oxidation).
- If groups (oil types) exist, compare trends or ANOVA at endpoints.
- Main figure: ratio vs time with fitted line. Supplement: ANOVA table, spectra snapshots.

### Qualitative & quantitative interpretation
- **Qualitative:** Ratio vs time plots reveal whether degradation markers rise/fall; optional PCA scores can show separation of early vs late stages.
- **Quantitative:** Report slope/p-value and R² from trend models; ANOVA/ANCOVA p-values and effect sizes for grouped stages (see [ANOVA/MANOVA](../stats/anova_and_manova.md)); silhouette on PCA (if used) for stage structure.
- **Reviewer phrasing:** “The unsaturation ratio decreases with heating time (slope = …, p < …); grouped ANOVA confirms stage differences (p < …); PCA shows partial separation of early vs late stages (silhouette ≈ …).”

## Summary
- Track unsaturation/oxidation markers via ratios over time/temperature.
- Use simple linear models; verify significance and direction of trends.
- Provide plots and statistics to support quality decisions or reporting.

## Statistical analysis
- **Why:** Test whether degradation markers change with heating; quantify slope significance.
- **Example (correlation/linear fit):**
```python
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
- [Baseline correction](../../preprocessing/baseline_correction/)
- [Normalization & smoothing](../../preprocessing/normalization_smoothing/)
- [Derivatives & feature enhancement](../../preprocessing/derivatives_and_feature_enhancement/)
- [Model evaluation](../ml/model_evaluation_and_validation.md)
