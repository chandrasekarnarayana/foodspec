# Calibration / Regression Example

## 📋 Standard Header

**Purpose:** Build calibration models to predict continuous quality metrics (mixture fractions, degradation scores) from spectra.

**When to Use:**
- Predict adulterant concentration in blends
- Estimate degradation score (peroxide value, oxidation index)
- Quantify moisture content or impurity levels
- Calibrate spectral features to reference lab measurements
- Validate spectroscopic methods against gold-standard analyses

**Inputs:**
- Format: HDF5 or CSV with spectra + reference values
- Required metadata: `target_value` (continuous quality metric)
- Optional metadata: `batch`, `replicate_id`, `reference_method`
- Wavenumber range: Same as classification workflows (600–1800 cm⁻¹)
- Min samples: 50+ with diverse target values (span full quality range)

**Outputs:**
- calibration_curve.png — Predicted vs true scatter with diagonal
- residual_plot.png — Residuals vs predicted values
- metrics.json — RMSE, MAE, R², MAPE
- model.pkl — Trained PLS or MLP regressor
- report.md — Calibration performance and prediction uncertainty

**Assumptions:**
- Target values measured accurately (low reference method error)
- Linear or mildly non-linear relationship between spectra and target
- Training samples span full operational range of target values
- No extrapolation beyond training range (predictions within calibrated domain)

---

## 🔬 Minimal Reproducible Example (MRE)

```python
import numpy as np
import matplotlib.pyplot as plt
from foodspec.chemometrics.models import make_pls_regression, make_mlp_regressor
from foodspec.chemometrics.validation import compute_regression_metrics
from foodspec.viz.regression import plot_calibration_curve, plot_residual_plot
from foodspec.stats import bootstrap_metric
from sklearn.model_selection import cross_val_predict

# Generate synthetic regression data
np.random.seed(42)
n_samples, n_features = 120, 15
X = np.random.normal(0, 1, size=(n_samples, n_features))
true_coefs = np.random.normal(0.4, 0.2, size=n_features)
y = X @ true_coefs + np.random.normal(0, 0.4, size=n_samples)

print(f"Samples: {n_samples}")
print(f"Features: {n_features}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")

# PLS Regression (linear baseline)
model_pls = make_pls_regression(n_components=5)
model_pls.fit(X, y)
y_pred_pls = cross_val_predict(model_pls, X, y, cv=5)

metrics_pls = compute_regression_metrics(y, y_pred_pls)
print(f"\nPLS Regression Metrics:")
print(f"  RMSE: {metrics_pls['rmse']:.3f}")
print(f"  MAE: {metrics_pls['mae']:.3f}")
print(f"  R²: {metrics_pls['r2']:.3f}")
print(f"  MAPE: {metrics_pls['mape']:.1f}%")

# MLP Regression (non-linear option)
model_mlp = make_mlp_regressor(
    hidden_layer_sizes=(64, 32),
    max_iter=400,
    random_state=0
)
model_mlp.fit(X, y)
y_pred_mlp = cross_val_predict(model_mlp, X, y, cv=5)

metrics_mlp = compute_regression_metrics(y, y_pred_mlp)
print(f"\nMLP Regression Metrics:")
print(f"  RMSE: {metrics_mlp['rmse']:.3f}")
print(f"  MAE: {metrics_mlp['mae']:.3f}")
print(f"  R²: {metrics_mlp['r2']:.3f}")
print(f"  MAPE: {metrics_mlp['mape']:.1f}%")

# Bootstrap confidence intervals
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

boot_pls = bootstrap_metric(
    rmse,
    y,
    y_pred_pls,
    n_bootstrap=500,
    random_state=0
)
print(f"\nPLS RMSE Bootstrap CI: [{boot_pls['ci'][0]:.3f}, {boot_pls['ci'][1]:.3f}]")

# Plot calibration curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PLS calibration
plot_calibration_curve(y, y_pred_pls, ax=axes[0])
axes[0].set_title(f"PLS Calibration (R²={metrics_pls['r2']:.3f})")
axes[0].set_xlabel("True Value")
axes[0].set_ylabel("Predicted Value")

# Residual plot
plot_residual_plot(y_pred_pls, y - y_pred_pls, ax=axes[1])
axes[1].set_title("Residual Plot")
axes[1].set_xlabel("Predicted Value")
axes[1].set_ylabel("Residual (True - Predicted)")
axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("calibration_example.png", dpi=150, bbox_inches='tight')
print("\nSaved: calibration_example.png")
```

**Expected Output:**
```yaml
Samples: 120
Features: 15
Target range: -2.87 to 3.45

PLS Regression Metrics:
  RMSE: 0.432
  MAE: 0.342
  R²: 0.887
  MAPE: 24.3%

MLP Regression Metrics:
  RMSE: 0.398
  MAE: 0.315
  R²: 0.905
  MAPE: 22.1%

PLS RMSE Bootstrap CI: [0.385, 0.483]

Saved: calibration_example.png
```

---

## ✅ Validation & Sanity Checks

### Success Indicators

**Calibration Curve:**
- ✅ Points cluster tightly around diagonal (y = x)
- ✅ No systematic bias (residuals centered at zero)
- ✅ Prediction error uniform across target range (homoscedastic)

**Metrics:**
- ✅ R² > 0.85 (85% variance explained)
- ✅ RMSE < 10% of target range
- ✅ MAPE < 15% (mean absolute percentage error)

**Residuals:**
- ✅ Residuals normally distributed (Q-Q plot linear)
- ✅ No heteroscedasticity (residual variance constant)
- ✅ No outliers beyond 3 SD

### Failure Indicators

**⚠️ Warning Signs:**

1. **Calibration curve shows S-shape (non-linearity)**
   - Problem: PLS insufficient; relationship non-linear
   - Fix: Try MLP regressor; add polynomial features; check for saturation effects

2. **R² < 0.70**
   - Problem: Poor predictive power; target not correlated with spectra
   - Fix: Check preprocessing; verify target values correct; increase sample size

3. **Residuals increase with predicted value (cone shape)**
   - Problem: Heteroscedasticity; model uncertainty higher at extremes
   - Fix: Transform target (log); use weighted regression; collect more samples at extremes

4. **Large outliers (residuals > 3 SD)**
   - Problem: Reference measurement error; sample mislabeling; matrix effects
   - Fix: Investigate outliers; remove if justified; check sample quality

5. **MLP RMSE >> PLS RMSE (overfitting)**
   - Problem: MLP too complex; insufficient regularization
   - Fix: Reduce hidden layer size; increase dropout; use PLS instead

### Quality Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|--------|
| R² | 0.75 | 0.88 | 0.95 |
| RMSE (% of range) | < 15% | < 8% | < 5% |
| MAPE | < 20% | < 12% | < 8% |
| Residual Normality (Shapiro p) | > 0.05 | > 0.10 | > 0.20 |

---

## ⚙️ Parameters You Must Justify

### Critical Parameters

**1. Regression Method**
- **Parameter:** Model type (PLS, Ridge, MLP)
- **Default:** PLS (linear baseline)
- **When to adjust:** Use MLP if calibration curve clearly non-linear; always benchmark against PLS
- **Justification:** "PLS regression (5 components) used as baseline; captures linear relationships while avoiding overfitting."

**2. Number of PLS Components**
- **Parameter:** `n_components`
- **Default:** 5–10
- **When to adjust:** Use cross-validation to choose; increase if underfitting
- **Justification:** "Five PLS components chosen via cross-validation; captures 92% cumulative variance in X."

**3. Cross-Validation Strategy**
- **Parameter:** `cv` (number of folds)
- **Default:** 5-fold CV
- **Justification:** "Five-fold cross-validation used to estimate unbiased prediction error."

**4. Target Value Range**
- **Parameter:** Min/max of training targets
- **Critical:** Must report; predictions outside range unreliable
- **Justification:** "Calibration valid for target values 0.5–5.0 (training range); extrapolation beyond this range not recommended."

**5. Bootstrap Iterations**
- **Parameter:** `n_bootstrap` (for confidence intervals)
- **Default:** 500
- **Justification:** "Bootstrap confidence intervals (500 iterations) quantify prediction uncertainty."

---

## Data and setup
- Synthetic spectral features are used here for illustration; replace with real ratios/PCs in practice.
- Preprocessing would normally precede this step (baseline, smoothing, normalization).

## Code example (PLS regression + metrics + robustness)
```python
import numpy as np
from foodspec.chemometrics.models import make_pls_regression
from foodspec.chemometrics.validation import compute_regression_metrics
from foodspec.stats import bootstrap_metric, permutation_test_metric

rng = np.random.default_rng(42)
n_samples, n_features = 120, 15
X = rng.normal(0, 1, size=(n_samples, n_features))
true_coefs = rng.normal(0.4, 0.2, size=n_features)
y = X @ true_coefs + rng.normal(0, 0.4, size=n_samples)

model = make_pls_regression(n_components=5)
model.fit(X, y)
y_pred = model.predict(X).ravel()

metrics = compute_regression_metrics(y, y_pred)
print(metrics)  # RMSE, MAE, R^2

# Optional: MLP regression if non-linear bias persists
from foodspec.chemometrics.models import make_mlp_regressor

mlp = make_mlp_regressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0)
mlp.fit(X, y)
y_pred_mlp = mlp.predict(X)
mlp_metrics = compute_regression_metrics(y, y_pred_mlp)
print("MLP metrics:", mlp_metrics)

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

boot = bootstrap_metric(rmse, y, y_pred, n_bootstrap=500, random_state=0)
perm = permutation_test_metric(rmse, y, y_pred, n_permutations=500, metric_higher_is_better=False, random_state=0)
print("Bootstrap CI:", boot["ci"], "Permutation p-value:", perm["p_value"])
```

![Regression calibration plot: predicted vs true values](../../assets/regression_calibration.png)

*Figure: Predicted vs true values for a PLS regression on synthetic data. Points close to the diagonal indicate good calibration; systematic deviation signals bias. Generated via `docs/examples/ml/generate_regression_calibration_figure.py`.*

Optionally add uncertainty/agreements (DL optional—use only with sufficient data and always benchmark against PLS/linear baselines):

```python
from foodspec.viz import plot_calibration_with_ci, plot_bland_altman

ax = plot_calibration_with_ci(y_true, y_pred)
ax.figure.savefig("calibration_ci.png", dpi=150)
ax = plot_bland_altman(y_true, y_pred)
ax.figure.savefig("bland_altman.png", dpi=150)
```

## Reporting
- Report RMSE/MAE/R² with confidence intervals (bootstrap) and, if needed, permutation p-values for chance-level checks.
- Include predicted-vs-true plots and residual diagnostics for transparency.
- Note preprocessing steps, feature choices (ratios/PCs), model settings (components), and validation design.

### Qualitative & quantitative interpretation
- **Qualitative:** Predicted vs true should cluster around the 1:1 line; residuals should be structureless and homoscedastic.
- **Quantitative:** Report RMSE/MAE/R² (and adjusted R² if multiple predictors); consider bootstrap CIs and permutation checks for small n. Add CI bands on calibration plots (`plot_calibration_with_ci`) and, when comparing methods, use Bland–Altman to assess agreement (bias, limits). Link to [Metrics & evaluation](../../reference/metrics_reference.md) and [Hypothesis testing](../../methods/statistics/hypothesis_testing_in_food_spectroscopy.md) for supporting stats.
- **Reviewer phrasing:** “Calibration achieved R² = … and RMSE = …; residuals show no trend with fitted values, suggesting adequate model form.”
---

## When Results Cannot Be Trusted

⚠️ **Red flags for calibration/regression workflow:**

1. **Extremely high R² (0.99+) on small dataset without independent validation**
   - High R² on training data doesn't guarantee generalization
   - Overfitting: model learns noise, not true relationship
   - **Fix:** Use cross-validation or hold-out test set; bootstrap confidence intervals on R²

2. **Calibration range too narrow (model trained on 10–20% property range, deployed on 0–50% range)**
   - Linear relationships valid only in training range
   - Extrapolation beyond training range produces unreliable predictions
   - **Fix:** Ensure calibration samples span full operational range; mark and test extrapolation regions separately

3. **Calibration standards from single source (all "low", "medium", "high" from same batch/supplier)**
   - Intra-source variability unknown; model may learn supplier-specific patterns
   - Different sources with same property value may have different spectra
   - **Fix:** Include multiple sources per property level; validate on independent reference materials

4. **Residuals show systematic trend with fitted values (heteroscedasticity)**
   - Violates homogeneity assumption; confidence intervals unreliable
   - May indicate nonlinear relationship or missing variable
   - **Fix:** Visualize residuals vs fitted; log-transform if variance increases; consider nonlinear model

5. **No replication in calibration (each standard measured once)**
   - Measurement error unquantified; precision of calibration unknown
   - Single outlier can disproportionately influence fit
   - **Fix:** Measure each standard ≥3 times; report residual SD; use robust regression

6. **Calibration model not validated on new samples (RMSE only computed on training data)**
   - Training metrics optimistic; test set RMSE is ground truth
   - Real-world performance may be worse
   - **Fix:** Hold out independent test set; cross-validate; measure RMSE on truly new samples

7. **Reference method uncertainty not considered (assuming reference measurements are error-free)**
   - Reference method has uncertainty; can't achieve better precision than reference
   - Model R² inflated if reference error ignored
   - **Fix:** Quantify reference method error; report measurement uncertainty for calibration samples

8. **Instrumental drift over calibration period not checked (calibration measured over weeks with no QC)**
   - Drift shifts spectral baselines; affects all samples
   - Calibration may fit drift, not true property relationship
   - **Fix:** Include QC standards throughout calibration; check for time-dependent drift; recalibrate periodically
## See also
- [Classification & regression](../../methods/chemometrics/classification_regression.md)
- [Metrics & evaluation](../../reference/metrics_reference.md)
- [Workflow design](../workflow_design_and_reporting.md)
- [Stats: nonparametric & robustness](../../methods/statistics/nonparametric_methods_and_robustness.md)
