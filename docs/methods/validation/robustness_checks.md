# Robustness Checks

!!! abstract "Test Model Stability"
    A robust model performs consistently under **realistic perturbations**:
    
    - **Preprocessing variations:** Baseline tolerance, smoothing window, normalization parameters
    - **Outlier contamination:** Individual spectra with artifacts, spikes, or anomalous baselines
    - **Batch/day effects:** Performance stability across different measurement conditions
    - **Adversarial inputs:** Simulated adulteration, degradation, or matrix effects
    
    This page provides systematic protocols for stress-testing models before deployment.

---

## Why Robustness Testing Matters

### The Fragility Problem

Many high-performing models are **brittle**—they work well on clean validation data but fail under realistic operational conditions:

- **Preprocessing sensitivity:** Accuracy drops 10% when baseline smoothing window changes by ±2 points
- **Outlier vulnerability:** Single spike in spectrum misclassifies entire sample
- **Batch effects:** Model trained on Day 1-4 fails on Day 5 (temperature drift)
- **Matrix interference:** Model trained on pure oils fails on oil-in-sauce samples

!!! warning "Real-World Deployment Failure"
    A food authentication model with 94% validation accuracy **dropped to 68% in production** due to:
    
    1. Different operator technique (sample preparation variance)
    2. Seasonal temperature changes (baseline drift)
    3. New instrument batch (spectral resolution slightly different)
    
    None of these factors were tested during validation.

---

## Robustness Testing Framework

We recommend a **4-pillar robustness protocol**:

1. **Preprocessing Sensitivity Analysis** → Test parameter ranges
2. **Outlier Robustness** → Inject realistic artifacts
3. **Batch/Day Perturbations** → Test temporal/instrumental stability
4. **Adversarial Testing** → Simulate out-of-distribution samples

---

## 1. Preprocessing Sensitivity Analysis

### Goal
Quantify how performance changes when preprocessing parameters vary within reasonable ranges.

### Protocol

**Step 1: Define Parameter Ranges**

Identify critical preprocessing steps and their typical ranges:

| Preprocessing Step | Parameter | Typical Range | Test Range |
|-------------------|-----------|---------------|------------|
| **Baseline Correction (ALS)** | `lam` (smoothness) | 10⁵–10⁷ | [10⁴, 10⁵, 10⁶, 10⁷, 10⁸] |
| **Baseline Correction (ALS)** | `p` (asymmetry) | 0.001–0.1 | [0.0001, 0.001, 0.01, 0.1] |
| **Savitzky-Golay Smoothing** | `window_length` | 5–15 | [3, 5, 7, 9, 11, 13, 15] |
| **Savitzky-Golay Smoothing** | `polyorder` | 2–3 | [2, 3] |
| **SNV Normalization** | `ddof` | 0–1 | [0, 1] |

**Step 2: Grid Search Performance**

Test all combinations (or Latin hypercube sampling for large grids):

```python
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from foodspec.preprocessing import baseline_als, savgol_filter, snv
import numpy as np
import pandas as pd

# Define parameter grid
lam_values = [1e5, 1e6, 1e7]
window_values = [5, 7, 9, 11]

results = []

for lam in lam_values:
    for window in window_values:
        # Apply preprocessing
        X_baseline = baseline_als(X_raw, lam=lam, p=0.01)
        X_smoothed = savgol_filter(X_baseline, window_length=window, polyorder=2)
        X_norm = snv(X_smoothed)
        
        # Cross-validate
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(
            RandomForestClassifier(random_state=42),
            X_norm, y, cv=cv, groups=sample_ids
        )
        
        results.append({
            'lam': lam,
            'window': window,
            'accuracy_mean': scores.mean(),
            'accuracy_std': scores.std()
        })

# Convert to DataFrame
df_results = pd.DataFrame(results)
print(df_results.sort_values('accuracy_mean', ascending=False))
```

**Step 3: Visualize Sensitivity**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap: lam (rows) vs. window (cols)
pivot = df_results.pivot(index='lam', columns='window', values='accuracy_mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.75, vmax=0.95)
plt.title('Preprocessing Sensitivity: Accuracy vs. ALS λ and Smoothing Window')
plt.xlabel('Savitzky-Golay Window Length')
plt.ylabel('ALS Baseline λ')
plt.show()
```

**Step 4: Define Robustness Criterion**

**Robust model:** Performance varies by <5% across reasonable parameter ranges.

**Example:**
```yaml
Best:  lam=1e6, window=9  → Accuracy = 0.885
Worst: lam=1e5, window=13 → Accuracy = 0.842
Range: 4.3 percentage points → ✅ ACCEPTABLE (< 5%)

If range > 10%: ⚠️ Model is fragile—consider ensemble or different features.
```

---

### FoodSpec Shortcut

```python
from foodspec.ml.robustness import preprocessing_sensitivity_analysis

report = preprocessing_sensitivity_analysis(
    X_raw, y,
    groups=sample_ids,
    model=RandomForestClassifier(),
    preprocessing_steps=['baseline_als', 'savgol', 'snv'],
    param_grid={
        'baseline_als__lam': [1e5, 1e6, 1e7],
        'baseline_als__p': [0.001, 0.01, 0.1],
        'savgol__window_length': [5, 7, 9, 11],
        'snv__ddof': [0, 1]
    },
    cv=5,
    n_repeats=3
)

# Generates:
# - Sensitivity heatmaps
# - Robustness score (mean performance ± std across grid)
# - Recommended "safe" parameter ranges
```

---

## 2. Outlier Robustness

### Goal
Ensure model does not collapse when encountering spectra with artifacts (spikes, baseline shifts, saturation).

### Types of Spectral Outliers

| Artifact | Cause | Example |
|----------|-------|---------|
| **Cosmic Ray Spikes** | Detector noise (Raman) | Single-pixel intensity spike (10-100× normal) |
| **Baseline Offset** | Sample positioning | Entire spectrum shifted vertically by +0.5 absorbance |
| **Saturation** | Overexposure | Intensities clipped at max detector value (e.g., 65535 counts) |
| **Atmospheric Water** | Poor background correction (FTIR) | Broad peaks at 1650 cm⁻¹ and 3500 cm⁻¹ |
| **Fluorescence** | Sample autofluorescence (Raman) | Exponentially rising baseline |

---

### Protocol: Inject Artificial Outliers

**Step 1: Create Synthetic Outliers**

```python
import numpy as np

def inject_spike(spectrum, position=500, magnitude=10):
    """Inject a cosmic ray spike."""
    corrupted = spectrum.copy()
    corrupted[position] = corrupted[position] * magnitude
    return corrupted

def inject_baseline_shift(spectrum, shift=0.5):
    """Shift entire spectrum vertically."""
    return spectrum + shift

def inject_saturation(spectrum, threshold=3.0):
    """Clip intensities above threshold."""
    return np.clip(spectrum, None, threshold)

# Example: Corrupt 10% of test spectra
n_outliers = int(0.1 * len(X_test))
outlier_indices = np.random.choice(len(X_test), n_outliers, replace=False)

X_test_corrupted = X_test.copy()
for idx in outlier_indices:
    artifact_type = np.random.choice(['spike', 'shift', 'saturation'])
    if artifact_type == 'spike':
        X_test_corrupted[idx] = inject_spike(X_test_corrupted[idx])
    elif artifact_type == 'shift':
        X_test_corrupted[idx] = inject_baseline_shift(X_test_corrupted[idx])
    else:
        X_test_corrupted[idx] = inject_saturation(X_test_corrupted[idx])
```

**Step 2: Evaluate Performance on Corrupted Data**

```python
from sklearn.metrics import accuracy_score

# Clean test set
y_pred_clean = model.predict(X_test)
acc_clean = accuracy_score(y_test, y_pred_clean)

# Corrupted test set
y_pred_corrupted = model.predict(X_test_corrupted)
acc_corrupted = accuracy_score(y_test, y_pred_corrupted)

print(f"Clean Accuracy:     {acc_clean:.3f}")
print(f"Corrupted Accuracy: {acc_corrupted:.3f}")
print(f"Performance Drop:   {(acc_clean - acc_corrupted):.3f}")

# Robustness criterion: Drop < 5% acceptable
if (acc_clean - acc_corrupted) < 0.05:
    print("✅ Model is robust to outliers")
else:
    print("⚠️ Model is sensitive to outliers—consider robust preprocessing")
```

---

### Mitigation Strategies

If model fails outlier robustness test:

1. **Robust Preprocessing:**
   - Median filter instead of Gaussian smoothing
   - Iterative outlier clipping (sigma-clipping)
   - Robust baseline correction (asymmetric least squares with weights)

2. **Outlier Detection:**
   - Flag suspicious spectra before modeling (Hotelling's T², Mahalanobis distance)
   - Train separate outlier detector (Isolation Forest, One-Class SVM)

3. **Ensemble Methods:**
   - Random Forest (inherently robust via bagging)
   - Median aggregation of predictions from multiple models

**Example (Robust Preprocessing):**
```python
from scipy.ndimage import median_filter

# Replace Gaussian smoothing with median filter
X_robust = np.apply_along_axis(median_filter, 1, X_raw, size=5)

# Evaluate robustness again
y_pred_robust = model.fit(X_robust_train, y_train).predict(X_robust_test_corrupted)
acc_robust = accuracy_score(y_test, y_pred_robust)
print(f"Robust Accuracy (corrupted data): {acc_robust:.3f}")
```

---

## 3. Batch/Day Robustness

### Goal
Test performance stability when model encounters new batches, measurement days, or instruments.

### Protocol: Leave-One-Batch-Out Analysis

**Step 1: Identify Batch Structure**

Determine batch/day groupings in your dataset:

```python
import pandas as pd

# Metadata with batch information
metadata = pd.DataFrame({
    'sample_id': sample_ids,
    'batch': ['B1', 'B1', 'B1', 'B2', 'B2', ...],  # Measurement batch
    'day': [1, 1, 1, 2, 2, ...]  # Measurement day
})

print(metadata.groupby('batch').size())
# B1    30 samples
# B2    25 samples
# B3    28 samples
# B4    32 samples
# B5    25 samples
```

**Step 2: Leave-One-Batch-Out Cross-Validation**

```python
from sklearn.model_selection import LeaveOneGroupOut

cv = LeaveOneGroupOut()
scores_per_batch = []

for train_idx, test_idx in cv.split(X, y, groups=metadata['batch']):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    test_batch = metadata.iloc[test_idx]['batch'].unique()[0]
    
    scores_per_batch.append({'batch': test_batch, 'accuracy': score})
    print(f"Test Batch {test_batch}: Accuracy = {score:.3f}")

# Aggregate results
df_batch_scores = pd.DataFrame(scores_per_batch)
print(f"\nMean Batch-Out Accuracy: {df_batch_scores['accuracy'].mean():.3f}")
print(f"Std Dev: {df_batch_scores['accuracy'].std():.3f}")
print(f"Range: [{df_batch_scores['accuracy'].min():.3f}, {df_batch_scores['accuracy'].max():.3f}]")
```

**Example Output:**
```yaml
Test Batch B1: Accuracy = 0.867
Test Batch B2: Accuracy = 0.840
Test Batch B3: Accuracy = 0.880
Test Batch B4: Accuracy = 0.820
Test Batch B5: Accuracy = 0.853

Mean Batch-Out Accuracy: 0.852
Std Dev: 0.023
Range: [0.820, 0.880]

✅ Acceptable batch robustness (std dev = 2.3%, range = 6%)
```

**Robustness Criterion:**
- **Std Dev <5%:** Excellent batch robustness
- **Std Dev 5-10%:** Moderate batch effects (consider batch correction)
- **Std Dev >10%:** Severe batch effects (model not deployable without harmonization)

---

### Step 3: Batch Effect Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# PCA projection (first 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot colored by batch
plt.figure(figsize=(10, 6))
for batch in metadata['batch'].unique():
    mask = metadata['batch'] == batch
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Batch {batch}', alpha=0.7)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Batch Effect Visualization (PCA)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Interpretation:**
- **Batches overlap in PCA space:** Minimal batch effect → Good
- **Batches form distinct clusters:** Strong batch effect → Problem

---

### Step 4: Statistical Test for Batch Effects

```python
from scipy.stats import f_oneway

# ANOVA on PC1 scores (test if batches differ significantly)
pc1_by_batch = [X_pca[metadata['batch'] == b, 0] for b in metadata['batch'].unique()]
f_stat, p_value = f_oneway(*pc1_by_batch)

print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("⚠️ Significant batch effect detected (p < 0.05)")
    print("   → Consider batch correction (e.g., ComBat, PCA residuals)")
else:
    print("✅ No significant batch effect (p ≥ 0.05)")
```

---

### Mitigation: Batch Effect Correction

If batch effects are severe:

**1. ComBat Harmonization (Parametric)**
```python
from neuroCombat import neuroCombat

# Harmonize spectra to remove batch effects (preserves biological signal)
X_harmonized = neuroCombat(
    dat=X.T,  # Features × samples
    covars={'batch': metadata['batch'].values},
    categorical_cols=['batch']
)['data'].T

# Re-evaluate with harmonized data
scores_harmonized = cross_val_score(model, X_harmonized, y, cv=cv_batch)
print(f"Post-Harmonization Accuracy: {scores_harmonized.mean():.3f}")
```

**2. Domain Adaptation (Transfer Learning)**
```python
from foodspec.ml.harmonization import transfer_component_analysis

# Learn transformation to align batches
X_aligned = transfer_component_analysis(
    X_source=X[metadata['batch'] != target_batch],
    X_target=X[metadata['batch'] == target_batch],
    n_components=10
)
```

---

## 4. Adversarial Testing

### Goal
Simulate **out-of-distribution** samples that the model was not trained to handle.

### Scenarios

#### A) Adulteration Simulation

**Example:** Model trained on pure EVOO → Test on EVOO + 5% hazelnut oil

```python
# Simulate adulteration (linear mixture)
def simulate_adulteration(pure_spectrum, adulterant_spectrum, level=0.05):
    """Mix two spectra at given adulteration level."""
    return (1 - level) * pure_spectrum + level * adulterant_spectrum

# Load pure EVOO and hazelnut oil spectra
evoo_spectrum = X[y == 'EVOO'][0]
hazelnut_spectrum = X_hazelnut[0]  # From separate dataset

# Generate adulterated test set (1%, 5%, 10%, 20%)
X_adulterated = []
for level in [0.01, 0.05, 0.10, 0.20]:
    adulterated = simulate_adulteration(evoo_spectrum, hazelnut_spectrum, level)
    X_adulterated.append(adulterated)

X_adulterated = np.array(X_adulterated)

# Predict: Should ideally flag as anomalous (low confidence)
y_pred = model.predict(X_adulterated)
proba = model.predict_proba(X_adulterated)

for i, level in enumerate([0.01, 0.05, 0.10, 0.20]):
    print(f"Adulteration {level*100:.0f}%: Predicted '{y_pred[i]}', Confidence = {proba[i].max():.3f}")

# Expected: Confidence should decrease with higher adulteration
```

**Robustness Criterion:**
- Model should output **low confidence** (<0.7) for out-of-distribution samples
- Avoid false negatives (classifying adulterated as pure with high confidence)

---

#### B) Degradation Simulation

**Example:** Model trained on fresh oil → Test on thermally degraded oil

```python
# Simulate degradation (increase oxidation peak at ~1740 cm⁻¹)
def simulate_degradation(spectrum, wavenumbers, degradation_level=0.5):
    """Increase carbonyl peak (oxidation marker)."""
    degraded = spectrum.copy()
    carbonyl_region = (wavenumbers > 1700) & (wavenumbers < 1780)
    degraded[carbonyl_region] += degradation_level
    return degraded

# Test on degraded samples
X_degraded = np.array([simulate_degradation(X[0], wavenumbers, level=0.5)])
y_pred_degraded = model.predict(X_degraded)
proba_degraded = model.predict_proba(X_degraded)

print(f"Degraded Sample: Predicted '{y_pred_degraded[0]}', Confidence = {proba_degraded.max():.3f}")

# Expected: Low confidence or explicit "degraded" class (if trained with degraded samples)
```

---

#### C) Matrix Effect Simulation

**Example:** Model trained on pure oils → Test on oil extracted from fried chips

```python
# Simulate matrix interference (add broad background from food matrix)
def simulate_matrix_effect(oil_spectrum, matrix_spectrum, matrix_contribution=0.2):
    """Add food matrix contribution to pure oil spectrum."""
    return oil_spectrum + matrix_contribution * matrix_spectrum

# Load matrix spectrum (e.g., fried chips)
chips_matrix = X_chips[0]  # Separate dataset

X_matrix = simulate_matrix_effect(evoo_spectrum, chips_matrix, matrix_contribution=0.3)
y_pred_matrix = model.predict([X_matrix])
proba_matrix = model.predict_proba([X_matrix])

print(f"Matrix Effect: Predicted '{y_pred_matrix[0]}', Confidence = {proba_matrix.max():.3f}")

# Expected: Performance degradation unless model trained with diverse matrices
```

---

## Robustness Checklist

Use this checklist before deploying a model:

| Test | Pass Criterion | Status |
|------|---------------|--------|
| **Preprocessing Sensitivity** | Performance varies <5% across parameter ranges | ☐ |
| **Outlier Robustness** | Accuracy drop <5% with 10% corrupted spectra | ☐ |
| **Leave-One-Batch-Out** | Std dev <5% across batches | ☐ |
| **Batch Effect Visualization** | PCA shows overlapping batches (not distinct clusters) | ☐ |
| **Adulteration Simulation** | Low confidence (<0.7) on out-of-distribution samples | ☐ |
| **Degradation Simulation** | Detects or flags degraded samples appropriately | ☐ |
| **Matrix Effect Simulation** | Performance stable or explicitly warns | ☐ |

**Deployment Recommendation:**
- **All tests passed (✅):** Model ready for deployment
- **1-2 tests failed:** Address specific weaknesses (e.g., batch correction, robust preprocessing)
- **3+ tests failed:** Model not robust—reconsider feature engineering or data augmentation

---

## FoodSpec Robustness Report

Generate a comprehensive robustness report:

```python
from foodspec.ml.robustness import comprehensive_robustness_test

report = comprehensive_robustness_test(
    model=model,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    groups=sample_ids,
    batches=measurement_batches,
    preprocessing_grid={
        'baseline_als__lam': [1e5, 1e6, 1e7],
        'savgol__window_length': [5, 7, 9, 11]
    },
    outlier_types=['spike', 'baseline_shift', 'saturation'],
    outlier_fraction=0.1,
    adversarial_tests=['adulteration', 'degradation', 'matrix_effect'],
    save_path='robustness_report.html'
)

# Generates:
# - Preprocessing sensitivity heatmaps
# - Outlier robustness curves
# - Leave-one-batch-out performance table
# - Batch effect PCA plots
# - Adversarial test results
# - Overall robustness score (0-100)
```

---

## Further Reading

- **Marini et al. (2007).** "Artificial neural networks in food analysis: Trends and perspectives." *Anal. Chim. Acta*, 605:111-121. [DOI](https://doi.org/10.1016/j.aca.2007.10.028)
- **Goodfellow et al. (2014).** "Explaining and harnessing adversarial examples." *arXiv:1412.6572*. [Link](https://arxiv.org/abs/1412.6572)
- **Johnson et al. (2007).** "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics*, 8:118-127. [DOI](https://doi.org/10.1093/biostatistics/kxj037)

---

## Related Pages

- [Cross-Validation & Leakage](cross_validation_and_leakage.md) – Prevent data leakage
- [Metrics & Uncertainty](metrics_and_uncertainty.md) – Quantify confidence intervals
- [Reporting Standards](reporting_standards.md) – Document robustness tests
- [Workflows → Harmonization](../../workflows/harmonization/harmonization_automated_calibration.md) – Batch correction methods
- [Cookbook → Preprocessing](../preprocessing/normalization_smoothing.md) – Robust preprocessing recipes

---

**Next:** Learn to [document and report validation results](reporting_standards.md) for publication →
