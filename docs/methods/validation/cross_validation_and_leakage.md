# Cross-Validation and Data Leakage

**Purpose:** Understand leakage mechanisms and implement best-practice validation workflows to obtain honest performance estimates.

**Audience:** Researchers, data scientists, QA engineers validating models for deployment.

**Time:** 30–40 minutes to read; reference during analysis.

**Prerequisites:** Familiarity with train/test splits, cross-validation, Python/scikit-learn.

---

## Statement of Need

Many food spectroscopy studies report >95% accuracy but fail on new data. The root cause is silent **data leakage**: information from test samples influencing model training. This page shows how to detect and prevent it.

---

## What is Data Leakage?

**Data leakage** occurs when the model has access to information during training that it will not have during deployment. This creates an **artificially inflated performance estimate** that does not reflect real-world generalization.

### Types of Leakage in Spectroscopy

#### 1. Replicate Leakage (Most Common)

**Problem:** Technical replicates of the same biological sample are split across training and test sets.

**Example (FTIR Olive Oil Authentication):**
```yaml
Dataset: 30 olive oil samples, 3 replicates each (90 spectra total)
Labels: EVOO (15 samples), Lampante (15 samples)

❌ WRONG (Random Split):
Train: [OO_001_rep1, OO_001_rep2, OO_002_rep1, ..., OO_030_rep1]
Test:  [OO_001_rep3, OO_002_rep2, ..., OO_030_rep2]

Problem: The model sees OO_001 replicates 1&2 during training, 
         then "predicts" replicate 3 during testing.
         This is NOT prediction—it's interpolation between 
         technical replicates (which share 95%+ variance).

✅ CORRECT (Grouped by Sample):
Train: [OO_001_rep1/2/3, OO_002_rep1/2/3, ..., OO_020_rep1/2/3]
Test:  [OO_021_rep1/2/3, ..., OO_030_rep1/2/3]

Now the model must generalize to entirely new biological samples.
```

**Variance Decomposition:**
$$
\sigma_{\text{total}}^2 = \underbrace{\sigma_{\text{biological}}^2}_{\text{sample-to-sample}} + \underbrace{\sigma_{\text{technical}}^2}_{\text{replicate-to-replicate}}
$$

In spectroscopy, $\sigma_{\text{biological}}^2 \gg \sigma_{\text{technical}}^2$ (often 10-100×). Random CV only tests technical variance—**not biological generalization**.

---

#### 2. Preprocessing Leakage

**Problem:** Preprocessing methods (normalization, baseline correction) are fit on the entire dataset before splitting.

**Example (SNV Normalization):**
```python
❌ WRONG: Fit SNV on all data before splitting
X_normalized = StandardNormalVariate().fit_transform(X_all)
X_train, X_test = train_test_split(X_normalized, ...)

# Problem: Test spectra are normalized using statistics (mean, std) 
# computed from the entire dataset, including test samples themselves.
# This leaks information about the test distribution into training.

✅ CORRECT: Fit SNV only on training data
X_train, X_test = train_test_split(X_raw, ...)
snv = StandardNormalVariate()
X_train_norm = snv.fit_transform(X_train)  # Fit on train only
X_test_norm = snv.transform(X_test)        # Apply to test (no refitting)
```

**Impact:** Preprocessing leakage typically inflates accuracy by 2-8 percentage points.

**Common Preprocessing Leakage Sources:**
- **SNV/MSC normalization:** Mean/std computed from all samples
- **Baseline correction (ALS):** Reference spectrum computed from all samples
- **Feature selection:** Variables chosen based on correlation with labels (entire dataset)
- **PCA for visualization:** Principal components computed from all samples

!!! tip "Rule of Thumb"
    **Any operation that uses global statistics must be fit within CV folds.**

---

#### 3. Temporal Leakage

**Problem:** Training on data collected *after* the test period (causality violation).

**Example (Heating Quality Monitoring):**
```yaml
Dataset: Olive oil heated at 180°C, sampled every 2 hours for 20 hours
Goal: Predict degradation state at hour t

❌ WRONG (Random CV):
Train: [t=0h, t=2h, t=4h, t=8h, t=10h, t=14h, t=16h, t=18h, t=20h]
Test:  [t=6h, t=12h]

Problem: The model sees t=18h and t=20h during training, 
         then predicts t=12h during testing. This is impossible 
         in deployment (can't see the future).

✅ CORRECT (Time-Series CV):
Train: [t=0h, t=2h, t=4h, t=6h]
Test:  [t=8h]

Train: [t=0h, t=2h, t=4h, t=6h, t=8h, t=10h]
Test:  [t=12h]

Now the model only uses past data to predict future states.
```

---

#### 4. Batch Leakage

**Problem:** Samples from the same batch are split across train/test, making batch effects learnable.

**Example (Instrument Calibration):**
```yaml
Dataset: 100 samples measured on 5 days (20 samples/day)

❌ WRONG (Random CV):
Train: [Day1_S01-S15, Day2_S01-S15, ..., Day5_S01-S15]
Test:  [Day1_S16-S20, Day2_S16-S20, ..., Day5_S16-S20]

Problem: The model learns day-specific baseline shifts 
         (temperature, humidity) and "predicts" test samples 
         from the same days. This doesn't test day-to-day robustness.

✅ CORRECT (Leave-One-Day-Out CV):
Fold 1: Train on Days 1-4, Test on Day 5
Fold 2: Train on Days 1-3+5, Test on Day 4
...
Fold 5: Train on Days 2-5, Test on Day 1

Now the model must generalize to entirely new measurement days.
```

---

## Detecting Leakage

### Red Flags

| Symptom | Likely Cause |
|---------|--------------|
| **Accuracy >95%** on challenging tasks (e.g., 10-class EVOO origin) | Replicate or preprocessing leakage |
| **Train accuracy ≈ Test accuracy** (both very high) | Leakage (model not overfitting—just memorizing) |
| **Performance collapses on new batch/day** | Batch leakage in validation |
| **Test performance exceeds train** (rare but possible) | Temporal leakage (training on future data) |
| **Dramatic performance drop when grouping by sample** | Confirms replicate leakage |

### Leakage Detection Protocol

Run this 3-step diagnostic:

```yaml
from sklearn.model_selection import cross_val_score, GroupKFold, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Step 1: Random CV (leaky baseline)
random_scores = cross_val_score(
    RandomForestClassifier(random_state=42),
    X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42)
)

# Step 2: Grouped CV (no replicate leakage)
grouped_scores = cross_val_score(
    RandomForestClassifier(random_state=42),
    X, y, cv=GroupKFold(n_splits=5), groups=sample_ids
)

# Step 3: Compare
print(f"Random CV:  {random_scores.mean():.3f} ± {random_scores.std():.3f}")
print(f"Grouped CV: {grouped_scores.mean():.3f} ± {grouped_scores.std():.3f}")
print(f"Performance Drop: {(random_scores.mean() - grouped_scores.mean()):.3f}")

# Red flag if drop > 0.10 (10 percentage points)
if (random_scores.mean() - grouped_scores.mean()) > 0.10:
    print("⚠️  WARNING: Likely replicate leakage detected!")
```yaml

**Interpretation:**
- **Drop <5%:** Minimal leakage (technical variance low)
- **Drop 5-10%:** Moderate leakage (check grouping strategy)
- **Drop >10%:** Severe leakage (must use grouped CV)

---

## Recommended CV Strategies

Choose a strategy that matches your **deployment scenario**.

### 1. Random K-Fold CV

**When to Use:**
- Samples are truly independent (no replicates, batches, time-series)
- Exploratory analysis only (not for final validation)

**Implementation:**
```
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```yaml

**Limitations:**
- ⚠️ Does not account for replicates, batches, or time structure
- ⚠️ Overestimates performance in structured datasets

---

### 2. Stratified K-Fold CV

**When to Use:**
- Class imbalance (e.g., 80% EVOO, 20% adulterated)
- Samples independent, but classes must be balanced per fold

**Implementation:**
```
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```yaml

**Advantages:**
- Ensures each fold has representative class proportions
- Reduces variance in CV estimates

**Limitations:**
- ⚠️ Still does not account for replicates or batches

---

### 3. Grouped K-Fold CV (Recommended for Spectroscopy)

**When to Use:**
- Technical replicates of the same sample
- Batch/day structure in data collection
- **DEFAULT CHOICE for food spectroscopy studies**

**Implementation:**
```
from sklearn.model_selection import GroupKFold
import pandas as pd

# Example: 30 samples, 3 replicates each
data = pd.DataFrame({
    'sample_id': ['OO_001', 'OO_001', 'OO_001', 'OO_002', ...],  # 90 rows
    'replicate': [1, 2, 3, 1, 2, 3, ...],
    'label': ['EVOO', 'EVOO', 'EVOO', 'Lampante', ...]
})

# Group by sample_id (all replicates stay together)
cv = GroupKFold(n_splits=5)
scores = cross_val_score(
    model, X, y, 
    cv=cv, 
    groups=data['sample_id']  # Critical: groups parameter
)
```python

**Key Principle:**
All replicates of a sample must be in the **same fold** (all train or all test).

**FoodSpec Shortcut:**
```python
from foodspec.ml.validation import grouped_cross_validation

results = grouped_cross_validation(
    X, y, 
    groups=sample_ids,
    model=RandomForestClassifier(),
    n_splits=5,
    n_repeats=10  # Repeat for confidence intervals
)

print(f"Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_ci']:.3f}")
```

---

### 4. Leave-One-Group-Out CV (Maximum Robustness)

**When to Use:**
- Testing batch-to-batch or day-to-day generalization
- Small number of batches/days (<10)
- Regulatory validation (stringent requirement)

**Implementation:**
```python
from sklearn.model_selection import LeaveOneGroupOut

# Example: 5 days of measurement
cv = LeaveOneGroupOut()
scores = cross_val_score(
    model, X, y, 
    cv=cv, 
    groups=measurement_days  # Each fold leaves one day out
)
```

**Interpretation:**
- Tests "worst-case" generalization to entirely new batches/days
- Typically 5-15% lower accuracy than grouped K-fold
- **Gold standard for instrument calibration transfer**

**Example Output:**
```yaml
Leave-One-Day-Out Results:
  Fold 1 (Test Day 1): Accuracy = 0.88
  Fold 2 (Test Day 2): Accuracy = 0.85
  Fold 3 (Test Day 3): Accuracy = 0.91
  Fold 4 (Test Day 4): Accuracy = 0.83
  Fold 5 (Test Day 5): Accuracy = 0.89
  
Mean Accuracy: 0.872 ± 0.031
```

---

### 5. Time-Series CV (Temporal Validation)

**When to Use:**
- Time-series data (degradation monitoring, shelf-life prediction)
- Training must only use past data to predict future

**Implementation:**
```python
from sklearn.model_selection import TimeSeriesSplit

# Example: 20 time points (hourly sampling)
cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=cv)

# Fold structure:
# Fold 1: Train [t=0,1,2,3],   Test [t=4]
# Fold 2: Train [t=0,1,2,3,4], Test [t=5]
# ...
# Fold 5: Train [t=0-15],      Test [t=16-19]
```

**Advantages:**
- Respects temporal causality (no future leakage)
- Realistic for deployment (only historical data available)

**FoodSpec Example:**
```python
from foodspec.ml.validation import time_series_validation

results = time_series_validation(
    X, y, 
    time_points=timestamps,
    test_size=0.2,  # Use last 20% as test
    gap=1  # Skip 1 time point between train/test (simulate deployment lag)
)
```

---

## Nested CV for Hyperparameter Tuning

**Problem:** Tuning hyperparameters on the same data used for final evaluation leaks information.

**Solution:** Use **nested cross-validation**:
- **Outer loop:** Estimates generalization performance
- **Inner loop:** Tunes hyperparameters (on training folds only)

**Implementation:**
```python
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Outer CV: Final performance estimate (5-fold)
outer_cv = GroupKFold(n_splits=5)
outer_scores = []

for train_idx, test_idx in outer_cv.split(X, y, groups=sample_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = sample_ids[train_idx]
    
    # Inner CV: Hyperparameter tuning (3-fold, on training data only)
    inner_cv = GroupKFold(n_splits=3)
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=inner_cv,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train, groups=groups_train)
    
    # Evaluate best model on test fold (outer loop)
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test, y_test)
    outer_scores.append(score)

print(f"Nested CV Accuracy: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")
```

**FoodSpec Shortcut:**
```python
from foodspec.ml.validation import nested_cross_validation

results = nested_cross_validation(
    X, y,
    groups=sample_ids,
    model=RandomForestClassifier(),
    param_grid={'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
    outer_cv=5,
    inner_cv=3
)
```

---

## Repeated CV for Confidence Intervals

**Why Repeat?**
- Single CV run gives one performance estimate (subject to fold randomness)
- Repeated CV quantifies uncertainty: **mean ± 95% confidence interval**

**Implementation:**
```python
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)

# 50 total scores (5 folds × 10 repeats)
print(f"Accuracy: {scores.mean():.3f} ± {1.96 * scores.std():.3f}")  # 95% CI
```

**FoodSpec Recommendation:**
- **Preliminary analysis:** 5-fold CV, 5 repeats (25 scores)
- **Final validation:** 10-fold CV, 10 repeats (100 scores)
- **Publication:** 10-fold CV, 20 repeats (200 scores)

---

## Case Study: Detecting & Fixing Leakage

### Scenario: FTIR Olive Oil Authentication

**Dataset:**
- 40 olive oil samples (20 EVOO, 20 Lampante)
- 5 replicates per sample (200 spectra total)
- Collected over 8 days (5 samples/day)

**Initial Results (Random CV):**
```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(RandomForestClassifier(), X, y, cv=cv)
print(f"Accuracy: {scores.mean():.3f}")  # 0.985 (98.5%)
```

**Red Flag:** 98.5% accuracy on a 2-class problem—suspiciously high.

**Diagnostic (Grouped CV):**
```python
cv_grouped = GroupKFold(n_splits=5)
scores_grouped = cross_val_score(
    RandomForestClassifier(), X, y, cv=cv_grouped, groups=sample_ids
)
print(f"Grouped Accuracy: {scores_grouped.mean():.3f}")  # 0.825 (82.5%)
```

**Finding:** 16 percentage point drop → **replicate leakage confirmed**.

**Fix 1: Group by Sample**
```python
from foodspec.ml.validation import grouped_cross_validation

results = grouped_cross_validation(
    X, y,
    groups=sample_ids,
    model=RandomForestClassifier(),
    n_splits=5,
    n_repeats=10
)
print(f"Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_ci']:.3f}")
# 0.823 ± 0.041
```

**Fix 2: Leave-One-Day-Out (Batch Robustness)**
```python
from sklearn.model_selection import LeaveOneGroupOut

cv_day = LeaveOneGroupOut()
scores_day = cross_val_score(
    RandomForestClassifier(), X, y, cv=cv_day, groups=measurement_days
)
print(f"Day-Out Accuracy: {scores_day.mean():.3f}")  # 0.775 (77.5%)
```

**Interpretation:**
- **Random CV (98.5%):** Unrealistic (replicate leakage)
- **Grouped by Sample (82.3%):** Realistic biological generalization
- **Leave-One-Day-Out (77.5%):** Stringent (worst-case batch effect)

**Conclusion:** Report **82.3% ± 4.1%** as the primary result, with day-to-day robustness at **77.5%**.

---

## Best Practices Summary

### ✅ Do This

1. **Always group by sample** if replicates exist
2. **Split data before preprocessing** (fit preprocessing within CV folds)
3. **Use time-series CV** for temporal data
4. **Test leave-one-batch-out** to quantify batch effects
5. **Repeat CV 10-20 times** to compute confidence intervals
6. **Nest CV for hyperparameter tuning** (separate inner loop)
7. **Report both grouped and leave-one-batch-out** results

### ❌ Never Do This

1. ❌ **Random CV with replicates** (guaranteed leakage)
2. ❌ **Fit preprocessing on entire dataset** before splitting
3. ❌ **Select features using entire dataset** (leakage via variable selection)
4. ❌ **Train on future data** to predict past (temporal leakage)
5. ❌ **Report only best CV score** (cherry-picking folds)
6. ❌ **Tune hyperparameters on test set** (overfitting to test)
7. ❌ **Ignore batch/day structure** (unrealistic generalization)

---

## Further Reading

- **Brereton & Lloyd (2010).** "Support vector machines for classification and regression." *J. Chemometrics*, 24:1-11. [DOI](https://doi.org/10.1002/cem.1320)
- **Varmuza & Filzmoser (2016).** *Introduction to Multivariate Statistical Analysis in Chemometrics.* CRC Press. (Chapter 4: Validation)
- **Kapoor & Narayanan (2023).** "Leakage and the reproducibility crisis in ML-based science." *Patterns*, 4(9). [DOI](https://doi.org/10.1016/j.patter.2023.100804)
- **Raschka (2018).** "Model evaluation, model selection, and algorithm selection in machine learning." *arXiv:1811.12808*. [Link](https://arxiv.org/abs/1811.12808)

---

## Related Pages

- [Metrics & Uncertainty](metrics_and_uncertainty.md) – Quantify confidence intervals
- [Robustness Checks](robustness_checks.md) – Test preprocessing sensitivity
- [Reporting Standards](reporting_standards.md) – Minimum reporting checklist
- [Reference → Glossary](../../reference/glossary.md) – Terminology (Leakage, CV Strategy)
- [Cookbook → Validation Recipes](../validation/cross_validation_and_leakage.md) – Code examples

---

**Next:** Learn to [quantify uncertainty and choose appropriate metrics](metrics_and_uncertainty.md) →
