# Metrics & Uncertainty Quantification

!!! abstract "Beyond Point Estimates"
    A single accuracy number (e.g., "92.3%") is **insufficient** for scientific validation. Modern reporting standards require:
    
    1. **Uncertainty quantification:** Confidence intervals via repeated CV or bootstrapping
    2. **Metric diversity:** Accuracy, precision, recall, F1, MCC (not just accuracy)
    3. **Class-specific performance:** Per-class metrics for multiclass problems
    4. **Statistical significance:** Hypothesis tests when comparing models
    
    This page provides comprehensive guidance on metric selection, uncertainty estimation, and statistical testing for spectroscopy models.

---

## Why Uncertainty Matters

### The Problem with Point Estimates

Consider this scenario:

**Model A:** 92.3% accuracy (single 5-fold CV run)  
**Model B:** 90.8% accuracy (single 5-fold CV run)

**Question:** Is Model A truly better, or is the difference due to random fold assignment?

**Answer:** Without uncertainty estimates, **we cannot know**. The 1.5 percentage point difference might be:
- Real improvement (statistically significant)
- Random variation (not significant)
- Artifact of lucky fold assignment

### Repeated CV Solution

Run CV multiple times with different random seeds:

```python
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

scores_A = cross_val_score(model_A, X, y, cv=cv, scoring='accuracy')
scores_B = cross_val_score(model_B, X, y, cv=cv, scoring='accuracy')

# Compute 95% confidence intervals
mean_A, ci_A = scores_A.mean(), 1.96 * scores_A.std()
mean_B, ci_B = scores_B.mean(), 1.96 * scores_B.std()

print(f"Model A: {mean_A:.3f} ± {ci_A:.3f} (95% CI)")
print(f"Model B: {mean_B:.3f} ± {ci_B:.3f} (95% CI)")

# Result: 
# Model A: 0.923 ± 0.028 (95% CI: [0.895, 0.951])
# Model B: 0.908 ± 0.032 (95% CI: [0.876, 0.940])
# 
# Conclusion: CIs overlap → difference not statistically significant
```

!!! success "Reporting Recommendation"
    Always report **mean ± 95% CI** from **≥10 repeated CV runs** (preferably 20).

---

## Metric Selection Guide

Different metrics emphasize different aspects of model performance. Choose metrics that align with your **domain goals**.

### Classification Metrics

#### 1. Accuracy

**Definition:**
$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

**When to Use:**
- Balanced datasets (roughly equal class sizes)
- All errors equally costly

**When NOT to Use:**
- ⚠️ **Imbalanced datasets** (e.g., 95% EVOO, 5% adulterated)
  - A model that always predicts "EVOO" achieves 95% accuracy without learning anything useful!

**Example (Imbalance Problem):**
```python
# Dataset: 950 EVOO samples, 50 adulterated samples (1000 total)
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')  # Always predict majority class
dummy.fit(X_train, y_train)
print(f"Dummy Accuracy: {dummy.score(X_test, y_test):.3f}")  # 0.950 (95%)

# A useless model achieves 95% accuracy by never detecting adulteration!
```

---

#### 2. Precision & Recall

**Definitions:**
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad \text{(How many predicted positives are correct?)}
$$

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad \text{(How many actual positives are detected?)}
$$

**When to Use:**
- **Precision-critical:** Adulteration detection (minimize false alarms → avoid discarding good product)
- **Recall-critical:** Safety screening (maximize detection → avoid missing unsafe products)

**Example (Adulteration Detection):**
```python
from sklearn.metrics import classification_report

# Goal: Detect adulterated olive oil (positive class = adulterated)
y_true = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1]  # 0=EVOO, 1=Adulterated
y_pred = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1]  # Model predictions

print(classification_report(y_true, y_pred, target_names=['EVOO', 'Adulterated']))

#               precision    recall  f1-score   support
#         EVOO       0.86      0.86      0.86         7
# Adulterated       0.75      0.75      0.75         3
#     accuracy                           0.80        10
```

**Interpretation:**
- **Precision (Adulterated) = 0.75:** 75% of flagged samples are truly adulterated (25% false positives)
- **Recall (Adulterated) = 0.75:** 75% of adulterated samples are detected (25% missed)

---

#### 3. F1-Score

**Definition:**
$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}
$$

**When to Use:**
- Imbalanced datasets
- Equal weight to precision and recall
- **Recommended for binary classification in spectroscopy**

**FoodSpec Example:**
```python
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
print(f"F1-Score: {scores.mean():.3f} ± {1.96 * scores.std():.3f}")
```

---

#### 4. Matthews Correlation Coefficient (MCC)

**Definition:**
$$
\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}
$$

**Range:** -1 (total disagreement) to +1 (perfect prediction), 0 = random

**When to Use:**
- **Imbalanced datasets** (MCC accounts for all four confusion matrix quadrants)
- **Gold standard for binary classification**
- Robust to class imbalance (unlike accuracy)

**Advantages:**
- Takes into account true/false positives AND negatives
- Not inflated by class imbalance
- Symmetric (swapping positive/negative class gives same MCC)

**Example:**
```python
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import cross_val_score

mcc_scorer = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X, y, cv=cv, scoring=mcc_scorer)
print(f"MCC: {scores.mean():.3f} ± {1.96 * scores.std():.3f}")
```

!!! tip "Recommendation"
    For **imbalanced binary classification**, report **both F1-score and MCC**. They provide complementary perspectives on performance.

---

#### 5. Confusion Matrix (Multiclass)

For multiclass problems (e.g., 5 olive oil origins), show the full confusion matrix:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix: Olive Oil Origin Classification')
plt.show()
```

**Key Insights:**
- **Diagonal:** Correctly classified samples
- **Off-diagonal:** Misclassifications (row = true class, column = predicted class)
- **Class-specific errors:** Which classes are confused (e.g., Spanish vs. Italian EVOO)?

---

### Regression Metrics

#### 1. Mean Absolute Error (MAE)

**Definition:**
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**When to Use:**
- Robust to outliers (linear penalty)
- Interpretable in original units (e.g., "±0.5% adulteration level")

**Example (Quantitative Adulteration):**
```python
from sklearn.metrics import mean_absolute_error

y_true = [0.0, 2.5, 5.0, 10.0]  # True adulteration % (hazelnut oil in EVOO)
y_pred = [0.5, 2.0, 5.5, 11.0]  # Model predictions

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}%")  # 0.62% (average error magnitude)
```

---

#### 2. Root Mean Squared Error (RMSE)

**Definition:**
$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

**When to Use:**
- Penalizes large errors more than MAE (quadratic penalty)
- Standard metric for calibration tasks

**Comparison with MAE:**
- **MAE = 0.62%, RMSE = 0.75%:** Errors are mostly uniform (RMSE ≈ MAE)
- **MAE = 0.62%, RMSE = 1.50%:** Large outlier errors present (RMSE >> MAE)

---

#### 3. R² (Coefficient of Determination)

**Definition:**
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

**Range:** -∞ to 1 (1 = perfect fit, 0 = no better than mean prediction)

**When to Use:**
- Compare models (higher R² = better fit)
- ⚠️ **Caution:** R² can be misleading with nonlinear relationships or outliers

**FoodSpec Example:**
```python
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
print(f"R²: {scores.mean():.3f} ± {1.96 * scores.std():.3f}")
```

---

## Uncertainty Quantification Methods

### 1. Repeated Cross-Validation

**Recommended for:** Classification and regression tasks with sufficient data (n >50)

**Implementation:**
```python
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
import numpy as np

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=42)

# Compute multiple metrics simultaneously
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
results = cross_validate(model, X, y, cv=cv, scoring=scoring)

# Report mean ± 95% CI for each metric
for metric in scoring:
    scores = results[f'test_{metric}']
    mean, ci = scores.mean(), 1.96 * scores.std()
    print(f"{metric}: {mean:.3f} ± {ci:.3f} (95% CI: [{mean-ci:.3f}, {mean+ci:.3f}])")
```

**Output Example:**
```yaml
accuracy:         0.873 ± 0.032 (95% CI: [0.841, 0.905])
f1_macro:         0.865 ± 0.035 (95% CI: [0.830, 0.900])
precision_macro:  0.881 ± 0.038 (95% CI: [0.843, 0.919])
recall_macro:     0.852 ± 0.040 (95% CI: [0.812, 0.892])
```

---

### 2. Bootstrapping

**Recommended for:** Small datasets (n <50) or when CV folds are too coarse

**Principle:** Resample training data with replacement, train model, evaluate on out-of-bag samples.

**Implementation:**
```python
from sklearn.utils import resample
import numpy as np

n_bootstraps = 1000
scores = []

for i in range(n_bootstraps):
    # Resample training data with replacement
    X_boot, y_boot = resample(X_train, y_train, random_state=i)
    
    # Train model
    model.fit(X_boot, y_boot)
    
    # Evaluate on test set (fixed, not resampled)
    score = model.score(X_test, y_test)
    scores.append(score)

# Compute 95% confidence interval (percentile method)
ci_lower = np.percentile(scores, 2.5)
ci_upper = np.percentile(scores, 97.5)
print(f"Accuracy: {np.mean(scores):.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
```

---

### 3. Prediction Intervals (Regression)

**Goal:** Quantify uncertainty in **individual predictions** (not just model performance).

**Methods:**

#### a) Quantile Regression (Non-Parametric)

```python
from sklearn.ensemble import GradientBoostingRegressor

# Train two models: lower (10th percentile) and upper (90th percentile)
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.10)
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.90)

model_lower.fit(X_train, y_train)
model_upper.fit(X_train, y_train)

# Predict with 80% prediction interval
y_pred_lower = model_lower.predict(X_test)
y_pred_upper = model_upper.predict(X_test)

print(f"Prediction: {y_pred_mean:.2f}% ± [{y_pred_lower:.2f}%, {y_pred_upper:.2f}%]")
```

#### b) Conformal Prediction (Distribution-Free)

**Advantages:**
- Valid under minimal assumptions (exchangeability only)
- Provides finite-sample guarantees (not asymptotic)

**FoodSpec Implementation:**
```python
from foodspec.ml.uncertainty import conformal_prediction_interval

# Calibrate on validation set
cal_residuals = np.abs(y_val - model.predict(X_val))
alpha = 0.10  # 90% coverage target

# Compute prediction intervals
y_pred, intervals = conformal_prediction_interval(
    model, X_test, cal_residuals, alpha=alpha
)

print(f"Prediction: {y_pred[0]:.2f}% (90% PI: [{intervals[0,0]:.2f}%, {intervals[0,1]:.2f}%])")
```

---

## Statistical Significance Testing

### When to Compare Models

Use hypothesis tests to determine if performance differences are statistically significant:

| Test | Use Case | Null Hypothesis |
|------|----------|----------------|
| **Paired t-test** | Compare two models (same CV folds) | $\mu_A = \mu_B$ |
| **McNemar's test** | Compare two classifiers (same test set) | Models make same errors |
| **Friedman test** | Compare >2 models (multiple datasets) | All models perform equally |
| **Wilcoxon signed-rank** | Non-parametric alternative to t-test | Median difference = 0 |

---

### 1. Paired t-test (Repeated CV)

**Scenario:** Compare two preprocessing pipelines (e.g., SNV vs. MSC normalization)

**Implementation:**
```python
from scipy.stats import ttest_rel

# Run repeated CV for both models (same folds)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

scores_SNV = cross_val_score(model_with_SNV, X, y, cv=cv)
scores_MSC = cross_val_score(model_with_MSC, X, y, cv=cv)

# Paired t-test (samples are paired by CV fold)
t_stat, p_value = ttest_rel(scores_SNV, scores_MSC)

print(f"SNV: {scores_SNV.mean():.3f} ± {scores_SNV.std():.3f}")
print(f"MSC: {scores_MSC.mean():.3f} ± {scores_MSC.std():.3f}")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Difference is statistically significant (p < 0.05)")
else:
    print("❌ No significant difference (p ≥ 0.05)")
```

**Interpretation:**
- **p <0.05:** Reject null hypothesis → Models are significantly different
- **p ≥0.05:** Fail to reject → No evidence of difference (could be equivalent or underpowered test)

---

### 2. McNemar's Test (Binary Classification)

**Scenario:** Compare two models on a fixed test set (e.g., PLS-DA vs. Random Forest)

**Implementation:**
```python
from statsmodels.stats.contingency_tables import mcnemar

# Predictions from two models on same test set
y_pred_A = model_A.predict(X_test)
y_pred_B = model_B.predict(X_test)

# Construct contingency table
# Rows: Model A correct/incorrect
# Cols: Model B correct/incorrect
table = [[np.sum((y_pred_A == y_test) & (y_pred_B == y_test)),    # Both correct
          np.sum((y_pred_A == y_test) & (y_pred_B != y_test))],   # A correct, B wrong
         [np.sum((y_pred_A != y_test) & (y_pred_B == y_test)),    # A wrong, B correct
          np.sum((y_pred_A != y_test) & (y_pred_B != y_test))]]   # Both wrong

result = mcnemar(table, exact=True)
print(f"McNemar statistic: {result.statistic}, p-value: {result.pvalue:.4f}")

if result.pvalue < 0.05:
    print("✅ Models make significantly different errors (p < 0.05)")
else:
    print("❌ Models perform equivalently (p ≥ 0.05)")
```

**Use Case:**
- Fixed test set (cannot repeat CV)
- Binary classification
- Tests if models make **different types of errors** (not just overall accuracy)

---

### 3. Friedman Test (Multiple Models, Multiple Datasets)

**Scenario:** Compare 3+ models across multiple benchmark datasets

**Implementation:**
```python
from scipy.stats import friedmanchisquare

# Scores from 3 models on 5 datasets (each row = one dataset)
scores_model_A = [0.85, 0.90, 0.88, 0.92, 0.87]
scores_model_B = [0.83, 0.89, 0.86, 0.91, 0.85]
scores_model_C = [0.80, 0.85, 0.84, 0.88, 0.82]

stat, p_value = friedmanchisquare(scores_model_A, scores_model_B, scores_model_C)

print(f"Friedman statistic: {stat:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ At least one model differs significantly (p < 0.05)")
    print("   → Perform post-hoc pairwise tests (e.g., Nemenyi)")
else:
    print("❌ No evidence of model differences (p ≥ 0.05)")
```

---

## Reporting Best Practices

### Minimum Reporting Requirements

For **publication-quality validation**, include:

1. **Primary Metric with Uncertainty:**
   - Mean ± 95% CI from repeated CV (≥10 repeats)
   - Example: "Accuracy: 87.3% ± 3.2% (95% CI: [84.1%, 90.5%])"

2. **Multiple Metrics:**
   - Classification: Accuracy, F1-score (or MCC), per-class precision/recall
   - Regression: MAE, RMSE, R²

3. **Confusion Matrix (Classification):**
   - Show which classes are confused
   - Identify systematic errors

4. **CV Strategy:**
   - Specify: Grouped K-fold (by sample), Leave-one-batch-out, Time-series, etc.
   - Justify choice based on deployment scenario

5. **Statistical Significance:**
   - When comparing models, report p-values from appropriate test
   - Example: "SNV normalization significantly outperformed MSC (p = 0.023, paired t-test)"

---

### Example Results Table (Classification)

| Metric | Mean | 95% CI | Range |
|--------|------|--------|-------|
| **Accuracy** | 0.873 | ±0.032 | [0.841, 0.905] |
| **F1-Score** | 0.865 | ±0.035 | [0.830, 0.900] |
| **MCC** | 0.821 | ±0.041 | [0.780, 0.862] |
| **Precision (EVOO)** | 0.910 | ±0.028 | [0.882, 0.938] |
| **Recall (EVOO)** | 0.885 | ±0.035 | [0.850, 0.920] |
| **Precision (Adulterated)** | 0.701 | ±0.052 | [0.649, 0.753] |
| **Recall (Adulterated)** | 0.745 | ±0.048 | [0.697, 0.793] |

**Validation Strategy:** 10-fold Grouped CV (by sample), repeated 20 times (200 total folds)

---

### Example Results Table (Regression)

| Metric | Mean | 95% CI | Unit |
|--------|------|--------|------|
| **MAE** | 0.62 | ±0.15 | % adulteration |
| **RMSE** | 0.85 | ±0.21 | % adulteration |
| **R²** | 0.942 | ±0.018 | — |
| **90% Prediction Interval Width** | 1.8 | ±0.3 | % adulteration |

**Validation Strategy:** 5-fold CV, repeated 10 times (50 total folds)

---

## FoodSpec Utilities

### All-in-One Validation Report

```python
from foodspec.ml.validation import comprehensive_validation_report

report = comprehensive_validation_report(
    model=RandomForestClassifier(),
    X=X, y=y,
    groups=sample_ids,
    n_splits=10,
    n_repeats=20,
    metrics=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
    plot_confusion_matrix=True,
    plot_roc_curves=True,
    save_path='validation_report.html'
)

# Generates:
# - Metrics table with mean ± 95% CI
# - Confusion matrix heatmap
# - ROC curves (binary or one-vs-rest)
# - Per-class performance breakdown
# - Statistical significance tests (if comparing models)
```

---

## Further Reading

- **Oliveri (2017).** "Class-modelling in food analytical chemistry: Development, sampling, optimisation and validation issues." *Anal. Chim. Acta*, 982:9-19. [DOI](https://doi.org/10.1016/j.aca.2017.05.013)
- **Chicco & Jurman (2020).** "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." *BMC Genomics*, 21:6. [DOI](https://doi.org/10.1186/s12864-019-6413-7)
- **Efron & Tibshirani (1994).** *An Introduction to the Bootstrap.* CRC Press. (Chapter 14: Confidence Intervals)
- **Shafer & Vovk (2008).** "A tutorial on conformal prediction." *J. Machine Learning Research*, 9:371-421. [Link](https://www.jmlr.org/papers/v9/shafer08a.html)

---

## Related Pages

- [Cross-Validation & Leakage](cross_validation_and_leakage.md) – CV strategies to prevent leakage
- [Robustness Checks](robustness_checks.md) – Test preprocessing sensitivity
- [Reporting Standards](reporting_standards.md) – Minimum reporting checklist
- [Reference → Metric Significance Tables](../../reference/metric_significance_tables.md) – Interpret effect sizes
- [API → Metrics](../../api/metrics.md) – Code documentation

---

**Next:** Learn to [test model robustness](robustness_checks.md) under realistic perturbations →
