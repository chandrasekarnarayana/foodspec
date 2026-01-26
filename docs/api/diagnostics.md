# ROC/AUC Diagnostics

## Overview

The ROC (Receiver Operating Characteristic) diagnostics module provides comprehensive tools for evaluating classification model performance through ROC curves, AUC (Area Under Curve) metrics, and threshold optimization.

**Key capabilities:**
- **Binary & Multiclass Support**: Handles both binary classification and multiclass problems (OvR decomposition)
- **Bootstrap Confidence Intervals**: Distribution-free AUC CI estimation for statistical rigor
- **Threshold Optimization**: Youden's J statistic, cost-sensitive, sensitivity constraints
- **Reproducible Results**: Fixed random seeds for deterministic outputs
- **Sklearn Integration**: Compatible with scikit-learn classifiers and predict_proba arrays

## Core Concepts

### ROC Curves
A ROC curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1 - specificity) across all classification thresholds.

### AUC (Area Under Curve)
AUC is the probability that the model ranks a random positive example higher than a random negative example. Values range from 0 (worst) to 1 (best), with 0.5 indicating random guessing.

### Multiclass Strategy
For multiclass problems (K > 2 classes), FoodSpec uses **One-vs-Rest (OvR)** ROC decomposition:
- **Per-class**: Binary ROC computed treating each class vs. all others
- **Micro-average**: Aggregates TP/FP across all classes (emphasizes larger classes)
- **Macro-average**: Simple average of per-class AUCs (treats each class equally)

### Bootstrap Confidence Intervals
Bootstrap CI provides distribution-free confidence intervals for AUC:
1. Resample the data N times (default: 1000) with replacement
2. Compute AUC for each bootstrap sample
3. Use percentiles (e.g., 2.5th and 97.5th for 95% CI) as bounds

This approach requires no distributional assumptions and works well for small samples.

## API Reference

### Main Function: `compute_roc_diagnostics()`

```python
from foodspec.modeling.diagnostics import compute_roc_diagnostics

result = compute_roc_diagnostics(
    y_true,              # True labels (n,)
    y_proba,             # Predicted probabilities (n,) or (n, K)
    task="auto",         # "binary", "multiclass", or "auto"
    n_bootstrap=1000,    # Number of bootstrap replicates
    random_seed=42,      # For reproducibility
)
```

**Parameters:**
- `y_true` (array-like): Ground truth labels
- `y_proba` (array-like): Predicted class probabilities
  - For binary: shape (n,) or (n, 2)
  - For multiclass: shape (n, K)
- `task` (str): Classification task type (default: auto-detect)
- `n_bootstrap` (int): Bootstrap iterations for CI (default: 1000)
- `confidence_level` (float): CI level (default: 0.95)
- `random_seed` (int, optional): Reproducibility seed
- `sample_weight` (array-like, optional): Sample weights

**Returns:**
- `RocDiagnosticsResult` dataclass with:
  - `per_class`: Dict of per-class ROC metrics
  - `micro`: Micro-averaged ROC (multiclass only)
  - `macro_auc`: Macro-averaged AUC (multiclass only)
  - `optimal_thresholds`: Dict of optimal thresholds by policy
  - `metadata`: Computation details

## Usage Examples

### Binary Classification

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from foodspec.modeling.diagnostics import compute_roc_diagnostics

# Generate and train
X, y = make_classification(n_samples=200, n_features=20, random_state=42)
clf = LogisticRegression().fit(X[:150], y[:150])
y_proba = clf.predict_proba(X[150:])[:, 1]
y_test = y[150:]

# Compute ROC diagnostics
result = compute_roc_diagnostics(y_test, y_proba, random_seed=42)

# Access metrics
metrics = result.per_class[1]
print(f"AUC: {metrics.auc:.3f}")
print(f"95% CI: [{metrics.ci_lower:.3f}, {metrics.ci_upper:.3f}]")

# Get optimal threshold
youden_thr = result.optimal_thresholds["youden"]
print(f"Youden threshold: {youden_thr.threshold:.3f}")
print(f"  Sensitivity: {youden_thr.sensitivity:.3f}")
print(f"  Specificity: {youden_thr.specificity:.3f}")
```

### Multiclass Classification

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from foodspec.modeling.diagnostics import compute_roc_diagnostics

# 3-class problem
X, y = make_classification(n_samples=300, n_classes=3, n_clusters_per_class=2, random_state=42)
clf = LogisticRegression(multi_class="multinomial").fit(X[:200], y[:200])
y_proba = clf.predict_proba(X[200:])
y_test = y[200:]

# Compute diagnostics
result = compute_roc_diagnostics(y_test, y_proba, random_seed=42)

# Per-class metrics
for class_label, metrics in result.per_class.items():
    print(f"Class {class_label}: AUC = {metrics.auc:.3f}")

# Aggregate metrics
print(f"Macro AUC (equal weight): {result.macro_auc:.3f}")
print(f"Micro AUC (sample weight): {result.micro.auc:.3f}")
```

### Working with ROC Curves

```python
import matplotlib.pyplot as plt

result = compute_roc_diagnostics(y_test, y_proba, random_seed=42)

# For binary classification
metrics = result.per_class[1]
plt.plot(metrics.fpr, metrics.tpr, label=f"ROC (AUC={metrics.auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Shade confidence region
plt.fill_between([0, 1], 0, 1, alpha=0.1, color="gray")
plt.show()
```

## Result Structure

### `RocDiagnosticsResult`

```python
@dataclass
class RocDiagnosticsResult:
    per_class: Dict[Any, PerClassRocMetrics]       # {label: metrics}
    micro: Optional[PerClassRocMetrics] = None     # Multiclass only
    macro_auc: Optional[float] = None              # Multiclass only
    optimal_thresholds: Dict[str, ThresholdResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `PerClassRocMetrics`

```python
@dataclass
class PerClassRocMetrics:
    fpr: np.ndarray               # False positive rates
    tpr: np.ndarray               # True positive rates
    thresholds: np.ndarray        # Classification thresholds
    auc: float                    # Area under curve
    ci_lower: Optional[float]     # 95% CI lower bound
    ci_upper: Optional[float]     # 95% CI upper bound
    n_positives: Optional[int]    # Number of positive samples
    n_negatives: Optional[int]    # Number of negative samples
```

### `ThresholdResult`

```python
@dataclass
class ThresholdResult:
    threshold: float              # Decision threshold
    sensitivity: float            # True positive rate at threshold
    specificity: float            # True negative rate at threshold
    j_statistic: Optional[float]  # Youden's J = sensitivity + specificity - 1
    ppv: Optional[float]          # Positive predictive value (precision)
    npv: Optional[float]          # Negative predictive value
```

## Threshold Optimization Policies

### Youden's J Statistic
Maximizes sensitivity + specificity - 1. Provides a balanced threshold when both false positives and false negatives are equally costly.

```python
youden_thr = result.optimal_thresholds["youden"]
# Use threshold for classification
predictions = (y_proba >= youden_thr.threshold).astype(int)
```

**When to use:**
- General-purpose classification
- When no explicit cost information available
- Balanced datasets/costs

### Cost-Sensitive (Future)
Minimizes FP*cost_fp + FN*cost_fn. Useful when different error types have different costs.

Example: Medical screening where false negatives (missing disease) are more costly than false positives (unnecessary follow-up).

### Sensitivity Constraint (Future)
Maximizes specificity subject to achieving at least min_sensitivity. Useful when a minimum sensitivity is required.

Example: Disease detection where sensitivity >= 95% is a regulatory requirement.

## Best Practices

### 1. **Use Bootstrap CIs for Publication**
```python
result = compute_roc_diagnostics(
    y_true, y_proba, 
    n_bootstrap=5000,  # More samples for published results
    confidence_level=0.95,
    random_seed=42
)

metrics = result.per_class[1]
print(f"AUC: {metrics.auc:.3f} (95% CI: [{metrics.ci_lower:.3f}, {metrics.ci_upper:.3f}])")
```

### 2. **Ensure Reproducibility**
Always set `random_seed` when results must be reproducible:
```python
result = compute_roc_diagnostics(y_true, y_proba, random_seed=42)
```

### 3. **Handle Multiclass with Caution**
Be aware of class imbalance effects on micro vs. macro AUC:
```python
# Micro emphasizes larger classes
micro_auc = result.micro.auc

# Macro treats all classes equally
macro_auc = result.macro_auc

# If imbalanced dataset, consider reporting both
```

### 4. **Validate with External Data**
ROC metrics should be computed on held-out test data, not training data:
```python
# Train on train set
clf.fit(X_train, y_train)

# Evaluate on test set (never seen during training)
y_proba_test = clf.predict_proba(X_test)
result = compute_roc_diagnostics(y_test, y_proba_test, random_seed=42)
```

### 5. **Consider Sample Weights for Stratified Data**
If data is collected with different strata, use sample weights:
```python
result = compute_roc_diagnostics(
    y_true, y_proba,
    sample_weight=stratum_weights,
    random_seed=42
)
```

## Advanced Topics

### Interpreting Confidence Intervals

The 95% bootstrap CI provides a range where the true AUC likely falls 95% of the time:
- **Narrow CI** (e.g., [0.85, 0.90]): High confidence in AUC estimate
- **Wide CI** (e.g., [0.70, 0.95]): More uncertainty; consider larger sample size

### Multiclass AUC Interpretation

For K-class problems:
- **Per-class AUC**: How well does the model separate class k from all others?
- **Micro AUC**: Overall discriminative ability (sample-weighted)
- **Macro AUC**: Average discriminative ability across classes (equal weight)

Choose based on your goal:
- Imbalanced data where all classes matter equally → macro AUC
- Standard supervised learning → micro AUC
- Specific class performance → per-class AUC

### Bootstrap Stability

The random seed ensures identical results across runs:
```python
# Both calls produce identical results
result1 = compute_roc_diagnostics(y_true, y_proba, random_seed=42)
result2 = compute_roc_diagnostics(y_true, y_proba, random_seed=42)

assert result1.per_class[1].auc == result2.per_class[1].auc
```

For different random data subsets, use different seeds:
```python
# Different bootstrap samples for validation
result_fold1 = compute_roc_diagnostics(y_true, y_proba, random_seed=1)
result_fold2 = compute_roc_diagnostics(y_true, y_proba, random_seed=2)
```

## Common Issues

### Issue: CI is very narrow or [AUC, AUC]
**Cause**: Perfectly separable data or too few unique probability values.
**Solution**: Use more realistic data or check if model is overfitting.

### Issue: Multiclass ROC is slow
**Cause**: Large number of classes or large bootstrap samples.
**Solution**: Reduce `n_bootstrap` for exploratory analysis, increase for publication.

### Issue: Different results with different random seeds
**Cause**: This is expected due to bootstrap randomness.
**Solution**: Use consistent seed for reproducibility or report mean±std over seeds.

## See Also

- [Multivariate QC System](../concepts/qc_system.md) for model quality control
- [Chemometrics Validation](../api/chemometrics.md) for model validation workflows
- [Classification Metrics](../api/metrics.md) for additional evaluation methods
