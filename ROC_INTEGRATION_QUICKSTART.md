"""Quick Reference: ROC Integration Usage Guide

This guide shows how to use the new ROC/AUC integration in fit_predict().
"""

# Quick Start

## Basic Usage (ROC computed by default)
```python
from foodspec.modeling.api import fit_predict

result = fit_predict(
    X, y,
    model_name="logistic_regression",
    scheme="kfold",
    outer_splits=5,
    seed=42,
)

# ROC diagnostics available in result
print(result.roc_diagnostics.per_class)  # Per-class metrics
print(result.roc_diagnostics.macro_auc)  # Macro-averaged AUC
```

## With Artifact Saving
```python
result = fit_predict(
    X, y,
    model_name="logistic_regression",
    scheme="kfold",
    outer_splits=5,
    seed=42,
    compute_roc=True,
    roc_output_dir="/path/to/results",  # Saves CSV, JSON, PNG
)

# Access saved artifact paths
for artifact_type, file_path in result.roc_artifacts.items():
    print(f"{artifact_type}: {file_path}")
```

## Disabling ROC Computation
```python
result = fit_predict(
    X, y,
    model_name="logistic_regression",
    scheme="kfold",
    outer_splits=5,
    compute_roc=False,  # Skip ROC computation
)

# ROC not computed
assert result.roc_diagnostics is None
assert result.roc_artifacts == {}
```

## Custom Bootstrap Samples
```python
result = fit_predict(
    X, y,
    model_name="logistic_regression",
    scheme="kfold",
    outer_splits=5,
    roc_n_bootstrap=500,  # Fewer bootstrap samples for speed
)
```

# Accessing ROC Results

## Per-Class Metrics
```python
roc = result.roc_diagnostics

# Each class has AUC and bootstrap CI
for class_label, metrics in roc.per_class.items():
    print(f"Class {class_label}:")
    print(f"  AUC: {metrics.auc:.3f}")
    print(f"  CI: [{metrics.ci_lower:.3f}, {metrics.ci_upper:.3f}]")
```

## Multi-Class Metrics
```python
roc = result.roc_diagnostics

# Macro-averaged AUC (multiclass only)
if roc.macro_auc is not None:
    print(f"Macro AUC: {roc.macro_auc:.3f}")

# Micro-averaged ROC (multiclass only)
if roc.micro is not None:
    print(f"Micro AUC: {roc.micro.auc:.3f}")
```

## Optimal Thresholds
```python
roc = result.roc_diagnostics

# Threshold policy (e.g., Youden's J-statistic)
youden_result = roc.optimal_thresholds.get("youden")
if youden_result:
    print(f"Optimal threshold: {youden_result.threshold:.3f}")
    print(f"  Sensitivity: {youden_result.sensitivity:.3f}")
    print(f"  Specificity: {youden_result.specificity:.3f}")
```

## Metadata
```python
roc = result.roc_diagnostics

# Computation details
print(f"Method: {roc.metadata['method']}")
print(f"Bootstrap samples: {roc.metadata['n_bootstrap']}")
print(f"Seed: {roc.metadata['random_seed']}")
print(f"Warnings: {roc.metadata.get('warnings', [])}")
```

# Output Artifacts

When `roc_output_dir` is provided, the following files are saved:

## 1. ROC Summary (CSV)
File: `tables/roc_summary.csv`

Contains per-class AUC with confidence intervals:
```
class,auc,ci_lower,ci_upper
0,0.7824,0.6808,0.8663
1,0.7824,0.6808,0.8663
micro,,,
macro,0.7824,,
```

## 2. Optimal Thresholds (CSV)
File: `tables/roc_thresholds.csv`

Contains policy-specific thresholds:
```
policy,threshold,sensitivity,specificity
youden,0.6089,0.6200,0.8600
```

## 3. ROC Diagnostics (JSON)
File: `json/roc_diagnostics.json`

Contains full ROC result in JSON format (per-class FPR/TPR/AUC, etc.)

## 4. ROC Plots (PNG)
Files:
- `figures/roc_curve_per_class.png`: Per-class ROC curves
- `figures/roc_curve_micro.png`: Micro-averaged (multiclass)
- `figures/roc_curve_macro.png`: Macro-averaged (multiclass)

# API Parameters

## fit_predict() new parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compute_roc` | bool | `True` | Enable ROC diagnostics computation |
| `roc_output_dir` | str or None | `None` | Directory to save artifacts (optional) |
| `roc_n_bootstrap` | int | `1000` | Bootstrap samples for CI calculation |

# When ROC is Skipped

ROC computation is automatically skipped (no warning) when:
- `outcome_type != CLASSIFICATION` (regression/count tasks)
- `y_proba` is `None` (model doesn't support predict_proba)
- `compute_roc=False` (explicitly disabled)

ROC computation will warn if:
- Single class detected in fold data
- ROC computation fails for any reason

# Performance Considerations

- ROC computation adds ~5-10% overhead for typical datasets
- Artifact saving adds minimal overhead (PNG rendering slowest)
- Bootstrap CI calculation uses global seed (reproducible, deterministic)

For large datasets, consider:
- Reducing `roc_n_bootstrap` (e.g., 100-500 instead of 1000)
- Disabling `roc_output_dir` if plots not needed (keeps results in memory only)

# Examples

## Binary Classification with Full Diagnostics
```python
import numpy as np
from foodspec.modeling.api import fit_predict
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=20, n_classes=2)

result = fit_predict(
    X, y,
    model_name="logistic_regression",
    scheme="kfold",
    outer_splits=5,
    seed=42,
    compute_roc=True,
    roc_output_dir="/tmp/roc_results",
    roc_n_bootstrap=1000,
)

# Examine results
print(f"Overall accuracy: {result.metrics['accuracy']:.3f}")
for class_label, metrics in result.roc_diagnostics.per_class.items():
    print(f"Class {class_label} AUC: {metrics.auc:.3f}")
```

## Multiclass Classification
```python
X, y = make_classification(
    n_samples=300, 
    n_features=20, 
    n_classes=3,
    n_clusters_per_class=1,
)

result = fit_predict(
    X, y,
    model_name="logistic_regression",
    scheme="kfold",
    outer_splits=5,
    compute_roc=True,
    roc_output_dir="/tmp/roc_results",
)

print(f"Macro AUC: {result.roc_diagnostics.macro_auc:.3f}")
if result.roc_diagnostics.micro:
    print(f"Micro AUC: {result.roc_diagnostics.micro.auc:.3f}")
```

## Reproducible Results
```python
# Same seed -> identical ROC metrics
result1 = fit_predict(X, y, seed=42, roc_n_bootstrap=1000)
result2 = fit_predict(X, y, seed=42, roc_n_bootstrap=1000)

for class_label in result1.roc_diagnostics.per_class:
    auc1 = result1.roc_diagnostics.per_class[class_label].auc
    auc2 = result2.roc_diagnostics.per_class[class_label].auc
    assert np.isclose(auc1, auc2)
```

# Troubleshooting

## ROC not computed but compute_roc=True?
- Check outcome_type: ROC only for classification
- Check y_proba: Model must support predict_proba
- Check warnings: Failed computation will warn (see stderr)

## Missing artifact files?
- Verify roc_output_dir exists and is writable
- Check available disk space
- Review matplotlib installation (needed for plots)

## Slow ROC computation?
- Reduce roc_n_bootstrap (e.g., 100 instead of 1000)
- Skip artifact saving (no roc_output_dir)
- Use fewer bootstrap samples for quick exploration

# See Also

- [Full Integration Summary](ROC_INTEGRATION_SUMMARY.md)
- [Test Suite](tests/modeling/test_roc_integration.py)
- [ROC Diagnostics Module](src/foodspec/modeling/diagnostics/roc.py)
- [Artifact Saving Module](src/foodspec/modeling/diagnostics/artifacts.py)
