# Nested Cross-Validation

Nested cross-validation provides unbiased performance estimates when hyperparameter tuning is required. FoodSpec v2 supports nested CV through the `NestedCVRunner` when `ValidationSpec.nested=True`.

## Why Nested CV?

**Problem:** Standard cross-validation with hyperparameter tuning can lead to optimistically biased performance estimates because the same data is used for both tuning and evaluation.

**Solution:** Nested CV separates hyperparameter selection (inner loop) from performance estimation (outer loop):

- **Outer loop**: Splits data into training and test sets for unbiased evaluation
- **Inner loop**: For each outer fold, run CV on training data to select hyperparameters
- **Final model**: Train on full outer training set with selected hyperparameters
- **Evaluation**: Report metrics only on outer test sets (unbiased)

## Architecture

```
Outer CV (Performance Estimation)
├─ Fold 1
│  ├─ Inner CV (Hyperparameter Selection)
│  │  ├─ Try C=0.1 → avg accuracy=0.85
│  │  ├─ Try C=1.0 → avg accuracy=0.88  ← Best
│  │  └─ Try C=10.0 → avg accuracy=0.86
│  ├─ Train final model with C=1.0 on full fold 1 training set
│  └─ Evaluate on fold 1 test set → accuracy=0.87
├─ Fold 2
│  ├─ Inner CV → selects C=10.0
│  ├─ Train with C=10.0
│  └─ Evaluate on fold 2 test set → accuracy=0.89
└─ Fold 3
   ├─ Inner CV → selects C=1.0
   ├─ Train with C=1.0
   └─ Evaluate on fold 3 test set → accuracy=0.86

Final Result: Mean accuracy=0.873 ± 0.012 (unbiased)
Hyperparameters per fold: [{"C": 1.0}, {"C": 10.0}, {"C": 1.0}]
```

## Usage

### Basic Example

```python
from foodspec.validation.nested import NestedCVRunner
from foodspec.models import LogisticRegressionClassifier
import numpy as np

# Estimator factory: creates model with given hyperparameters
def make_model(**params):
    return LogisticRegressionClassifier(**params)

# Define hyperparameter grid
param_grid = {
    "C": [0.1, 1.0, 10.0],
    "max_iter": [100, 200, 500],
}

# Create nested CV runner
runner = NestedCVRunner(
    estimator_factory=make_model,
    param_grid=param_grid,
    n_outer_splits=5,  # 5-fold outer CV for performance estimation
    n_inner_splits=3,  # 3-fold inner CV for hyperparameter tuning
    tuning_metric="accuracy",  # Metric to optimize in inner loop
    seed=42,
)

# Run nested CV
X = ...  # Feature matrix
y = ...  # Labels
result = runner.evaluate(X, y)

# Inspect results
print(f"Accuracy: {np.mean([m['accuracy'] for m in result.fold_metrics]):.3f}")
print(f"95% CI: {result.bootstrap_ci['accuracy']}")
print(f"Hyperparameters per fold:")
for i, params in enumerate(result.hyperparameters_per_fold):
    print(f"  Fold {i}: {params}")
```

### With Protocol Configuration

```python
from foodspec.core.protocol import ProtocolV2, ValidationSpec

protocol = ProtocolV2(
    data=...,
    task=...,
    model=ModelSpec(
        family="sklearn",
        estimator="logreg",
        params={"random_state": 0},
    ),
    validation=ValidationSpec(
        scheme="group_kfold",
        group_key="batch",
        nested=True,  # Enable nested CV
        metrics=["accuracy", "macro_f1", "auroc"],
    ),
)
```

## Parameters

### NestedCVRunner

- **`estimator_factory`**: Callable that takes hyperparameters as kwargs and returns an Estimator
- **`param_grid`**: Dict mapping hyperparameter names to lists of values to try
  - Example: `{"C": [0.1, 1.0, 10.0], "max_iter": [100, 200]}`
  - If `None` or empty, runs standard CV without tuning
- **`n_outer_splits`**: Number of outer CV folds (default: 5)
- **`n_inner_splits`**: Number of inner CV folds for tuning (default: 3)
- **`tuning_metric`**: Metric to optimize in inner loop (default: `"accuracy"`)
  - Options: `"accuracy"`, `"macro_f1"`, `"auroc"`
- **`seed`**: Random seed for deterministic splits (default: 0)
- **`output_dir`**: Directory for artifact output (optional)

### GridSearchTuner

- Exhaustive grid search over hyperparameter combinations
- Evaluates each combination via inner CV
- Selects hyperparameters with best average performance

## Artifacts

When `output_dir` is specified, nested CV saves:

```
output_dir/
├── predictions.csv           # Per-sample predictions from outer test sets
├── metrics.csv               # Per-fold metrics from outer loop
├── hyperparameters_per_fold.csv  # Selected hyperparameters for each outer fold
└── plots/                    # (if visualization enabled)
```

### hyperparameters_per_fold.csv

```csv
fold_id,C,max_iter
0,1.0,100
1,10.0,200
2,1.0,100
3,1.0,100
4,10.0,100
```

Each row shows which hyperparameters were selected by inner CV for that outer fold.

## Metrics Interpretation

**Key principle:** Metrics are computed **only on outer test sets** (never seen during hyperparameter tuning).

- **`fold_metrics`**: Per-fold metrics from outer loop (unbiased)
- **`bootstrap_ci`**: Confidence intervals computed from outer fold metrics
- Inner loop metrics are used for tuning but **not reported**

## Grouped Cross-Validation

Nested CV respects group structure in both loops:

```python
# Grouped nested CV (e.g., leave-one-batch-out)
groups = ["batch_A", "batch_A", "batch_B", "batch_B", ...]

runner = NestedCVRunner(
    estimator_factory=make_model,
    param_grid=param_grid,
    n_outer_splits=5,
    n_inner_splits=3,
)

result = runner.evaluate(X, y, groups=groups)
```

- **Outer loop**: Ensures entire groups are held out (e.g., leave-one-batch-out)
- **Inner loop**: Preserves group structure during hyperparameter tuning

## Comparison: Standard vs Nested CV

| Aspect | Standard CV | Nested CV |
|--------|------------|-----------|
| **Hyperparameter tuning** | Manual or separate | Automated in inner loop |
| **Performance estimate** | Biased if tuned on same data | Unbiased |
| **Computation time** | Faster (1 loop) | Slower (nested loops) |
| **Use case** | Fixed hyperparameters | Unknown optimal hyperparameters |
| **Output** | Metrics only | Metrics + hyperparameters per fold |

## When to Use Nested CV

✅ **Use nested CV when:**
- Hyperparameters are unknown and need tuning
- Unbiased performance estimates are critical (e.g., method comparison, reporting)
- You have sufficient data for nested splits

❌ **Skip nested CV when:**
- Hyperparameters are predetermined (use standard CV)
- Dataset is too small for nested splits
- Computational budget is limited (nested CV is ~10x slower)

## Computational Cost

**Outer folds × Inner folds × Grid size**

Example:
- 5 outer folds
- 3 inner folds  
- Grid with 9 combinations (3 C values × 3 max_iter values)

**Total fits:** 5 × 3 × 9 = 135 model fits

**Standard CV:** 5 fits

## Best Practices

### 1. Grid Design
```python
# Too fine (expensive, overfitting risk)
param_grid = {"C": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]}

# Good (reasonable coverage)
param_grid = {"C": [0.1, 1.0, 10.0]}

# Too coarse (might miss optimum)
param_grid = {"C": [1.0]}
```

### 2. Inner/Outer Split Ratio
- **Outer:** More folds = less variance, more computation
  - Typical: 5–10 folds
- **Inner:** Fewer folds = faster tuning
  - Typical: 3–5 folds

### 3. Seed for Reproducibility
```python
runner = NestedCVRunner(..., seed=42)  # Deterministic splits
```

### 4. Hyperparameter Variability
After nested CV, check if different folds selected different hyperparameters:

```python
hyperparams = result.hyperparameters_per_fold
print(f"Unique C values selected: {set(h['C'] for h in hyperparams)}")
```

**High variability** → Dataset characteristics vary across folds (expected for small datasets)  
**Low variability** → Stable optimal hyperparameters (good sign)

## Examples

### Example 1: Logistic Regression with C Tuning

```python
from foodspec.validation.nested import NestedCVRunner
from foodspec.models import LogisticRegressionClassifier

runner = NestedCVRunner(
    estimator_factory=lambda **p: LogisticRegressionClassifier(**p),
    param_grid={"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    n_outer_splits=5,
    n_inner_splits=3,
    tuning_metric="accuracy",
    seed=42,
)

result = runner.evaluate(X, y)

# Report unbiased performance
import numpy as np
accuracies = [m["accuracy"] for m in result.fold_metrics]
print(f"Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
print(f"95% CI: {result.bootstrap_ci['accuracy']}")

# Inspect hyperparameter selection
for i, params in enumerate(result.hyperparameters_per_fold):
    print(f"Fold {i}: C={params['C']}")
```

### Example 2: LOBO Validation with Nested CV

```python
# Leave-one-batch-out with hyperparameter tuning
groups = metadata["batch"].values

runner = NestedCVRunner(
    estimator_factory=lambda **p: LogisticRegressionClassifier(**p),
    param_grid={"C": [0.1, 1.0, 10.0]},
    n_outer_splits=5,  # Will use GroupKFold if enough groups
    n_inner_splits=3,
    seed=42,
)

result = runner.evaluate(X, y, groups=groups)

# Each outer fold leaves out entire batches
# Inner loop tunes on remaining batches
```

### Example 3: No Hyperparameter Tuning (Standard CV)

```python
# If no tuning needed, pass empty param_grid
runner = NestedCVRunner(
    estimator_factory=lambda: LogisticRegressionClassifier(C=1.0),
    param_grid=None,  # or param_grid={}
    n_outer_splits=5,
    seed=42,
)

result = runner.evaluate(X, y)

# All folds will have hyperparameters_per_fold = {}
```

## Integration with Orchestrator

The `ExecutionEngine` automatically uses `NestedCVRunner` when `ValidationSpec.nested=True`:

```python
from foodspec.core.orchestrator import ExecutionEngine
from foodspec.core.protocol import ProtocolV2

protocol = ProtocolV2(
    data=DataSpec(...),
    task=TaskSpec(...),
    model=ModelSpec(
        estimator="logreg",
        params={"random_state": 0},
    ),
    validation=ValidationSpec(
        scheme="group_kfold",
        nested=True,  # Trigger nested CV
    ),
)

engine = ExecutionEngine(protocol, output_dir="run_output", seed=42)
result = engine.execute()

# Hyperparameters per fold recorded in manifest
manifest = engine.manifest
print(manifest.hyperparameters_per_fold)
```

## Troubleshooting

### Issue: "Not enough groups for GroupKFold"

**Cause:** Too few unique groups for requested splits

**Solution:** Reduce `n_outer_splits` or `n_inner_splits`, or use ungrouped CV

### Issue: Nested CV takes too long

**Solutions:**
- Reduce grid size (fewer hyperparameter combinations)
- Reduce `n_inner_splits` (e.g., from 5 to 3)
- Reduce `n_outer_splits` (e.g., from 10 to 5)
- Use coarser hyperparameter spacing

### Issue: High hyperparameter variability across folds

**Explanation:** Different folds have different optimal hyperparameters

**Actions:**
- Check dataset size (small datasets have high variability)
- Check group structure (some batches may be outliers)
- Consider using median/mode of selected hyperparameters for final model

## References

- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
- Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079-2107.

## See Also

- [Validation Strategies](../methods/validation.md)
- [Standard Cross-Validation](./evaluation.md)
- [Model Selection](../methods/models.md)
- [Grouped Validation](./grouped_validation.md)
