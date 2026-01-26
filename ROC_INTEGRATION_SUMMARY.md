"""ROC/AUC Integration Summary

This document summarizes the successful integration of ROC/AUC/threshold evaluation 
into the FoodSpec modeling pipeline.
"""

# Integration Overview

## Objective
Integrate ROC/AUC/threshold evaluation into the existing modeling pipeline so that 
whenever a classification task runs with predicted probabilities available, ROC 
diagnostics are computed and optionally saved to disk.

## Implementation Summary

### 1. Modified Files

#### src/foodspec/modeling/api.py
- **Added imports**: `compute_roc_diagnostics` and `save_roc_artifacts`
- **Modified FitPredictResult dataclass** (lines 37-56):
  - Added field: `roc_diagnostics: Optional[Any] = None`
  - Added field: `roc_artifacts: Dict[str, str] = field(default_factory=dict)`
  
- **Updated fit_predict() function signature** (lines 379-398):
  - Added parameter: `compute_roc: bool = True` (default enabled)
  - Added parameter: `roc_output_dir: Optional[str] = None` (optional artifact saving)
  - Added parameter: `roc_n_bootstrap: int = 1000` (controls CI bootstrap samples)
  
- **Added ROC computation logic** (lines 513-547):
  - After CV prediction aggregation, conditionally compute ROC diagnostics
  - Only runs for classification tasks with probabilities available
  - Computes `RocDiagnosticsResult` with per-class AUC, micro-averaged metrics, bootstrap CIs
  - Saves artifacts to disk if `roc_output_dir` provided
  - Graceful error handling with warnings on failure
  
- **Updated return statement** (lines 570-581):
  - Populated `roc_diagnostics` and `roc_artifacts` fields in `FitPredictResult`

### 2. New Module: src/foodspec/modeling/diagnostics/artifacts.py
Created a comprehensive artifact saving utility module with the following functions:

- **save_roc_artifacts()**: Main entry point
  - Creates directory structure: `output_dir/tables/`, `output_dir/json/`, `output_dir/figures/`
  - Saves CSV summaries, JSON diagnostics, and ROC plots
  - Returns artifact dictionary with file paths
  
- **_build_roc_summary_df()**: Builds per-class AUC summary with:
  - Class labels
  - AUC values
  - Bootstrap confidence intervals (lower/upper)
  - Micro-averaged and macro-averaged metrics for multiclass
  
- **_build_thresholds_df()**: Builds optimal thresholds with:
  - Policy-specific optimal thresholds (Youden, etc.)
  - Sensitivity, specificity, PPV, NPV at threshold
  
- **_serialize_roc_result()**: Converts RocDiagnosticsResult to JSON-compatible format

- **_plot_roc_curves()**: Creates ROC plots for:
  - Binary classification
  - Multiclass (per-class One-vs-Rest)
  - Micro-averaged ROC for multiclass

### 3. Artifact Output Structure
When `roc_output_dir` is provided, artifacts are saved as:
```
output_dir/
├── tables/
│   ├── roc_summary.csv          # Per-class AUC + CI
│   └── roc_thresholds.csv       # Optimal thresholds
├── json/
│   └── roc_diagnostics.json     # Full ROC result serialization
└── figures/
    ├── roc_curve_per_class.png  # Per-class ROC curves
    ├── roc_curve_micro.png      # Micro-averaged (multiclass)
    └── roc_curve_macro.png      # Macro-averaged (multiclass)
```

### 4. Integration Test Suite
Created comprehensive tests in `tests/modeling/test_roc_integration.py`:

- ✅ `test_roc_computed_by_default`: Verifies ROC computed by default
- ✅ `test_roc_can_be_disabled`: Verifies compute_roc=False skips computation
- ✅ `test_roc_artifacts_saved`: Verifies artifacts saved to disk correctly
- ✅ `test_roc_multiclass`: Verifies multiclass support
- ✅ `test_roc_respects_seed`: Verifies reproducibility with same seed
- ✅ `test_roc_skipped_for_regression`: Verifies ROC not computed for regression
- ✅ `test_roc_bootstrap_parameter`: Verifies bootstrap CI computation
- ✅ `test_roc_results_consistency`: Verifies results independent of compute_roc flag

**All 8 tests passing** ✓

## Key Features

### Model-Agnostic
- Works with any scikit-learn compatible classifier
- Supports decision_function with sigmoid conversion
- Automatically handles binary and multiclass problems

### Robust Error Handling
- Gracefully handles missing probabilities
- Detects single-class folds and warns appropriately
- Wraps ROC computation in try-except with informative warnings

### Reproducible
- Uses global random seed for bootstrap sampling
- Same seed produces identical ROC metrics and CIs
- Metadata stored with computation details

### Backward Compatible
- ROC computation enabled by default (compute_roc=True)
- Can be disabled for performance or specific use cases
- Existing code continues to work unchanged

### Flexible Output
- Optional artifact saving (no overhead if roc_output_dir=None)
- Structured output suitable for downstream report generation
- ROC diagnostics always available in memory

## Usage Example

```python
from foodspec.modeling.api import fit_predict

# With ROC diagnostics and artifact saving
result = fit_predict(
    X, y,
    model_name="logistic_regression",
    scheme="kfold",
    outer_splits=5,
    seed=42,
    compute_roc=True,  # Enable ROC (default)
    roc_output_dir="/path/to/output",  # Save artifacts
    roc_n_bootstrap=1000,  # 1000 bootstrap samples for CI
)

# Access ROC diagnostics
print(result.roc_diagnostics.per_class)  # Per-class metrics
print(result.roc_diagnostics.macro_auc)  # Macro-averaged AUC
print(result.roc_diagnostics.optimal_thresholds)  # Optimal thresholds

# Access artifact paths
print(result.roc_artifacts)  # Dict of saved file paths
```

## Technical Details

### Integration Point
ROC computation occurs after CV prediction aggregation:
1. Cross-validation completes and predictions are aggregated
2. y_true_all_arr and y_proba_all_arr constructed from all folds
3. ROC diagnostics computed via compute_roc_diagnostics()
4. Artifacts optionally saved via save_roc_artifacts()
5. Results stored in FitPredictResult

### Classification-Only Execution
ROC computation is skipped for:
- Regression tasks (outcome_type=REGRESSION)
- Count tasks (outcome_type=COUNT)
- Cases where y_proba is None
- Cases where compute_roc=False

### Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib (optional, for plots)
- foodspec.modeling.diagnostics.roc (RocDiagnosticsResult)

## Testing

All tests pass with pytest:
```bash
pytest tests/modeling/test_roc_integration.py::TestROCIntegration -v
# 8 passed in 55.54s
```

Test coverage includes:
- Binary and multiclass classification
- Artifact saving and disk verification
- Reproducibility with seeds
- Parameter validation
- Regression task handling
- Bootstrap CI computation
- Consistency across runs

## Next Steps

1. Integration with reporting system to display ROC visualizations
2. Threshold policy customization (currently Youden only)
3. Performance optimization for large datasets
4. Integration with model card generation
5. Cross-validation specific threshold selection strategies

## Backward Compatibility

✅ All existing code continues to work unchanged
✅ ROC computation is optional (can disable)
✅ No breaking changes to fit_predict() API
✅ Existing tests continue to pass
