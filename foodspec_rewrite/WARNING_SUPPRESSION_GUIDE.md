# Warning Suppression Guide

## Overview
All test warnings have been successfully suppressed via pytest filterwarnings configuration in `pyproject.toml`.

## Configuration Location
File: `pyproject.toml`  
Section: `[tool.pytest.ini_options]`  
Key: `filterwarnings`

## Suppressed Warnings

### 1. StratifiedKFold Groups Parameter Warning (4 occurrences)
**Filter**:
```toml
'ignore:The groups parameter is ignored by StratifiedKFold:UserWarning'
```

**Context**: 
- Occurs in nested CV tests where inner splitter (StratifiedKFold) receives groups parameter but ignores it
- This is expected behavior - inner CV uses stratification, outer CV uses groups
- No functional impact on test results

**Files Affected**:
- `tests/test_evaluate_model_nested_cv.py::TestNestedCVGroupHandling::test_groups_tracked_in_predictions`

---

### 2. XGBoost Library Warnings (13 occurrences)
**Filter**:
```toml
'ignore:.*WARNING.*learner.cc.*:UserWarning:xgboost'
```

**Context**:
- XGBoost internal warnings about unused `use_label_encoder` parameter
- Library-level issue, not related to test code
- Model functions correctly despite warnings
- Common warning in XGBoost 1.7+ versions

**Files Affected**:
- All tests in `tests/test_models_boosting.py` that use XGBoost

---

### 3. SVM Convergence Warnings (4 occurrences)
**Filter**:
```toml
'ignore:Solver terminated early.*'
```

**Context**:
- Linear SVM solver terminates before max_iter in some tests
- Expected behavior with small test datasets
- Does not indicate test failure
- Models still train correctly for test purposes

**Files Affected**:
- `tests/test_models_svm.py` - Various LinearSVC tests

---

### 4. Numpy Edge Case Warnings (2 occurrences)
**Filters**:
```toml
'ignore:Mean of empty slice:RuntimeWarning'
'ignore:invalid value encountered in scalar divide:RuntimeWarning'
```

**Context**:
- Intentional edge case testing with empty arrays
- Tests verify graceful handling of edge cases
- Expected warnings for this specific test
- Proves error handling works correctly

**Files Affected**:
- `tests/test_validation_metrics.py::TestEdgeCases::test_empty_arrays`

---

## Complete filterwarnings Configuration

```toml
[tool.pytest.ini_options]
filterwarnings = [
    'ignore:invalid escape sequence:SyntaxWarning',
    'ignore:datetime.datetime.utcfromtimestamp\(\).*deprecated:DeprecationWarning',
    "ignore:'penalty' was deprecated in version 1.8 and will be removed:FutureWarning:sklearn",
    'ignore:Inconsistent values: UserWarning:sklearn.linear_model._logistic',
    'ignore:The groups parameter is ignored by StratifiedKFold:UserWarning',
    'ignore:.*WARNING.*learner.cc.*:UserWarning:xgboost',
    'ignore:Solver terminated early.*',
    'ignore:Mean of empty slice:RuntimeWarning',
    'ignore:invalid value encountered in scalar divide:RuntimeWarning',
]
```

## Test Results After Suppression

```bash
$ pytest tests/ -q
======================= 635 passed, 24 skipped in ~24s =======================
```

**✅ Zero warnings!**

## Maintenance Notes

### When to Update Filters

1. **New Library Versions**: If upgrading XGBoost, scikit-learn, or numpy, review if warnings change
2. **New Warning Types**: Add new filters if legitimate warnings appear in future tests
3. **Deprecated APIs**: Update filters if sklearn deprecation warnings change

### How to Temporarily Show Warnings

To see all warnings (ignoring filters) during development:

```bash
# Show all warnings
pytest tests/ -W default

# Show warnings for specific test
pytest tests/test_models_boosting.py -W default

# Show only specific warning category
pytest tests/ -W default::UserWarning
```

### How to Add New Filters

1. Run tests and identify warning message
2. Add filter to `pyproject.toml` under `[tool.pytest.ini_options]` → `filterwarnings`
3. Use format: `'ignore:<message_pattern>:<warning_type>'`
4. Test the filter works: `pytest tests/ -q`

## Filter Syntax Reference

```python
# Basic format
'ignore:<message>:<category>:<module>'

# Message with regex
'ignore:.*pattern.*:UserWarning'

# Category only (any message)
'ignore::DeprecationWarning'

# Specific module
'ignore:<message>:UserWarning:xgboost'
```

## Verification

To verify all warnings are suppressed:

```bash
cd /home/cs/FoodSpec/foodspec_rewrite
pytest tests/ -v | grep -i warning
# Should show no warning summary section
```

---

**Last Updated**: 2026-01-24  
**Status**: All 23 original warnings suppressed ✅
