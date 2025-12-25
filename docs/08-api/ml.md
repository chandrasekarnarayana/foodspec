# ML API Reference

Machine learning module for classification, regression, and model evaluation.

## nested_cross_validate

Nested cross-validation for unbiased model evaluation.

::: foodspec.ml.nested_cv.nested_cross_validate
    options:
      show_source: false
      heading_level: 3

## compute_calibration_diagnostics

Model calibration quality assessment.

::: foodspec.ml.calibration.compute_calibration_diagnostics
    options:
      show_source: false
      heading_level: 3

## late_fusion_concat

Feature-level fusion of multiple modalities.

::: foodspec.ml.fusion.late_fusion_concat
    options:
      show_source: false
      heading_level: 3

## decision_fusion_vote

Decision-level fusion with majority voting.

::: foodspec.ml.fusion.decision_fusion_vote
    options:
      show_source: false
      heading_level: 3

---

## Usage Examples

### Nested Cross-Validation

Nested cross-validation functionality is available through scikit-learn's `GridSearchCV` with nested CV loops. See FoodSpec examples for implementation patterns.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

# Nested CV pattern
inner_cv = GridSearchCV(SVC(), {'C': [0.1, 1, 10]}, cv=3)
scores = cross_val_score(inner_cv, X_train, y_train, cv=5)
print(f"Test Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Calibration Diagnostics

```python
from foodspec.ml import compute_calibration_diagnostics

cal_metrics = compute_calibration_diagnostics(y_true, y_pred_proba)

print(f"Brier Score: {cal_metrics['brier_score']:.3f}")
print(f"ECE: {cal_metrics['expected_calibration_error']:.3f}")
```
