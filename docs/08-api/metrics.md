# Metrics API Reference

!!! info "Module Purpose"
    Evaluation metrics for classification and regression from scikit-learn.

---

## Quick Navigation

| Function | Purpose | Use Case |
|----------|---------|----------|
| `accuracy_score()` | Classification accuracy | Overall correctness |
| `classification_report()` | Comprehensive metrics | Multi-class evaluation |
| `confusion_matrix()` | Error analysis | Identify misclassifications |
| `r2_score()` | R² coefficient | Regression goodness-of-fit |
| `mean_squared_error()` | MSE/RMSE | Regression error |
| `mean_absolute_error()` | MAE | Robust regression error |

---

## Common Patterns

### Pattern 1: Classification Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Get predictions
y_pred = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(report)
```

### Pattern 2: Regression Evaluation

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Get predictions
y_pred = model.predict(X_test)

# Compute metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
```

---

## Classification Metrics

::: sklearn.metrics.accuracy_score
    options:
      show_source: false

::: sklearn.metrics.classification_report
    options:
      show_source: false

::: sklearn.metrics.confusion_matrix
    options:
      show_source: false

::: sklearn.metrics.roc_auc_score
    options:
      show_source: false

---

## Regression Metrics

::: sklearn.metrics.r2_score
    options:
      show_source: false

::: sklearn.metrics.mean_squared_error
    options:
      show_source: false

::: sklearn.metrics.mean_absolute_error
    options:
      show_source: false

---

## Cross-References

**Related Modules:**
- [ML](ml.md) - Model training
- [Chemometrics](chemometrics.md) - Model validation

**External:**
- [scikit-learn Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
