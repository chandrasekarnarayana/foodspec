# ML API Reference

!!! info "Module Purpose"
    Machine learning for classification, regression, model evaluation, and multi-modal fusion.

---

## Quick Navigation

| Function | Purpose | Common Use |
|----------|---------|------------|
| [`train_classifier()`](#train_classifier) | Train sklearn classifier | Quick model training |
| [`nested_cross_validate()`](#nested_cross_validate) | Unbiased hyperparameter tuning | Rigorous evaluation |
| [`compute_calibration_diagnostics()`](#compute_calibration_diagnostics) | Model calibration quality | Probability reliability |
| [`late_fusion_concat()`](#late_fusion_concat) | Feature-level fusion | Multi-modal combination |
| [`decision_fusion_vote()`](#decision_fusion_vote) | Decision-level fusion | Ensemble predictions |

---

## Common Patterns

### Pattern 1: Train + Evaluate Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(fs.x, fs.labels)

print(f"Training Accuracy: {model.score(fs.x, fs.labels):.1%}")

# Cross-validation
y_pred_cv = cross_val_predict(model, fs.x, fs.labels, cv=5)
print(f"CV Accuracy: {accuracy_score(fs.labels, y_pred_cv):.1%}")
print(classification_report(fs.labels, y_pred_cv))
```

### Pattern 2: Nested CV for Hyperparameter Tuning

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

# Inner CV: hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
inner_cv = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')

# Outer CV: unbiased evaluation
outer_scores = cross_val_score(inner_cv, fs.x, fs.labels, cv=5, scoring='accuracy')

print(f"Nested CV Accuracy: {outer_scores.mean():.1%} ± {outer_scores.std():.3f}")
print(f"Individual folds: {outer_scores}")
```

### Pattern 3: Multi-Modal Fusion

```python
from foodspec.ml.fusion import late_fusion_concat
from sklearn.ensemble import RandomForestClassifier

# Load two modalities
fs_raman = load_folder("data/raman/", modality="raman")
fs_ftir = load_folder("data/ftir/", modality="ftir")

# Feature-level fusion (concatenate features)
X_fused = late_fusion_concat(fs_raman.x, fs_ftir.x)
model_fused = RandomForestClassifier().fit(X_fused, fs_raman.labels)

print(f"Raman: {fs_raman.x.shape[1]} features")
print(f"FTIR: {fs_ftir.x.shape[1]} features")
print(f"Fused: {X_fused.shape[1]} features")
print(f"Accuracy: {model_fused.score(X_fused, fs_raman.labels):.1%}")
```

---

<a id="train_classifier"></a>
## train_classifier

High-level helper to train a classifier with sensible defaults.

**See Also:** [Model Evaluation Guide](../methods/chemometrics/model_evaluation_and_validation.md)

---

## nested_cross_validate

Nested cross-validation for unbiased model evaluation.

**Example:**

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001]
}

# Inner CV: tune hyperparameters
inner_cv = GridSearchCV(
    SVC(kernel='rbf'),
    param_grid,
    cv=3,  # 3-fold inner CV
    scoring='accuracy',
    n_jobs=-1
)

# Outer CV: evaluate generalization
outer_scores = cross_val_score(
    inner_cv,
    fs.x,
    fs.labels,
    cv=5,  # 5-fold outer CV
    scoring='accuracy'
)

print(f"Nested CV Results:")
print(f"  Mean Accuracy: {outer_scores.mean():.1%}")
print(f"  Std Dev: {outer_scores.std():.3f}")

# Fit final model on all data
inner_cv.fit(fs.x, fs.labels)
print(f"Best Parameters: {inner_cv.best_params_}")
```

**When to Use:**
- Hyperparameter tuning without test set leakage
- Unbiased performance estimation
- Comparing multiple algorithms fairly

**See Also:** [Model Evaluation Guide](../methods/chemometrics/model_evaluation_and_validation.md)

---

## compute_calibration_diagnostics

Model calibration quality assessment (probability reliability).

::: foodspec.ml.calibration.compute_calibration_diagnostics
    options:
      show_source: false
      heading_level: 3

**Example:**

```python
from foodspec.ml import compute_calibration_diagnostics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# Get probability predictions via CV
model = RandomForestClassifier(n_estimators=100, random_state=42)
y_proba_cv = cross_val_predict(
    model,
    fs.x,
    fs.labels,
    cv=5,
    method='predict_proba'
)

# Compute calibration metrics
cal_metrics = compute_calibration_diagnostics(
    y_true=fs.labels,
    y_proba=y_proba_cv
)

print(f"Calibration Diagnostics:")
print(f"  Brier Score: {cal_metrics['brier_score']:.3f}")
print(f"  ECE: {cal_metrics['expected_calibration_error']:.3f}")
```

**Metrics:**
- **Brier Score**: Mean squared error of probability predictions (0 = perfect)
- **ECE (Expected Calibration Error)**: Average deviation from perfect calibration
- **MCE (Max Calibration Error)**: Worst-case bin error

---

## late_fusion_concat

Feature-level fusion by concatenating feature matrices.

::: foodspec.ml.fusion.late_fusion_concat
    options:
      show_source: false
      heading_level: 3

**Example:**

```python
from foodspec.ml.fusion import late_fusion_concat

# Concatenate features (column-wise)
X_fused = late_fusion_concat(fs_raman.x, fs_ftir.x)
print(f"Fused shape: {X_fused.shape}")  # (n_samples, sum_of_features)
```

**When to Use:**
- Combining complementary information (Raman + FTIR)
- Early fusion strategy
- When modalities have aligned samples

**See Also:** [Multimodal Workflows](../05-advanced-topics/multimodal_workflows.md)

---

<a id="decision_fusion_vote"></a>
## decision_fusion_vote

Decision-level fusion with majority or weighted voting.

::: foodspec.ml.fusion.decision_fusion_vote
    options:
      show_source: false
      heading_level: 3

**Example:**

```python
from foodspec.ml.fusion import decision_fusion_vote
from sklearn.ensemble import RandomForestClassifier

# Train separate models
model_raman = RandomForestClassifier().fit(fs_raman.x, fs_raman.labels)
model_ftir = RandomForestClassifier().fit(fs_ftir.x, fs_ftir.labels)

# Get predictions
preds_raman = model_raman.predict(fs_raman.x)
preds_ftir = model_ftir.predict(fs_ftir.x)

# Majority voting
preds_voted = decision_fusion_vote([preds_raman, preds_ftir])
acc_voted = (preds_voted == fs_raman.labels).mean()
print(f"Voted Accuracy: {acc_voted:.1%}")
```

**When to Use:**
- Combining independent classifiers
- Late fusion strategy
- Ensemble methods

---

## Cross-References

**Related Modules:**
- [Core](core.md) - `FoodSpectrumSet` data structure
- [Metrics](metrics.md) - Evaluation metrics
- [Chemometrics](chemometrics.md) - PLS-DA, PCA

**Related Workflows:**
- [Oil Authentication](../workflows/authentication/oil_authentication.md) - Classification workflow
- [Model Evaluation Guide](../methods/chemometrics/model_evaluation_and_validation.md)
