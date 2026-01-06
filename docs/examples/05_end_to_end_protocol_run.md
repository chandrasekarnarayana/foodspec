# End-to-End Protocol Run: Unified FoodSpec API

**Level**: Capstone (Advanced)  
**Runtime**: ~3 seconds  
**Key Concepts**: Chainable API, workflow composition, reproducibility, audit trails

---

## What You Will Learn

In this capstone example, you'll learn how to:
- Master the Phase 1 unified FoodSpec API
- Compose complete workflows: QC → Preprocess → Train → Evaluate → Export
- Leverage built-in diagnostics for quality assurance
- Implement reproducible science with full audit trails
- Export results with provenance and complete parameter documentation

After completing this example, you'll understand best practices for professional, reproducible analysis workflows that meet regulatory and scientific standards.

---

## Prerequisites

- Completion of at least 2-3 prior examples (Oil Auth, Heating, Mixture)
- Understanding of Python classes and method chaining
- Knowledge of cross-validation and model evaluation
- Familiarity with JSON and parameter documentation
- `numpy`, `pandas`, `scikit-learn`, `foodspec` installed

**Optional background**: Read [Protocols & YAML](../user-guide/protocols_and_yaml.md) and [Phase 1 Quickstart](../getting-started/quickstart_protocol.md)

---

## The Problem

**Real-world scenario**: You're implementing a production system for oil classification. It needs to:
1. Pass quality checks on incoming data
2. Apply preprocessing consistently
3. Train a robust classifier with validation
4. Generate detailed diagnostics
5. Export with complete audit trail (what was done, when, by whom, parameters used)

**Goal**: Build and validate a reproducible end-to-end pipeline.

---

## Step 1: Initialize & Quality Check

```python
import numpy as np
import pandas as pd
from foodspec.core import FoodSpec
from foodspec.datasets import load_example_data

# Load or create your spectroscopy dataset
X, y = load_example_data("oil_classification")  # or your own data

# Initialize FoodSpec with protocol name
fs = FoodSpec(task="classification", name="oil_auth_production")

# Perform quality checks (QC)
qc_report = fs.quality_check(
    X=X, 
    y=y,
    check_type="complete",  # checks: missing values, outliers, class balance, etc.
)

print("Quality Check Report:")
print(f"  Data health score: {qc_report['health']:.2f}")
print(f"  Issues detected: {qc_report['issues']}")
print(f"  Recommendations: {qc_report['recommendations']}")

# Proceed only if quality is acceptable
if qc_report["health"] < 0.5:
    raise ValueError("Data quality too low. Address issues before proceeding.")
```

**What's happening**:
- QC checks data for common issues (missing values, extreme outliers, class imbalance)
- Health score ranges 0–1 (1 = perfect quality)
- Issues and recommendations guide data improvement
- **Production practice**: Never skip QC; log results for audit trail

---

## Step 2: Preprocessing Pipeline

```python
# Add preprocessing steps (chainable API)
fs.add_preprocessing_step(
    "baseline_removal",
    method="polyfit",
    order=5,
    description="Remove instrumental baseline"
)

fs.add_preprocessing_step(
    "normalization",
    method="vector",
    description="L2 normalization to remove intensity effects"
)

fs.add_preprocessing_step(
    "snv",  # Standard Normal Variate
    method="snv",
    description="Scatter correction for multiplicative effects"
)

# Apply preprocessing
X_processed = fs.preprocess(X, fit=True)  # fit=True: learn parameters from training data

print(f"Original shape: {X.shape}")
print(f"Processed shape: {X_processed.shape}")
print(f"Preprocessing pipeline: {[s['method'] for s in fs.preprocessing_steps]}")
```

**Interpretation**:
- **Baseline removal**: Removes instrument artifacts (polynomial fit)
- **Normalization**: Removes sample thickness effects (L2 norm)
- **SNV**: Corrects multiplicative scatter effects (concentration-independent)
- Preprocessing parameters are stored for consistency

---

## Step 3: Train with Cross-Validation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Initialize model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# Perform cross-validation with detailed metrics
cv_results = cross_validate(
    model, X_processed, y,
    cv=5,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    return_train_score=True
)

# Summarize results
print("\nCross-Validation Results (5-fold):")
print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
print(f"  Precision: {cv_results['test_precision_macro'].mean():.3f} ± {cv_results['test_precision_macro'].std():.3f}")
print(f"  Recall:    {cv_results['test_recall_macro'].mean():.3f} ± {cv_results['test_recall_macro'].std():.3f}")
print(f"  F1:        {cv_results['test_f1_macro'].mean():.3f} ± {cv_results['test_f1_macro'].std():.3f}")

# Check for overfitting
train_acc = cv_results['train_accuracy'].mean()
test_acc = cv_results['test_accuracy'].mean()
overfitting = train_acc - test_acc
print(f"\nOverfitting check:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test accuracy:  {test_acc:.3f}")
print(f"  Gap: {overfitting:.3f}")
if overfitting > 0.10:
    print("  ⚠️  Significant overfitting detected!")
else:
    print("  ✓ Model generalization is good")
```

**What's happening**:
- Cross-validation evaluates on unseen data (prevents overfitting)
- Multiple metrics provide comprehensive performance view
- Train-test gap indicates generalization quality
- Results stored for reporting and reproducibility

---

## Step 4: Generate Diagnostics

```python
# Generate comprehensive diagnostics
diagnostics = fs.generate_diagnostics(
    X_processed=X_processed,
    y=y,
    model=model,
    cv_results=cv_results,
    include_pca=True,
    include_confusion_matrix=True
)

print("\nDiagnostics Summary:")
print(f"  PCA variance (PC1): {diagnostics['pca_variance'][0]:.3f}")
print(f"  Feature importance (top 3): {diagnostics['feature_importance'][:3]}")
print(f"  Class distribution: {diagnostics['class_counts']}")
print(f"  Data health: {diagnostics['data_health']:.2f}")

# Store diagnostics for later review
import json
with open("diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2, default=str)  # default=str for non-JSON types
```

**What's happening**:
- Diagnostics provide comprehensive view of model and data
- PCA variance: How much spectral variation is captured by first few components
- Feature importance: Which wavelengths are most predictive
- Class distribution: Balance assessment
- All diagnostics stored for audit trail

---

## Step 5: Export with Provenance

```python
# Train final model on full dataset
model.fit(X_processed, y)

# Create comprehensive export with audit trail
export_data = {
    "metadata": {
        "timestamp": pd.Timestamp.now().isoformat(),
        "analyst": "Production System",
        "task": "oil_classification",
        "model": "RandomForestClassifier",
        "dataset": "oil_synthetic.csv"
    },
    "parameters": {
        "preprocessing": [
            {"method": s["method"], "params": s.get("params", {})}
            for s in fs.preprocessing_steps
        ],
        "model": model.get_params()
    },
    "performance": {
        "cv_accuracy": float(cv_results['test_accuracy'].mean()),
        "cv_f1": float(cv_results['test_f1_macro'].mean()),
        "cv_std": float(cv_results['test_accuracy'].std())
    },
    "diagnostics": diagnostics
}

# Save export
with open("export_oil_auth_model.json", "w") as f:
    json.dump(export_data, f, indent=2, default=str)

print("\n✓ Export complete!")
print(f"  Saved to: export_oil_auth_model.json")
print(f"  Model class: {model.__class__.__name__}")
print(f"  Training samples: {X_processed.shape[0]}")
print(f"  Spectral features: {X_processed.shape[1]}")
```

**What's happening**:
- **Metadata**: Timestamp, analyst, task, dataset (who, what, when)
- **Parameters**: Exact preprocessing steps + model hyperparameters (reproducibility)
- **Performance**: Cross-validation metrics (validation evidence)
- **Diagnostics**: All diagnostic data (transparency)
- Everything saved as JSON (human-readable, versionable)

---

## Step 6: Use the Trained Model

```python
# Load new unknown sample
X_unknown = pd.read_csv("unknown_oil_sample.csv", index_col=0).values

# Apply same preprocessing (using learned parameters)
X_unknown_processed = fs.preprocess(X_unknown, fit=False)  # fit=False: use stored parameters

# Make predictions
predictions = model.predict(X_unknown_processed)
probabilities = model.predict_proba(X_unknown_processed)

print("\nPrediction for Unknown Sample:")
print(f"  Predicted class: {predictions[0]}")
print(f"  Confidence: {probabilities.max():.3f}")
print(f"  All probabilities: {dict(zip(model.classes_, probabilities[0]))}")
```

**Critical point**: Preprocessing must use **same parameters** learned during training (fit=False).

---

## Full Working Script

See the production script with complete workflow:

📄 **[`examples/phase1_quickstart.py`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/phase1_quickstart.py)** – Full working code (139 lines)

---

## Key Takeaways

✅ **QC first**: Always check data quality before training  
✅ **Chainable API**: Compose workflows step-by-step with clear syntax  
✅ **Preprocessing consistency**: Learn parameters on training data, apply to test/production  
✅ **Cross-validation**: Essential for reliable performance estimates  
✅ **Comprehensive diagnostics**: Understand your model and data  
✅ **Full export**: Store metadata, parameters, metrics, and diagnostics for reproducibility  

---

## Production Best Practices

| Practice | Why | How |
|----------|-----|-----|
| **QC first** | Catch garbage early | Always check health score |
| **Preprocessing parameters** | Consistency | Save learned parameters, reuse on new data |
| **Cross-validation** | Prevents overfitting | Never evaluate on training data |
| **Hyperparameter tuning** | Model optimization | Use GridSearchCV or RandomizedSearchCV |
| **Diagnostics** | Transparency | Generate for every model |
| **Audit trail** | Reproducibility | Save metadata, parameters, metrics |
| **Version control** | Traceability | Commit models and exports to Git |

---

## Real-World Deployment

**Your production system would:**
1. Load unknown sample
2. Apply preprocessing (learned parameters)
3. Make prediction with trained model
4. Log prediction to audit trail
5. Alert if confidence below threshold
6. Store all results with timestamp

---

## Advanced Topics

**Want to go deeper?**
- **Hyperparameter tuning**: Optimize model parameters with GridSearchCV
- **Ensemble methods**: Combine multiple models for robustness
- **Feature selection**: Reduce wavelengths while maintaining accuracy
- **Retraining strategy**: When to retrain with new samples
- **Model monitoring**: Detect performance drift over time

See [Reproducible Pipelines Workflow](../workflows/end_to_end_pipeline.md) for complete details.

---

## Next Steps

1. **Try it**: Run the full script end-to-end
2. **Customize**: Modify preprocessing steps, model hyperparameters
3. **Test**: Make predictions on new data using trained model
4. **Deploy**: Save/load models, integrate into production system
5. **Learn more**: Read [Protocols & YAML](../user-guide/protocols_and_yaml.md)

---

## Interactive Notebook

For step-by-step exploration and parameter experimentation:

📓 **[`examples/tutorials/05_protocol_unified_api_teaching.ipynb`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/05_protocol_unified_api_teaching.ipynb)**

---

## Workflow Diagram

```
Load Data → QC Check → Preprocess → Train (CV) → Diagnostics → Export
             ↓         ↓            ↓
          Health   Parameters   Metrics
          Score    Stored       Analyzed
```

---

## Example Output Structure

```json
{
  "metadata": {
    "timestamp": "2026-01-06T14:30:00",
    "analyst": "Production System",
    "task": "oil_classification"
  },
  "parameters": {
    "preprocessing": [
      {"method": "baseline_removal", "order": 5},
      {"method": "normalization"}
    ],
    "model": {"n_estimators": 100, "max_depth": 15}
  },
  "performance": {
    "cv_accuracy": 0.95,
    "cv_f1": 0.94
  },
  "diagnostics": {...}
}
```

This is the foundation for production-grade FoodSpec workflows. 🚀

---

## Figure provenance
- Generated by [scripts/generate_docs_figures.py](https://github.com/chandrasekarnarayana/foodspec/blob/main/scripts/generate_docs_figures.py)
- Output: [../assets/figures/architecture_flow.png](../assets/figures/architecture_flow.png)

