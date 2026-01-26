# ML & Validation API

Model training, cross-validation, and hyperparameter optimization.

The `foodspec.ml` module provides tools for building, training, and validating machine learning models with rigorous cross-validation strategies.

## Outcome Types

FoodSpec supports `classification`, `regression`, and `count` outcomes through the `model_fit_predict` protocol step. Set the task in your protocol:

```yaml
task:
  outcome_type: regression
  target_column: moisture
steps:
  - type: model_fit_predict
    params:
      model: ridge
      scheme: kfold
```

The step auto-selects numeric feature columns (excluding the target) and stores metrics, CIs, and residual diagnostics as protocol artifacts.

## Multivariate Embeddings

Use optional dimensionality reduction before modeling or as a standalone protocol step. The `model_fit_predict` step accepts an `embedding` block that is applied fold-wise to avoid leakage:

```yaml
steps:
  - type: model_fit_predict
    params:
      model: logreg
      scheme: nested
      target_column: class
      embedding:
        method: pca
        params:
          n_components: 3
          random_state: 0
```

To emit artifacts directly, add the `multivariate_analysis` step:

```yaml
steps:
  - type: multivariate_analysis
    params:
      method: pca
      feature_columns: [f1, f2, f3]
      outlier_z: 3.5
```

Artifacts written under `tables/` include `multivariate_scores.csv`, `multivariate_loadings.csv`, `multivariate_summary.csv`, and `multivariate_qc.csv`; figures are stored under `figures/multivariate/` for reporting.

## Cross-Validation

### nested_cross_validate

::: foodspec.ml.nested_cv.nested_cross_validate
    options:
      show_source: false
      heading_level: 4

## Hyperparameter Tuning

### grid_search_classifier

::: foodspec.ml.hyperparameter_tuning.grid_search_classifier
    options:
      show_source: false
      heading_level: 4

### quick_tune_classifier

::: foodspec.ml.hyperparameter_tuning.quick_tune_classifier
    options:
      show_source: false
      heading_level: 4

## Multi-Modal Fusion

### late_fusion_concat

::: foodspec.ml.fusion.late_fusion_concat
    options:
      show_source: false
      heading_level: 4

### decision_fusion_vote

::: foodspec.ml.fusion.decision_fusion_vote
    options:
      show_source: false
      heading_level: 4

## See Also

- **[Diagnostics & Model Evaluation](./diagnostics.md)** - ROC/AUC analysis, threshold optimization
- **[Model Validation](../methods/validation/index.md)** - Validation strategies
- **[Cross-Validation Guide](../methods/validation/cross_validation_and_leakage.md)** - CV best practices
- **[Examples](../examples_gallery.md)** - ML workflows
