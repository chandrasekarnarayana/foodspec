# ML & Validation API

Model training, cross-validation, and hyperparameter optimization.

The `foodspec.ml` module provides tools for building, training, and validating machine learning models with rigorous cross-validation strategies.

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

- **[Model Validation](../methods/validation/index.md)** - Validation strategies
- **[Cross-Validation Guide](../methods/validation/cross_validation_and_leakage.md)** - CV best practices
- **[Examples](../examples_gallery.md)** - ML workflows
