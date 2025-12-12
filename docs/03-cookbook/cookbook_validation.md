# Cookbook: Validation

- Default CV: 5-fold (safe for moderate datasets).
- Small datasets: reduce CV automatically; watch warnings in reports.
- Batch-aware / group-stratified CV for multi-instrument or grouped data.
- Nested CV for performance estimation; see docs/05-advanced-topics/validation_strategies.md.
# Cookbook – Validation recipes

Quick, practical recipes for reliable validation.

## Run batch-aware CV
**Problem:** Avoid leakage across batches/instruments.

- **CLI:** Use a protocol with `validation_strategy: batch_aware` or pass `--validation-strategy batch_aware` if supported.  
  ```bash
  foodspec-run-protocol --input my.csv --protocol examples/protocols/oil_basic.yaml --validation-strategy batch_aware
  ```
- **GUI:** Choose “batch-aware” in the validation strategy selector before running.  
- **Python:**  
  ```python
  from foodspec import validation
  metrics = validation.batch_aware_cv(X, y, batch_labels)
  ```
- **Outputs:** balanced accuracy, per-class recall; reported in bundle tables/report and validation pane.

## Nested CV
**Problem:** Need inner-loop feature selection/hyperparameter tuning without optimistic bias.

- **CLI:** Enable nested CV in the protocol or via flag (if exposed).  
  ```bash
  foodspec-run-protocol --input my.csv --protocol examples/protocols/oil_basic.yaml --validation-strategy nested
  ```
- **GUI:** Select “nested CV” if available in the validation dropdown.  
- **Python:**  
  ```python
  from foodspec import validation
  results = validation.nested_cv(X, y, outer_folds=5, inner_folds=3)
  ```
- **Outputs:** outer performance metrics; inner selection details; confusion/ROC figures if generated.

## Adjust CV folds for small samples
**Problem:** Few samples per class make standard k-fold unstable.

- **CLI:** FoodSpec auto-reduces folds when classes are tiny; you can also set `--cv-folds 3`.  
- **GUI:** Choose fewer folds in validation settings if exposed.  
- **Python:** pass a smaller `cv_folds` to validation functions.
- **Tip:** Watch QC warnings in the report if class counts are low; interpret metrics cautiously.

## Example figure
![CV metric distribution](../assets/figures/cv_boxplot.png)
