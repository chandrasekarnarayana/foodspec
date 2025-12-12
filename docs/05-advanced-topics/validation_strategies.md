# Advanced Topic – Validation Strategies

FoodSpec includes batch/group-aware and nested cross-validation to improve robustness.

## Batch/group-aware CV
- Keeps all samples from a batch/instrument together in either train or test to avoid leakage.
- Stratifies by class when possible to maintain balance.
- Configured via protocols (`validation_strategy: batch_aware`) or CLI/GUI selectors.

## Nested CV
- Outer CV reports performance; inner CV performs feature selection/hyperparameter tuning.
- Useful when minimal panel selection or model tuning is needed without optimistic bias.
- Metrics from outer folds are reported; inner selections are logged.

## Metrics reported
- Balanced accuracy, per-class recall/sensitivity.
- Confusion matrix; ROC/PR curves when applicable.
- Warnings/QC flags when class counts are low or splits are under-powered.

## Where to configure
- `validation.py` provides `batch_aware_cv`, `nested_cv`, and split helpers.
- Protocols can specify `validation_strategy`; CLI flags may override; GUI provides a dropdown.

See `03-cookbook/cookbook_validation.md` for quick recipes and `07-theory-and-background/chemometrics_and_ml_basics.md` for background.
