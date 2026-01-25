# Trust & Uncertainty Quantification Module

## Overview

The `foodspec.trust` module provides rigorous, production-ready tools for uncertainty quantification in FoodSpec analysis pipelines. It enables reliable abstention and human-machine collaboration through:

- **Conformal Prediction**: Distribution-free uncertainty sets with coverage guarantees
- **Mondrian Conditioning**: Group-aware coverage (by batch, instrument, protocol)
- **Probability Calibration**: Temperature scaling and isotonic regression
- **Abstention Rules**: Principled rejection when confidence is low
- **Comprehensive Evaluation**: Integration with ArtifactRegistry and high-level evaluators

## Directory Structure

```
foodspec/trust/
├── __init__.py              # Package exports
├── conformal.py             # Conformal prediction (Mondrian)
├── calibration.py           # Probability calibration methods
├── abstain.py               # Abstention utilities
└── evaluator.py             # High-level evaluator integration
```

## Quick Start

### 1. Basic Conformal Prediction

```python
from foodspec.trust.conformal import MondrianConformalClassifier
from sklearn.linear_model import LogisticRegression

# Assume: X_train, y_train (training set)
#         X_cal, y_cal (calibration set, must be disjoint)
#         X_test, y_test (test set)

# Train base model
model = LogisticRegression().fit(X_train, y_train)

# Conformal prediction
cp = MondrianConformalClassifier(model, target_coverage=0.9)
cp.fit(X_train, y_train)  # Verify model fitted
cp.calibrate(X_cal, y_cal)  # Fit on disjoint calibration set

# Predict uncertainty sets
result = cp.predict_sets(X_test, y_true=y_test)
print(f"Coverage: {result.coverage:.1%}")  # ~90%
print(f"Mean set size: {np.mean(result.set_sizes):.2f}")
```

### 2. Probability Calibration

```python
from foodspec.trust.calibration import TemperatureScaler

# Fit temperature scaling on calibration set
calibrator = TemperatureScaler()
proba_cal = model.predict_proba(X_cal)
calibrator.fit(y_cal, proba_cal)

# Apply to test predictions
proba_test = model.predict_proba(X_test)
proba_test_calibrated = calibrator.predict(proba_test)
```

### 3. Abstention with Decision Rules

```python
from foodspec.trust.abstain import evaluate_abstention

# Reject when confidence < 0.7
abstain_result = evaluate_abstention(
    proba_test_calibrated,
    y_test,
    threshold=0.7,
)

print(f"Abstention rate: {abstain_result.abstention_rate:.1%}")
print(f"Accuracy (non-abstained): {abstain_result.accuracy_non_abstained:.1%}")
```

### 4. High-Level Evaluator

```python
from foodspec.trust.evaluator import TrustEvaluator
from foodspec.core.artifacts import ArtifactRegistry

# Create evaluator
evaluator = TrustEvaluator(
    model,
    artifact_registry=ArtifactRegistry(Path("/run")),
    target_coverage=0.9,
    abstention_threshold=0.7,
    calibration_method="temperature",
)

# Fit on calibration set
evaluator.fit_conformal(X_cal, y_cal)
evaluator.fit_calibration(y_cal, model.predict_proba(X_cal))

# Comprehensive evaluation
result = evaluator.evaluate(X_test, y_test, model_name="my_model")

# Save artifacts and print report
evaluator.save_artifacts(result, ...)
print(evaluator.report(result))
```

## Core Components

### Conformal Prediction (`conformal.py`)

**`MondrianConformalClassifier`**

Generates prediction sets $\hat{C}(x)$ with guaranteed coverage:

$$P(y \in \hat{C}(x)) \geq 1 - \alpha$$

Key features:
- Distribution-free: works for any classifier with `predict_proba()`
- Leakage-safe: calibration only on disjoint calibration set
- Mondrian conditioning: per-group coverage guarantees
- Deterministic: reproducible with seed control

```python
cp = MondrianConformalClassifier(model, target_coverage=0.9)
cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal, bins=batch_ids_cal)
result = cp.predict_sets(X_test, bins=batch_ids_test, y_true=y_test)
```

### Probability Calibration (`calibration.py`)

**`TemperatureScaler`**: Post-hoc scaling without retraining
```python
calibrator = TemperatureScaler()
calibrator.fit(y_cal, proba_cal)
proba_cal = calibrator.predict(proba_raw)
```

**`IsotonicCalibrator`**: Non-parametric calibration
```python
calibrator = IsotonicCalibrator()
calibrator.fit(y_cal, proba_cal)
proba_cal = calibrator.predict(proba_raw)
```

**`expected_calibration_error()`**: Quality metric
```python
ece = expected_calibration_error(y_test, proba_test)  # 0 = perfect
```

### Abstention (`abstain.py`)

**`evaluate_abstention()`**: Principled rejection rules

```python
result = evaluate_abstention(
    proba,
    y_true,
    threshold=0.7,  # Reject if max prob < 0.7
    prediction_sets=sets,  # Or reject if |set| > max_size
    max_set_size=1,
)

# Results
result.abstain_mask  # Which samples rejected
result.abstention_rate  # Fraction rejected
result.accuracy_non_abstained  # Accuracy on accepted samples
result.coverage  # Coverage on accepted samples
```

### High-Level Evaluator (`evaluator.py`)

**`TrustEvaluator`**: Unified interface

```python
evaluator = TrustEvaluator(
    model,
    artifact_registry,
    target_coverage=0.9,
    abstention_threshold=0.7,
    calibration_method="temperature",  # or "isotonic", None
)

# Workflow
evaluator.fit_conformal(X_cal, y_cal, bins_cal=batches)
evaluator.fit_calibration(y_cal, proba_cal)
result = evaluator.evaluate(X_test, y_test, bins_test=batches)

# Save and report
evaluator.save_artifacts(result, ...)
print(evaluator.report(result))
```

**`TrustEvaluationResult`**: Aggregated metrics

```
conformal_coverage              # P(y ∈ C)
conformal_set_size_*            # Set size statistics
ece                             # Calibration error
abstention_rate                 # Fraction rejected
accuracy_non_abstained          # Accuracy if accepted
coverage_under_abstention       # Coverage after rejection
efficiency_gain                 # Coverage / baseline
per_bin_coverage                # Coverage by group
group_metrics                   # Per-group statistics
```

## Design Principles

### Leakage-Safe
- Conformal calibration **only** on disjoint calibration set
- Calibration and prediction phases strictly separated
- No information leakage from test to thresholds

### Group-Aware
- Mondrian conditioning for per-group coverage guarantees
- Per-batch/instrument/stage coverage diagnostics
- Fairness analysis through group metrics

### Deterministic
- All randomness controlled by `random_state`
- No global state or implicit dependencies
- Reproducible across runs

### Model-Agnostic
- Works with any classifier with `predict_proba()`
- No model retraining needed
- Post-hoc layers (calibration, conformal, abstention)

### Artifact-Integrated
- All outputs saved to ArtifactRegistry
- Seamless integration with RunManifest
- CSV exports for manual inspection

## Real-World Examples

### Oil Authentication (High-Stakes)
```python
# Very high coverage and confidence threshold for authentication
evaluator = TrustEvaluator(
    model,
    registry,
    target_coverage=0.95,      # 95% coverage guarantee
    abstention_threshold=0.95, # Reject unless very confident
)
```

### Heating Quality (Exploratory)
```python
# Looser bounds for exploratory analysis
evaluator = TrustEvaluator(
    model,
    registry,
    target_coverage=0.80,      # 80% coverage
    abstention_threshold=0.60, # Moderate confidence
)
```

### Multi-Protocol Study
```python
# Group-aware evaluation across protocols
evaluator.fit_conformal(X_cal, y_cal, bins_cal=protocol_ids)
result = evaluator.evaluate(
    X_test, y_test,
    bins_test=protocol_ids,
    batch_ids=protocol_ids,  # Group metrics by protocol
    df_test=metadata,
    group_col="protocol",
)

# Check per-protocol coverage
for proto, metrics in result.group_metrics.items():
    print(f"Protocol {proto}: {metrics['coverage']:.1%}")
```

## Testing

Run comprehensive test suite:

```bash
# All trust tests
pytest tests/test_trust_subsystem.py -v
pytest tests/test_trust_integration.py -v

# Specific test class
pytest tests/test_trust_subsystem.py::TestMondrianConformal -v

# With coverage
pytest tests/test_trust*.py --cov=foodspec.trust --cov-report=html
```

Key test categories:
- Coverage guarantees (Mondrian vs global)
- Calibration methods and ECE
- Abstention rules and efficiency
- Group-aware metrics
- Determinism and reproducibility
- Edge cases and error handling
- End-to-end workflows

## Troubleshooting

### Coverage < Target

**Problem**: Achieved coverage below target

**Causes**:
- Calibration set too small (n_cal < 100)
- Distribution mismatch (different batch than test)
- Non-exchangeability (covariate shift)

**Solutions**:
- Increase calibration set size
- Ensure calibration and test from same source
- Check for batch effects or temporal drift

### Large Prediction Sets

**Problem**: Set sizes larger than expected

**Causes**:
- Model highly uncertain
- Target coverage too high (0.99 requires large sets)
- Poor calibration

**Solutions**:
- Lower target_coverage (0.9 instead of 0.99)
- Improve model or calibration data
- Apply calibration before conformal

### High Abstention Rate

**Problem**: Rejecting too many samples

**Causes**:
- Threshold too conservative
- Poorly calibrated model

**Solutions**:
- Lower threshold (0.7 → 0.6)
- Apply calibration
- Check per-group metrics for disparities

## API Reference

See [TRUST_SUBSYSTEM.md](../TRUST_SUBSYSTEM.md) for detailed API documentation.

## References

- **Conformal Prediction**: Vovk et al. (2005)
- **Mondrian CP**: Barber et al. (2023)
- **Calibration**: Guo et al. (2017)
- **Adaptive Conformal**: Angelopoulos & Bates (2021)

## Contributing

To extend the trust module:

1. Add new calibration methods in `calibration.py`
2. Add new conditioning strategies in `conformal.py`
3. Add new abstention rules in `abstain.py`
4. Add comprehensive tests
5. Update documentation

Follow module standards:
- Type hints on all functions
- Docstrings (Google style)
- Error handling with actionable messages
- Determinism (no global state, seed control)
- Max 600 lines per file

## License

FoodSpec Trust & Uncertainty Quantification subsystem is part of FoodSpec v2.
See LICENSE file for details.
