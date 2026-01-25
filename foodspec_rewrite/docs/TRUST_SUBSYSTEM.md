# Trust & Uncertainty Quantification Subsystem

## Overview

The Trust & Uncertainty subsystem provides **rigorous, leakage-safe** tools for quantifying prediction confidence, calibrating probabilities, and enabling human-machine collaboration through conformal prediction and principled abstention.

### Key Capabilities

- **Conformal Prediction**: Rigorous uncertainty sets with coverage guarantees
- **Mondrian Conditioning**: Group-aware coverage (by batch, instrument, stage)
- **Probability Calibration**: Temperature scaling, isotonic regression
- **Abstention Rules**: Principled rejection when confidence is low
- **Reproducibility**: Deterministic outputs with seed control
- **Integration**: Seamless ArtifactRegistry and RunManifest support

## Quick Start

### Basic Conformal Prediction

```python
from sklearn.linear_model import LogisticRegression
from foodspec.trust.conformal import MondrianConformalClassifier
import numpy as np

# Prepare data (must split: train, calibration, test)
X_train, y_train = ...  # Training set
X_cal, y_cal = ...      # Calibration set (disjoint from train)
X_test, y_test = ...    # Test set

# Train base model
model = LogisticRegression()
model.fit(X_train, y_train)

# Conformal prediction
cp = MondrianConformalClassifier(model, target_coverage=0.9)
cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)  # Fit conformal threshold on calibration set

# Predict uncertainty sets
result = cp.predict_sets(X_test, y_true=y_test)
print(f"Coverage: {result.coverage:.1%}")
print(f"Mean set size: {np.mean(result.set_sizes):.2f}")
```

### Probability Calibration

```python
from foodspec.trust.calibration import TemperatureScaler

# Fit temperature scaling on calibration set
calibrator = TemperatureScaler()
proba_cal = model.predict_proba(X_cal)
calibrator.fit(y_cal, proba_cal)

# Apply to predictions
proba_test = model.predict_proba(X_test)
proba_test_cal = calibrator.predict(proba_test)
```

### Abstention with Decision Rules

```python
from foodspec.trust.abstain import evaluate_abstention

# Get predictions and prediction sets
proba_test = model.predict_proba(X_test)
result = cp.predict_sets(X_test)

# Abstain on low confidence
abstain_result = evaluate_abstention(
    proba_test,
    y_test,
    threshold=0.7,  # Reject if max probability < 0.7
    prediction_sets=result.prediction_sets,
    max_set_size=1,  # Reject if set size > 1
)

print(f"Abstention rate: {abstain_result.abstention_rate:.1%}")
print(f"Accuracy (non-abstained): {abstain_result.accuracy_non_abstained:.1%}")
```

### Group-Aware Coverage

```python
# Specify batch/instrument/stage for conditional coverage
X_cal, y_cal, bins_cal = ...
X_test, y_test, bins_test = ...

cp.calibrate(X_cal, y_cal, bins=bins_cal)
result = cp.predict_sets(X_test, bins=bins_test, y_true=y_test)

# Per-bin coverage guarantee
for bin_id, coverage in result.per_bin_coverage.items():
    print(f"Bin {bin_id}: {coverage:.1%}")
```

### High-Level Evaluator Integration

```python
from foodspec.trust.evaluator import TrustEvaluator
from foodspec.core.artifacts import ArtifactRegistry

# Create evaluator
evaluator = TrustEvaluator(
    model,
    artifact_registry=ArtifactRegistry(Path("/output/run")),
    target_coverage=0.9,
    abstention_threshold=0.7,
    calibration_method="temperature",
)

# Fit on calibration set
evaluator.fit_conformal(X_cal, y_cal, bins_cal=bins_cal)
evaluator.fit_calibration(y_cal, model.predict_proba(X_cal))

# Comprehensive evaluation
result = evaluator.evaluate(
    X_test, y_test,
    bins_test=bins_test,
    model_name="my_model",
)

# Save all artifacts
artifacts = evaluator.save_artifacts(
    result,
    prediction_sets=result.prediction_sets,
    set_sizes=result.set_sizes,
    abstention_mask=abstain_result.abstain_mask,
    output_dir=Path("/output/run/trust"),
)

# Print report
print(evaluator.report(result))
```

## Core Concepts

### Conformal Prediction

Conformal prediction provides **calibrated uncertainty sets** with distribution-free coverage guarantees:

$$P(\text{true label} \in \hat{C}(x)) \geq 1 - \alpha$$

where $\alpha$ controls the target miscoverage and $\hat{C}(x)$ is the prediction set for sample $x$.

**Key Property**: Coverage is guaranteed for **any distribution** if the test set is exchangeable with the calibration set (true under i.i.d. assumptions).

### Mondrian Conditioning

Standard conformal prediction may yield imbalanced set sizes across groups. Mondrian (group-conditional) conformal prediction provides:

- **Per-group coverage guarantees** using group-specific thresholds
- Tighter prediction sets for well-calibrated groups
- Diagnostics for model fairness and group disparities

### Temperature Scaling

Rescale predicted probabilities to improve calibration:

$$\hat{p}_\text{cal}(y|x) = \text{softmax}(\log p(y|x) / T)$$

where $T$ (temperature) is fit on the calibration set to minimize NLL.

### Isotonic Regression

Fit monotonic (non-parametric) transformation:

$$\hat{p}_\text{cal} = \text{isotonic}(p)$$

More expressive than temperature scaling; handles diverse miscalibration patterns.

### Abstention Rules

Abstain (reject) prediction when:
1. **Low confidence**: $\max_y p(y|x) < \tau$
2. **Large prediction set**: $|\hat{C}(x)| > s_\max$
3. **Selective classification**: Combine both for efficiency-coverage tradeoffs

## API Reference

### Conformal Prediction

**`MondrianConformalClassifier`**

```python
class MondrianConformalClassifier:
    def __init__(model, target_coverage=0.9):
        """Initialize with fitted sklearn classifier."""
    
    def fit(X, y):
        """Verify model is fitted (X, y unused for pre-fitted models)."""
    
    def calibrate(X_cal, y_cal, bins=None):
        """Fit conformal threshold on calibration set."""
    
    def predict_sets(X_test, bins=None, y_true=None):
        """Return ConformalPredictionResult with uncertainty sets."""
```

**`ConformalPredictionResult`**

```python
@dataclass
class ConformalPredictionResult:
    prediction_sets: List[List[int]]       # Predicted label sets
    set_sizes: List[int]                  # Size of each set
    coverage: float                       # Empirical coverage
    per_bin_coverage: Dict[str, float]   # Coverage per bin
    nonconformity_scores: np.ndarray     # Conformal nonconformity
    threshold: float                     # Conformal threshold used
    target_coverage: float               # Target coverage level
```

### Calibration

**`TemperatureScaler`**

```python
class TemperatureScaler:
    def fit(y_true, proba, lr=0.01, n_epochs=1000):
        """Fit temperature on calibration set."""
    
    def predict(proba):
        """Apply temperature scaling."""
```

**`IsotonicCalibrator`**

```python
class IsotonicCalibrator:
    def fit(y_true, proba):
        """Fit isotonic calibration."""
    
    def predict(proba):
        """Apply per-class isotonic regression."""
```

**`expected_calibration_error(y_true, proba, n_bins=10)`**

Compute ECE with binning strategy.

### Abstention

**`evaluate_abstention(proba, y_true, threshold=0.7, prediction_sets=None, max_set_size=None)`**

```python
@dataclass
class AbstentionResult:
    abstain_mask: np.ndarray              # Abstention decisions
    predictions: np.ndarray               # Predicted labels
    accuracy_non_abstained: Optional[float]
    accuracy_abstained: Optional[float]
    abstention_rate: float
    coverage: float                       # Coverage on non-abstained
    confidence_scores: np.ndarray         # Max probabilities
```

### Evaluator Integration

**`TrustEvaluator`**

```python
class TrustEvaluator:
    def fit_conformal(X_cal, y_cal, bins_cal=None):
        """Fit conformal prediction."""
    
    def fit_calibration(y_cal, proba_cal):
        """Fit probability calibrator."""
    
    def evaluate(X_test, y_test, bins_test=None, batch_ids=None, ...):
        """Comprehensive evaluation → TrustEvaluationResult."""
    
    def save_artifacts(result, prediction_sets, set_sizes, abstention_mask, ...):
        """Save to ArtifactRegistry and disk."""
    
    def report(result):
        """Human-readable evaluation report."""
```

## Design Principles

### Leakage-Safe

- Conformal calibration fit **only** on calibration set disjoint from training
- No information flows from test set into thresholds
- Calibration and conformal steps are sequentially disjoint

### Group-Aware

- Mondrian conditioning computes per-group coverage and thresholds
- Batch/instrument/stage binning reveals fairness issues
- Per-group diagnostics in coverage tables

### Deterministic

- All randomness controlled by `random_state`
- No global state or implicit randomness
- Reproducible across runs and environments

### Model-Agnostic

- Works with **any** classifier with `predict_proba()`
- No retraining of base model needed
- Calibration and conformal are independent layers

### Artifact-Integrated

- All outputs saved to ArtifactRegistry
- Prediction sets, coverage tables, calibration params, abstention summaries
- Compatible with RunManifest for workflow tracking

## Examples

### Example 1: Oil Authentication

```python
# Train classifier on training set
model = LogisticRegression()
model.fit(X_train_auth, y_train_auth)

# Fit conformal on calibration set (from different batch)
cp = MondrianConformalClassifier(model, target_coverage=0.95)
cp.calibrate(X_cal_auth, y_cal_auth, bins=cal_batches)

# Evaluate on test set
result = cp.predict_sets(X_test_auth, bins=test_batches, y_true=y_test_auth)

# Abstain on low confidence for critical decisions
proba = model.predict_proba(X_test_auth)
abstain_result = evaluate_abstention(
    proba, y_test_auth,
    threshold=0.95,  # Very high threshold for authentication
    max_set_size=1,   # Allow only singleton sets
)

# Report per-batch coverage
for batch_id, cov in result.per_bin_coverage.items():
    print(f"Batch {batch_id}: {cov:.1%} coverage")
```

### Example 2: Heating Quality with Calibration

```python
# Fit temperature scaling on calibration set
calibrator = TemperatureScaler()
proba_cal = model.predict_proba(X_cal_heat)
calibrator.fit(y_cal_heat, proba_cal)

# Apply calibration + conformal
proba_test_raw = model.predict_proba(X_test_heat)
proba_test_cal = calibrator.predict(proba_test_raw)

# Conformal on calibrated probabilities
cp = MondrianConformalClassifier(model, target_coverage=0.90)
cp.calibrate(X_cal_heat, y_cal_heat)
result = cp.predict_sets(X_test_heat)

# ECE on calibrated predictions
ece = expected_calibration_error(y_test_heat, proba_test_cal)
print(f"ECE after calibration: {ece:.4f}")
```

### Example 3: Group-Aware Coverage Diagnosis

```python
# Identify fairness issues via per-group coverage
evaluator = TrustEvaluator(model, registry, target_coverage=0.9)
evaluator.fit_conformal(X_cal, y_cal, bins_cal=instrument_ids)
result = evaluator.evaluate(
    X_test, y_test,
    bins_test=test_instruments,
    batch_ids=test_batches,
    df_test=metadata_df,
    group_col="protocol",
)

# Check for coverage disparities
for group, metrics in result.group_metrics.items():
    coverage = metrics["coverage"]
    if coverage < 0.85:
        print(f"⚠️  Group {group}: low coverage {coverage:.1%}")
```

## Troubleshooting

### Issue: Coverage < target_coverage

**Causes:**
- Calibration set too small (n_cal < 100 recommended)
- Calibration set from different distribution than test
- Non-exchangeability violation (covariate shift)

**Solutions:**
- Increase calibration set size
- Ensure calibration and test are from same batch/protocol
- Check for temporal drift or batch effects

### Issue: Very large prediction sets

**Causes:**
- Model predictions highly uncertain
- Target coverage too high (0.99 requires large sets)
- Poor model calibration

**Solutions:**
- Lower target coverage to 0.90
- Improve model or get more calibration data
- Apply calibration before conformal

### Issue: Abstention rate too high

**Causes:**
- Threshold too conservative (try lowering)
- Model poorly calibrated

**Solutions:**
- Lower `abstention_threshold` (e.g., 0.7 → 0.6)
- Apply calibration first
- Check per-group metrics for disparity

## Testing

Run comprehensive test suite:

```bash
pytest tests/test_trust_subsystem.py -v
```

Key test categories:
- Conformal prediction coverage guarantee
- Mondrian binning correctness
- Calibration ECE reduction
- Abstention behavior
- Group-aware metrics
- Determinism with seeds
- Edge cases and error handling
- Integration with evaluator

## References

- **Conformal Prediction**: Vovk et al. (2005), *Algorithmic Learning in a Random World*
- **Mondrian CP**: Barber et al. (2023), *Conformal Prediction Under Covariate Shift*
- **Temperature Scaling**: Guo et al. (2017), *On Calibration of Modern Neural Networks*
- **Adaptive Conformal**: Angelopoulos & Bates (2021), *A Gentle Introduction to Conformal Prediction*

## Contributing

To extend the trust subsystem:

1. Add new calibration methods in `calibration.py`
2. Add new abstention rules in `abstain.py`
3. Extend `MondrianConformalClassifier` with new conditioning strategies
4. Add tests and update documentation

Follow the module patterns: type hints, docstrings, error handling, determinism with seeds.
