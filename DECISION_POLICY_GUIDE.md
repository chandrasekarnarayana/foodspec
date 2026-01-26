"""Decision Policy and Operating Points: User Guide

This guide explains how to use FoodSpec's decision policy subsystem to convert ROC
diagnostics into explicit operating points for deployment and regulatory compliance.

## Overview

A decision policy is a specification for how to choose classification thresholds based on
business requirements, regulatory constraints, or operational objectives. FoodSpec's
decision policy system bridges ROC diagnostics (from compute_roc_diagnostics) and
deployment requirements, with audit-friendly outputs for compliance and governance.

## Quick Start

### Research Mode (Default: Youden Policy)

```python
from foodspec.trust import DecisionPolicy, choose_operating_point
from foodspec.modeling.api import fit_predict
from foodspec.modeling.diagnostics.roc import compute_roc_diagnostics

# Train model and get ROC diagnostics
result = fit_predict(X, y, model_name="logistic_regression", compute_roc=True)
roc_result = result.roc_diagnostics

# Choose operating point using Youden's J-statistic (balances TPR and TNR)
policy = DecisionPolicy(name="youden")
operating_point = choose_operating_point(y, y_proba, roc_result, policy)

print(f"Threshold: {operating_point.thresholds}")
print(f"Sensitivity: {operating_point.achieved_metrics['sensitivity']:.3f}")
print(f"Specificity: {operating_point.achieved_metrics['specificity']:.3f}")
print(operating_point.rationale)
```

### Regulatory Mode (Target Sensitivity = 95%)

```python
# Regulatory requirement: ensure ≥95% sensitivity (minimize false negatives)
policy = DecisionPolicy(
    name="target_sensitivity",
    params={"min_sensitivity": 0.95},
    regulatory_mode=True,
)
operating_point = choose_operating_point(y, y_proba, roc_result, policy)

# This guarantees achieved_metrics["sensitivity"] ≥ 0.95
print(f"Sensitivity: {operating_point.achieved_metrics['sensitivity']:.3f}")
print(f"Specificity: {operating_point.achieved_metrics['specificity']:.3f}")
```

## Available Policies

### 1. Youden (Research Default)

**When to use:** Exploratory analysis, when you don't have domain constraints.

**How it works:** Maximizes J = Sensitivity + Specificity - 1, balancing false positives
and false negatives equally.

```python
policy = DecisionPolicy(name="youden")
```

**Metrics:**
- Balanced sensitivity and specificity
- Good overall accuracy
- Equal cost for both error types assumed

---

### 2. Cost-Sensitive

**When to use:** When false positives and false negatives have different business costs.

**Example:** In fraud detection, a false positive costs $10 (manual review), a false
negative costs $500 (undetected fraud).

```python
policy = DecisionPolicy(
    name="cost_sensitive",
    params={
        "cost_fp": 10,    # Cost of false positive
        "cost_fn": 500,   # Cost of false negative
    },
)
operating_point = choose_operating_point(y, y_proba, roc_result, policy)
```

**How it works:** Searches thresholds to minimize expected cost:
```
Cost = cost_fp × FP + cost_fn × FN
```

**Metrics:**
- Minimized total business cost
- Sensitivity/specificity may be imbalanced
- Explicit cost model in audit trail

---

### 3. Target Sensitivity (Regulatory Standard)

**When to use:** Regulatory requirements mandate minimum sensitivity (e.g., medical screening,
safety-critical systems).

**Example:** Medical diagnostic device: "Must catch ≥95% of true positives"

```python
policy = DecisionPolicy(
    name="target_sensitivity",
    params={"min_sensitivity": 0.95},
    regulatory_mode=True,
)
operating_point = choose_operating_point(y, y_proba, roc_result, policy)

# Guaranteed: sensitivity ≥ 0.95
# Among thresholds meeting this, maximize specificity
```

**How it works:**
1. Identify all thresholds achieving sensitivity ≥ min_sensitivity
2. Among those, choose the one with highest specificity (lowest FPR)

**Metrics:**
- Sensitivity = exactly or exceeds minimum
- Specificity = maximized subject to sensitivity constraint
- Optimal for true positive-critical applications

---

### 4. Target Specificity

**When to use:** When false positives are extremely costly and must be minimized.

**Example:** Spam filter: "Minimize false positives (marking legitimate emails as spam)"

```python
policy = DecisionPolicy(
    name="target_specificity",
    params={"min_specificity": 0.99},
)
```

**How it works:**
1. Find all thresholds achieving specificity ≥ min_specificity
2. Among those, choose the one with highest sensitivity

---

### 5. Abstention-Aware (Selective Prediction)

**When to use:** Application allows deferring uncertain predictions (e.g., medical AI with
"send to expert review" option).

**Example:** Enable model to abstain on low-confidence predictions, subject to ≤10%
abstention rate.

```python
policy = DecisionPolicy(
    name="abstention_aware",
    params={"max_abstention_rate": 0.10},
)
operating_point = choose_operating_point(
    y, y_proba, roc_result, policy,
    abstention=abstention_result,  # Optional AbstentionResult
)

print(f"Effective abstention: {operating_point.uncertainty_metrics['abstention_rate']:.1%}")
```

**How it works:**
1. Grid search over thresholds
2. At each threshold, compute "effective utility" = coverage × accuracy
3. Keep only thresholds with abstention_rate ≤ max_abstention_rate
4. Choose threshold with highest utility

**Metrics:**
- Coverage = fraction of predictions made (1 - abstention_rate)
- Accuracy on predicted samples = high
- Useful for human-in-the-loop pipelines

---

## Integration with Trust Ecosystem

### With Conformal Prediction

```python
from foodspec.trust import MondrianConformalClassifier

# 1. Fit conformal predictor
cp = MondrianConformalClassifier(model, target_coverage=0.95)
cp.fit(X_cal, y_cal)
conformal_result = cp.predict_sets(X_test, bins=test_bins)

# 2. Choose operating point
operating_point = choose_operating_point(
    y_test, y_proba_test, roc_result,
    policy,
    conformal=conformal_result,  # Pass for uncertainty metrics
)

# Result includes coverage + set sizes in uncertainty_metrics
```

### With Calibration

```python
from foodspec.trust import IsotonicCalibrator

# 1. Calibrate probabilities
calibrator = IsotonicCalibrator()
calibrator.fit(X_cal, y_cal)
y_proba_calibrated = calibrator.predict(y_proba_test)

# 2. Choose operating point on calibrated probabilities
operating_point = choose_operating_point(
    y_test, y_proba_calibrated, roc_result,  # Use calibrated proba
    policy,
    calibration=y_proba_calibrated,  # Or pass separately
)
```

### With Abstention

```python
from foodspec.trust import evaluate_abstention

# 1. Evaluate abstention options
abstention_result = evaluate_abstention(y_proba_test, y_test, threshold=0.7)

# 2. Use abstention-aware policy to find optimal threshold
policy = DecisionPolicy(name="abstention_aware", params={"max_abstention_rate": 0.15})
operating_point = choose_operating_point(
    y_test, y_proba_test, roc_result,
    policy,
    abstention=abstention_result,
)
```

---

## Audit-Friendly Outputs

### Save Operating Point

```python
from foodspec.trust import save_operating_point
from pathlib import Path

artifacts = save_operating_point(Path("/output/decision_point"), operating_point)

# Creates:
# - decision_policy.json   (full operating point + rationale)
# - operating_point_thresholds.csv
# - operating_point_metrics.csv
```

### JSON Structure

```json
{
  "thresholds": 0.65,
  "policy": {
    "name": "target_sensitivity",
    "applies_to": "binary",
    "params": {"min_sensitivity": 0.95},
    "regulatory_mode": true
  },
  "achieved_metrics": {
    "sensitivity": 0.952,
    "specificity": 0.843,
    "ppv": 0.875,
    "npv": 0.934,
    "f1": 0.912,
    "balanced_accuracy": 0.898,
    "tp": 120, "fp": 18, "fn": 6, "tn": 106
  },
  "uncertainty_metrics": {},
  "rationale": "Target sensitivity policy enforces minimum sensitivity of 95.0%...",
  "warnings": [],
  "metadata": {
    "method": "target_sensitivity",
    "min_sensitivity": 0.95,
    "achieved_sensitivity": 0.952
  }
}
```

### Rationale (Human-Readable)

Operating points include one-paragraph rationales explaining the policy choice:

```
"Target sensitivity policy enforces minimum sensitivity of 95.0% and maximizes 
specificity. Selected threshold 0.6512. This policy is suitable for regulatory 
compliance and high-sensitivity requirements (e.g., medical screening where false 
negatives are costly)."
```

---

## Protocol-Driven Configuration

### YAML Protocol Example

```yaml
version: 2
model:
  type: logistic_regression
  
trust:
  decision_policy:
    name: target_sensitivity
    min_sensitivity: 0.95
    regulatory_mode: true
```

### Load and Execute

```python
from foodspec.protocol import ProtocolRunner

runner = ProtocolRunner.from_file("protocol.yaml")
result = runner.execute(X, y)

# result.operating_point contains selected thresholds
```

---

## CLI Integration (Future)

```bash
# Run with decision policy
foodspec run \
  --protocol classification.yaml \
  --input data.csv \
  --output-dir ./results \
  --decision-policy target_sensitivity \
  --min-sensitivity 0.95

# Or cost-sensitive
foodspec run \
  --protocol fraud_detection.yaml \
  --input transactions.csv \
  --decision-policy cost_sensitive \
  --cost-fp 10 \
  --cost-fn 500
```

---

## Best Practices

### ✅ Do's

1. **Choose policy based on application**, not data:
   - Medical: target_sensitivity
   - Fraud: cost_sensitive
   - Exploratory: youden

2. **Document assumptions**:
   - What costs/constraints did you use?
   - Were they validated with domain experts?
   - Are they still valid in production?

3. **Validate on held-out test set**:
   - Policy chosen on CV fold predictions
   - Always verify on completely held-out data

4. **Monitor in production**:
   - Does achieved sensitivity match expected?
   - Has data distribution shifted?
   - Are assumptions still valid?

### ❌ Don'ts

1. **Don't choose policy after seeing test results** (p-hacking)
   - Pre-specify policy before final evaluation

2. **Don't assume thresholds transfer across datasets**
   - Different data distribution → different optimal threshold
   - Retrain/recalibrate for new domains

3. **Don't ignore calibration**
   - Uncalibrated probabilities → wrong thresholds
   - Always calibrate on held-out calibration set

4. **Don't set impossible constraints**
   - If max AUC on test set is 0.85, can't achieve 0.99 sensitivity
   - FoodSpec will warn, but check warnings carefully

---

## Examples

### Example 1: Medical Diagnostic Device (Regulatory)

```python
# Requirement: ≥ 95% sensitivity (minimize missed diagnoses)
policy = DecisionPolicy(
    name="target_sensitivity",
    params={"min_sensitivity": 0.95},
    regulatory_mode=True,
)

operating_point = choose_operating_point(y_test, y_proba_test, roc_result, policy)

# Save for regulatory submission
save_operating_point(Path("regulatory_submission/"), operating_point)

# Result:
# Sensitivity: 96.2% (exceeds requirement)
# Specificity: 87.4% (maximized subject to sensitivity constraint)
# Rationale automatically generated
```

### Example 2: Fraud Detection (Cost-Sensitive)

```python
# Cost model:
# - False positive: $10 (manual review time)
# - False negative: $500 (undetected fraud loss)

policy = DecisionPolicy(
    name="cost_sensitive",
    params={"cost_fp": 10, "cost_fn": 500},
)

operating_point = choose_operating_point(y_test, y_proba_test, roc_result, policy)

# Result:
# Threshold: 0.32 (lower than Youden)
# Sensitivity: 89% (catches most fraud)
# Specificity: 78% (acceptable FP rate)
# Minimized expected cost per transaction
```

### Example 3: Research Analysis (Youden)

```python
# Initial exploration, no domain constraints

policy = DecisionPolicy(name="youden")
operating_point = choose_operating_point(y_test, y_proba_test, roc_result, policy)

# Result:
# Balanced sensitivity (88%) and specificity (88%)
# Good starting point for domain discussions
```

---

## Troubleshooting

### Q: Can't achieve target sensitivity?

**A:** Your ROC AUC is too low. Check:
- Feature quality
- Model complexity
- Data leakage
- Class imbalance

**Solution:** If achieved sensitivity < target, FoodSpec will warn and use best available
threshold. Consider model retraining.

### Q: Threshold seems too high/low?

**A:** Check:
- Is data calibrated? (Miscalibration shifts optimal threshold)
- Is threshold selection based on same data distribution?
- Did you switch datasets? (threshold doesn't transfer)

**Solution:** Always retrain/recalibrate for new data.

### Q: Why different thresholds in training vs validation?

**A:** Expected behavior due to data distribution shift. This is why operating points must
be chosen on held-out CV folds or test sets.

---

## API Reference

See `src/foodspec/trust/decision_policy.py` for full API documentation.

### Main Classes

- `DecisionPolicy`: Specifies policy name and parameters
- `OperatingPoint`: Result from choose_operating_point()
- `PolicyType`: Enum of available policies

### Main Functions

- `choose_operating_point()`: Select thresholds using policy
- `save_operating_point()`: Save to disk with audit trail

---

## References

- ROC/AUC: [ROC Curve Documentation](../docs/roc_integration_summary.md)
- Conformal Prediction: [Conformal Prediction Guide](./conformal_prediction.md)
- Calibration: [Probability Calibration Guide](./calibration.md)
- Abstention: [Selective Prediction Guide](./abstention.md)
