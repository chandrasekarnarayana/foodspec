# Phase 9: ProtocolV2 Trust Schema Integration — COMPLETE

## Summary

Phase 9 extends ProtocolV2 with comprehensive trust configuration options, enabling users to specify calibration, conformal prediction, abstention, and interpretability settings directly in protocol files.

## Implementation Details

### 1. **New Spec Classes** (in `foodspec/core/protocol.py`)

#### `CalibrationSpec`
```python
method: str = "none"  # Options: "none" | "platt" | "isotonic"
```
- Validates calibration method is supported
- Defaults to "none" (no calibration)

#### `ConformalSpec`
```python
enabled: bool = False
alpha: float = 0.1  # Miscoverage rate (0 < alpha ≤ 1)
condition_key: Optional[str] = None  # For Mondrian stratification
```
- Alpha: target coverage = 1 - alpha (e.g., alpha=0.1 for 90% coverage)
- condition_key: optional metadata column for stratified coverage (e.g., "batch", "stage")

#### `AbstentionRuleSpec`
```python
type: str  # "max_prob" | "conformal_size"
threshold: Optional[float] = None  # For max_prob: (0, 1]
max_size: Optional[int] = None  # For conformal_size: >= 1
```
- Each rule type has required parameters
- Validates rule configuration automatically

#### `AbstentionSpec`
```python
enabled: bool = False
rules: List[AbstentionRuleSpec] = []
mode: str = "any"  # "any" (OR) | "all" (AND)
```
- Combines multiple abstention rules
- Mode determines how rules are combined

#### `TrustInterpretabilitySpec`
```python
enabled: bool = False
methods: List[str] = []  # Options: "coefficients" | "permutation_importance" | "marker_panels"
```
- Specifies which interpretability methods to use
- Multiple methods can be enabled simultaneously

#### `TrustSpec` (Main trust container)
```python
calibration: CalibrationSpec
conformal: ConformalSpec
abstention: AbstentionSpec
interpretability: TrustInterpretabilitySpec
```
- Orchestrates all trust-related configuration
- Single `validate()` method validates all sub-components

### 2. **Integration with ProtocolV2**

- Added `trust: TrustSpec = Field(default_factory=TrustSpec)` to ProtocolV2
- Updated `validate()` method to call `self.trust.validate()`
- All trust configuration is validated during protocol validation

### 3. **Backward Compatibility**

- Kept legacy `uncertainty: UncertaintySpec` for backward compatibility
- New `trust: TrustSpec` is the modern way to configure trust features
- All defaults ensure minimal/no trust features if not specified

## Example Usage

### YAML Protocol with Trust Configuration

```yaml
version: "2.0.0"

data:
  input: "data.csv"
  modality: "raman"
  label: "label"
  metadata_map:
    sample_id: "sample_id"
    modality: "modality"
    label: "label"
    batch: "batch_id"

task:
  name: "classification"
  objective: "classify"

trust:
  # Calibration settings
  calibration:
    method: "isotonic"  # Apply isotonic regression

  # Conformal prediction
  conformal:
    enabled: true
    alpha: 0.1  # Target 90% coverage
    condition_key: "batch"  # Stratify coverage by batch

  # Selective classification
  abstention:
    enabled: true
    rules:
      - type: "max_prob"
        threshold: 0.7
      - type: "conformal_size"
        max_size: 3
    mode: "any"  # Abstain if EITHER rule triggers

  # Interpretability
  interpretability:
    enabled: true
    methods: ["coefficients", "permutation_importance", "marker_panels"]
```

### Programmatic Usage

```python
from foodspec.core.protocol import ProtocolV2

# Create protocol with custom trust config
protocol = ProtocolV2(
    version="2.0.0",
    data={
        "input": "data.csv",
        "modality": "raman",
        "label": "label",
        "metadata_map": {
            "sample_id": "sample_id",
            "modality": "modality",
            "label": "label",
        },
    },
    task={"name": "classification", "objective": "classify"},
    trust={
        "calibration": {"method": "platt"},
        "conformal": {"enabled": True, "alpha": 0.05},
        "abstention": {
            "enabled": True,
            "rules": [
                {"type": "max_prob", "threshold": 0.8},
            ],
        },
        "interpretability": {
            "enabled": True,
            "methods": ["coefficients"],
        },
    },
)

# Validate (raises error if configuration invalid)
protocol.validate()

# Access trust config
print(protocol.trust.calibration.method)  # "platt"
print(protocol.trust.conformal.enabled)  # True
print(protocol.trust.conformal.alpha)  # 0.05
```

## Test Coverage

Created `tests/test_protocol_trust.py` with **39 comprehensive tests**:

### CalibrationSpec Tests (4 tests)
- ✅ Valid methods: none, platt, isotonic
- ✅ Default method: "none"
- ✅ Invalid method raises ValueError
- ✅ Actionable error messages

### ConformalSpec Tests (4 tests)
- ✅ Default configuration
- ✅ Enable with custom alpha
- ✅ Alpha boundary validation (0 < alpha ≤ 1)
- ✅ Optional condition_key

### AbstentionRuleSpec Tests (7 tests)
- ✅ Valid max_prob rule
- ✅ Valid conformal_size rule
- ✅ Missing required parameters caught
- ✅ Invalid rule type rejected
- ✅ Threshold boundaries (0 < t ≤ 1)
- ✅ Max size boundaries (≥ 1)

### AbstentionSpec Tests (5 tests)
- ✅ Default configuration
- ✅ Enable with rules
- ✅ Valid combination modes
- ✅ Invalid mode rejected
- ✅ Full validation with rules

### TrustInterpretabilitySpec Tests (5 tests)
- ✅ Default configuration
- ✅ All valid methods: coefficients, permutation_importance, marker_panels
- ✅ Multiple methods support
- ✅ Invalid method rejected
- ✅ Actionable error messages

### TrustSpec Tests (6 tests)
- ✅ Default configuration
- ✅ Full trust configuration
- ✅ Validates invalid calibration
- ✅ Validates invalid abstention rules
- ✅ Validates invalid mode
- ✅ Validates invalid methods

### ProtocolV2 Integration Tests (6 tests)
- ✅ Protocol has trust field with defaults
- ✅ Custom trust configuration
- ✅ Protocol.validate() calls trust.validate()
- ✅ Invalid conformal alpha caught
- ✅ Invalid abstention rule caught
- ✅ Invalid interpretability method caught

### Serialization Tests (2 tests)
- ✅ Protocol dump includes trust config
- ✅ Loading from dict preserves trust config

**Total: 39/39 tests PASSING ✓**

## Validation Features

All specs include automatic validation:

1. **Type-level validation** via Pydantic constraints
   - Alpha: 0 < alpha ≤ 1 (via `Field(gt=0, le=1)`)
   - Threshold: 0 < threshold ≤ 1 (via `Field(gt=0, le=1)`)
   - Max size: ≥ 1 (via `Field(ge=1)`)

2. **Method-level validation** via custom methods
   - `validate_method()`: Checks against valid set {"none", "platt", "isotonic"}
   - `validate_rule()`: Ensures required parameters present
   - `validate_mode()`: Checks against {"any", "all"}
   - `validate_methods()`: Validates interpretability methods

3. **Orchestrated validation** via TrustSpec.validate()
   - Calls all component validators
   - Ensures complete configuration is valid before use

4. **Protocol-level validation**
   - ProtocolV2.validate() calls self.trust.validate()
   - Fails fast with actionable error messages

## Default Behavior

When not specified, all trust features default to **disabled/minimal**:

```python
trust = TrustSpec()  # All defaults

# Results in:
assert trust.calibration.method == "none"  # No calibration
assert trust.conformal.enabled is False  # No conformal prediction
assert trust.abstention.enabled is False  # No abstention
assert trust.interpretability.enabled is False  # No interpretability methods
```

This ensures protocols are **opt-in** for trust features with zero overhead if not requested.

## Schema Example: Full Featured Protocol

```yaml
version: "2.0.0"

data:
  input: "oils.csv"
  modality: "raman"
  label: "classification"
  metadata_map:
    sample_id: "sample_id"
    modality: "modality"
    label: "classification"
    batch: "batch_id"
    instrument: "instrument_code"

task:
  name: "classification"
  objective: "classify"

model:
  estimator: "logreg"

validation:
  scheme: "leave_one_group_out"
  group_key: "batch"
  metrics: ["accuracy", "f1", "precision", "recall"]

# ====== TRUST CONFIGURATION ======
trust:
  # Step 1: Calibrate probabilities
  calibration:
    method: "isotonic"  # Better for multiclass

  # Step 2: Get prediction sets
  conformal:
    enabled: true
    alpha: 0.1  # 90% coverage target
    condition_key: "instrument"  # Per-instrument thresholds

  # Step 3: Selective classification
  abstention:
    enabled: true
    rules:
      # Rule 1: Abstain if max probability < 0.75
      - type: "max_prob"
        threshold: 0.75
      # Rule 2: Abstain if conformal set too large
      - type: "conformal_size"
        max_size: 2
    mode: "any"  # Abstain if ANY rule triggered

  # Step 4: Explain predictions
  interpretability:
    enabled: true
    methods:
      - "coefficients"  # Model weights
      - "permutation_importance"  # Feature importance
      - "marker_panels"  # Selected biomarkers
```

## Benefits

1. **Explicit Trust Configuration**: All trust settings in one place (protocol file)
2. **Validation at Parse Time**: Errors caught immediately, not at runtime
3. **Actionable Errors**: Clear messages guide users to valid values
4. **Sensible Defaults**: Works out-of-box with minimal trust overhead
5. **Composable**: Mix and match trust features as needed
6. **Serializable**: Round-trip through YAML/JSON preserving all settings

## Files Modified/Created

### Modified:
- `foodspec/core/protocol.py`: Added 7 new spec classes + integration

### Created:
- `tests/test_protocol_trust.py`: 39 comprehensive tests

## Status: READY FOR NEXT PHASE

Phase 9 provides complete trust configuration infrastructure. The next phase can:

1. Read trust settings from protocol during orchestrator.run()
2. Pass trust config to evaluate_model_cv()
3. Execute trust-enhanced validation with calibration, conformal, abstention
4. Save all trust artifacts with confidence these settings match the protocol

**Test Summary**: 39/39 tests passing, zero regressions ✓
