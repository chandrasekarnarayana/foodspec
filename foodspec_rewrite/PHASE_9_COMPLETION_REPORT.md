# Phase 9: ProtocolV2 Trust Integration — Implementation Complete

## Overview

Phase 9 successfully extends ProtocolV2 with comprehensive, validated trust configuration options. Users can now specify trust/uncertainty settings directly in protocol files with full validation and clear error messages.

## Changes Made

### 1. Extended `foodspec/core/protocol.py` with 6 New Spec Classes

#### CalibrationSpec
- **Field**: `method: str = "none"`
- **Validation**: Must be one of {"none", "platt", "isotonic"}
- **Method**: `validate_method()` raises ValueError with actionable message

#### ConformalSpec
- **Fields**:
  - `enabled: bool = False`
  - `alpha: float = 0.1` (validated: 0 < alpha ≤ 1)
  - `condition_key: Optional[str] = None`
- **Purpose**: Configure conformal prediction sets

#### AbstentionRuleSpec
- **Fields**:
  - `type: str` (required: "max_prob" | "conformal_size")
  - `threshold: Optional[float]` (for max_prob, 0 < t ≤ 1)
  - `max_size: Optional[int]` (for conformal_size, ≥ 1)
- **Validation**: `validate_rule()` ensures type-specific parameters present

#### AbstentionSpec
- **Fields**:
  - `enabled: bool = False`
  - `rules: List[AbstentionRuleSpec] = []`
  - `mode: str = "any"` ("any" or "all")
- **Validation**: `validate_mode()` checks valid modes

#### TrustInterpretabilitySpec
- **Fields**:
  - `enabled: bool = False`
  - `methods: List[str] = []`
- **Validation**: `validate_methods()` checks methods in {"coefficients", "permutation_importance", "marker_panels"}

#### TrustSpec (Main container)
- **Aggregates**: CalibrationSpec, ConformalSpec, AbstentionSpec, TrustInterpretabilitySpec
- **Validation**: `validate()` orchestrates all component validators

### 2. Integrated TrustSpec into ProtocolV2

```python
class ProtocolV2(BaseModel):
    # ... existing fields ...
    trust: TrustSpec = Field(default_factory=TrustSpec)
    # ... rest of fields ...
    
    def validate(self, ...):
        # ... existing validations ...
        self.trust.validate()  # Added validation call
```

### 3. Created Comprehensive Test Suite

**File**: `tests/test_protocol_trust.py`
**Total Tests**: 39 (all passing)

#### Test Breakdown:
| Class | Tests | Status |
|-------|-------|--------|
| TestCalibrationSpec | 4 | ✅ PASSED |
| TestConformalSpec | 4 | ✅ PASSED |
| TestAbstentionRuleSpec | 7 | ✅ PASSED |
| TestAbstentionSpec | 5 | ✅ PASSED |
| TestTrustInterpretabilitySpec | 5 | ✅ PASSED |
| TestTrustSpec | 6 | ✅ PASSED |
| TestTrustProtocolV2Integration | 6 | ✅ PASSED |
| TestTrustProtocolSerialization | 2 | ✅ PASSED |
| **TOTAL** | **39** | **✅ ALL PASSING** |

## Usage Examples

### Example 1: Basic Protocol with Trust

```python
from foodspec.core.protocol import ProtocolV2

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
        "calibration": {"method": "isotonic"},
        "conformal": {"enabled": True, "alpha": 0.1},
    },
)

protocol.validate()  # All trust config validated
print(protocol.trust.calibration.method)  # "isotonic"
print(protocol.trust.conformal.enabled)  # True
```

### Example 2: Full Trust Configuration

```python
protocol = ProtocolV2(
    version="2.0.0",
    data={"input": "data.csv", "modality": "raman", "label": "label"},
    task={"name": "classification", "objective": "classify"},
    trust={
        "calibration": {"method": "platt"},
        "conformal": {"enabled": True, "alpha": 0.05, "condition_key": "batch"},
        "abstention": {
            "enabled": True,
            "rules": [
                {"type": "max_prob", "threshold": 0.8},
                {"type": "conformal_size", "max_size": 3},
            ],
            "mode": "any",
        },
        "interpretability": {
            "enabled": True,
            "methods": ["coefficients", "permutation_importance"],
        },
    },
)

protocol.validate()  # All validations pass
```

### Example 3: YAML Protocol

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
  calibration:
    method: "isotonic"
  
  conformal:
    enabled: true
    alpha: 0.1
    condition_key: "batch"
  
  abstention:
    enabled: true
    rules:
      - type: "max_prob"
        threshold: 0.7
      - type: "conformal_size"
        max_size: 3
    mode: "any"
  
  interpretability:
    enabled: true
    methods: ["coefficients", "permutation_importance", "marker_panels"]
```

## Validation Features

### Type-Level Validation (Pydantic)
- Alpha: `Field(gt=0, le=1)` enforces 0 < alpha ≤ 1
- Threshold: `Field(gt=0, le=1)` enforces 0 < threshold ≤ 1
- Max size: `Field(ge=1)` enforces max_size ≥ 1

### Method-Level Validation
- `CalibrationSpec.validate_method()`: Checks {"none", "platt", "isotonic"}
- `AbstentionRuleSpec.validate_rule()`: Ensures type-specific params
- `AbstentionSpec.validate_mode()`: Checks {"any", "all"}
- `TrustInterpretabilitySpec.validate_methods()`: Checks valid methods
- `TrustSpec.validate()`: Orchestrates all validators

### Protocol-Level Validation
- `ProtocolV2.validate()` calls `self.trust.validate()`
- All errors caught at parse time with actionable messages
- No runtime surprises with invalid configurations

## Error Examples

### Invalid Calibration Method
```python
protocol = ProtocolV2(
    ...,
    trust={"calibration": {"method": "bayesian"}},
)
protocol.validate()
# ValueError: Invalid calibration method: bayesian. Must be one of {'none', 'platt', 'isotonic'}.
```

### Missing Required Parameter
```python
# Missing threshold for max_prob rule
trust={"abstention": {"rules": [{"type": "max_prob"}]}}
# ValueError: max_prob rule requires 'threshold' parameter in (0, 1]
```

### Invalid Interpretability Method
```python
trust={"interpretability": {"methods": ["shapley"]}}
# ValueError: Invalid interpretability method: shapley. Must be one of {'coefficients', 'permutation_importance', 'marker_panels'}.
```

## Default Behavior

All trust features default to **disabled/minimal** for zero overhead:

```python
trust = TrustSpec()  # All defaults

assert trust.calibration.method == "none"  # No calibration
assert trust.conformal.enabled is False  # No conformal
assert trust.abstention.enabled is False  # No abstention  
assert trust.interpretability.enabled is False  # No interpretability
```

This ensures backward compatibility and opt-in trust features.

## Backward Compatibility

- Legacy `uncertainty: UncertaintySpec` kept for compatibility
- New `trust: TrustSpec` is the modern way forward
- All existing tests pass without modification
- Protocol validation doesn't break existing protocols

## Test Results

```
========== 39 passed in 0.58s ==========

Full Test Coverage:
- CalibrationSpec: 4/4 ✅
- ConformalSpec: 4/4 ✅
- AbstentionRuleSpec: 7/7 ✅
- AbstentionSpec: 5/5 ✅
- TrustInterpretabilitySpec: 5/5 ✅
- TrustSpec: 6/6 ✅
- ProtocolV2 Integration: 6/6 ✅
- Serialization: 2/2 ✅

No regressions in existing tests ✅
```

## Files Modified/Created

### Modified
- `foodspec/core/protocol.py` (~300 lines added)
  - 6 new Spec classes
  - Integration with ProtocolV2
  - Updated validate() method

### Created
- `tests/test_protocol_trust.py` (~430 lines)
  - 8 test classes
  - 39 comprehensive tests
  - Full coverage of all specs and validation paths

- `PHASE_9_SUMMARY.md` (comprehensive documentation)

## Key Achievements

✅ **Complete trust configuration schema**
✅ **All options validated with actionable errors**
✅ **Explicit, sensible defaults**
✅ **Full test coverage (39 tests, 100% passing)**
✅ **Zero regressions in existing code**
✅ **Backward compatible**
✅ **YAML/JSON serializable**
✅ **Production-ready**

## Next Phase Opportunities

Phase 10 can now:

1. Read trust settings from protocol in orchestrator.run()
2. Pass trust config to evaluate_model_cv()
3. Execute trust-enhanced validation with:
   - Calibration (platt/isotonic)
   - Conformal prediction sets
   - Abstention/selective classification
   - Interpretability analysis
4. Save all trust artifacts with confidence settings match protocol

## Status: COMPLETE ✓

Phase 9 successfully makes trust configuration first-class in FoodSpec's protocol schema. The implementation is production-ready with comprehensive validation, clear error messages, and full test coverage.
