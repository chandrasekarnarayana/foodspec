# Workflows API Reference

!!! info "Module Purpose"
    High-level workflow helpers for oil authentication, heating quality monitoring, and QC protocols.

---

## Quick Navigation

| Module | Purpose | Typical Use |
|--------|---------|-------------|
| `foodspec.apps.oils` | Oil authentication workflows | Verify oil authenticity |
| `foodspec.apps.heating` | Heating quality workflows | Monitor thermal degradation |
| `foodspec.apps.qc` | Quality control workflows | Batch QC automation |
| `foodspec.apps.protocol_validation` | Protocol validation | Validate analysis protocols |

---

## Common Patterns

### Pattern 1: Oil Authentication

```python
from foodspec.apps.oils import run_oil_authentication
from foodspec.io import load_folder

# Load oil spectra
fs = load_folder('data/olive_oils/')

# Run authentication workflow
result = run_oil_authentication(
    fs,
    reference_class='extra_virgin',
    method='pls_da',
    validation='nested_cv'
)

print(f"Accuracy: {result.accuracy:.3f}")
print(f"Authenticated: {result.n_authenticated}/{len(fs)}")
```

### Pattern 2: Heating Quality Monitoring

```python
from foodspec.apps.heating import run_heating_quality_analysis

# Analyze heating degradation
result = run_heating_quality_analysis(
    fs,
    temperature_col='temp',
    time_col='time',
    quality_threshold=0.7
)

print(f"Quality score: {result.quality_score:.2f}")
print(f"Degradation rate: {result.degradation_rate:.4f}")
```

### Pattern 3: Batch QC

```python
from foodspec.apps.qc import run_batch_qc

# Run QC on production batch
qc_result = run_batch_qc(
    fs,
    reference_spectra=fs_reference,
    tolerance=0.05,
    critical_peaks=[1650, 1450, 2850]
)

if qc_result.passed:
    print("✓ Batch passed QC")
else:
    print(f"✗ Failed: {qc_result.failure_reasons}")
```

---

## Oil Authentication

::: foodspec.apps.oils
    options:
      show_source: false

**Key Functions:**
- `run_oil_authentication()` - Main workflow
- `detect_adulteration()` - Adulteration detection
- `classify_variety()` - Variety classification

---

## Heating Quality

::: foodspec.apps.heating
    options:
      show_source: false

**Key Functions:**
- `run_heating_quality_analysis()` - Main workflow
- `compute_degradation_markers()` - Degradation indicators
- `predict_shelf_life()` - Shelf life prediction

---

## Quality Control

::: foodspec.apps.qc
    options:
      show_source: false

**Key Functions:**
- `run_batch_qc()` - Batch quality check
- `validate_against_reference()` - Reference comparison
- `detect_outliers()` - Outlier detection

---

## Protocol Validation

::: foodspec.apps.protocol_validation
    options:
      show_source: false

**Key Functions:**
- `validate_protocol()` - Protocol validation
- `run_protocol_test()` - Run test protocol
- `generate_protocol_report()` - Create validation report

---

## Cross-References

**Related Modules:**
- [Core](core.md) - Data structures
- [Preprocessing](preprocessing.md) - Preprocessing pipelines
- [Chemometrics](chemometrics.md) - Classification/regression models

**Related Workflows:**
- [Oil Authentication Workflow](../workflows/authentication/oil_authentication.md)
- [Heating Quality Workflow](../workflows/quality-monitoring/heating_quality_monitoring.md)
- [QC Workflow](../workflows/batch_quality_control.md)
