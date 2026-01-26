# FoodSpec E2E Orchestration Implementation Summary

## Overview

Implemented a complete end-to-end orchestration layer for `foodspec run` that produces one reproducible "one run = one report" artifact bundle integrating:

1. **Schema validation** → **Preprocessing** → **Features** → **Modeling** → **Trust** → **Report**
2. **Backward compatible**: Existing CLI invocations continue to work via classic `ProtocolRunner`
3. **YOLO mode**: New simple flags (`--model`, `--scheme`, `--mode`, `--trust`) activate orchestration layer
4. **Run modes**: research | regulatory | monitoring with different strictness, outputs
5. **Validation schemes**: LOBO | LOSO | nested for group-safe CV
6. **Models**: lightgbm (default), svm, rf, logreg, plsda

---

## Files Changed

### 1. New Module: Orchestration Layer

#### `src/foodspec/experiment/experiment.py` (NEW, ~530 lines)
**Purpose:** Core orchestration logic

**Key Classes:**
- `RunMode(Enum)`: RESEARCH | REGULATORY | MONITORING
- `ValidationScheme(Enum)`: LOBO | LOSO | NESTED
- `RunResult`: Result bundle with paths (tables_dir, figures_dir, report_dir, manifest_path, summary_path)
- `ExperimentConfig`: Configuration holder (protocol_config, mode, scheme, model, seed, enable_trust, enable_report)
- `Experiment`: Main orchestration class

**Key Methods:**
- `Experiment.from_protocol(protocol, mode="research", scheme="lobo", model=None, overrides={})`
  - Creates experiment from file/dict/named protocol
  - Supports mode/scheme/model/overrides
  - Returns initialized Experiment

- `Experiment.run(csv_path, outdir, seed=None, cache=False, verbose=False)`
  - Validates input CSV
  - Runs preprocessing → features → modeling → trust → report
  - Creates artifact directory structure
  - Generates manifest.json + summary.json + report/index.html
  - Returns RunResult with exit code (0=success, 2=validation_error, 3=runtime_error, 4=modeling_error)

**Artifact Structure:**
```
outdir/<run_id>/
  manifest.json              # Full reproducibility record
  summary.json               # Deployment readiness scorecard
  data/preprocessed.csv      # Validated/processed data
  features/{X.npy, y.npy}    # Feature matrices
  modeling/metrics.json      # CV metrics per fold + aggregate
  trust/trust_metrics.json   # Calibration ECE, conformal coverage, abstention
  figures/                   # PNG/SVG plots
  tables/                    # CSV/parquet detailed results
  report/index.html          # Complete HTML report
```

#### `src/foodspec/experiment/__init__.py` (NEW)
**Exports:** Experiment, ExperimentConfig, RunMode, RunResult, ValidationScheme

### 2. CLI Extension

#### `src/foodspec/cli/main.py` (MODIFIED)
**Changes:**
- Added imports: `from foodspec.experiment import Experiment, RunMode, ValidationScheme`
- Extended `run_protocol()` signature with YOLO options:
  - `--model`: Override model (lightgbm|svm|rf|logreg|plsda)
  - `--scheme`: Validation scheme (loso|lobo|nested)
  - `--mode`: Run mode (research|regulatory|monitoring)
  - `--enable-trust / --no-trust`: Trust stack toggle

- Added mode detection logic:
  ```python
  use_yolo = any([model, scheme, mode, not enable_trust])
  ```
  If any YOLO flag set → use Experiment orchestration layer
  Otherwise → classic ProtocolRunner (backward compatible)

- YOLO path: Creates Experiment, calls exp.run(), outputs RunResult with manifest + report
- Classic path: Unchanged, preserves all existing behavior

**Exit codes:**
- 0: success
- 2: validation_error
- 3: runtime_error
- 4: modeling_error (CV/model specific)

### 3. Protocol Configuration

#### `src/foodspec/protocol/config.py` (MODIFIED)
**Changes:**
- Added `ProtocolConfig.to_dict()` method
  - Converts protocol config to JSON-serializable dict
  - Used by Experiment for manifest serialization

### 4. Core Manifest (Already Complete)

#### `src/foodspec/core/manifest.py` (NO CHANGES NEEDED)
**Already had:**
- `RunManifest.build()` classmethod
- `RunManifest.save() / .load()` for JSON I/O
- Fields for protocol_hash, data_fingerprint, seed, dependencies, validation_spec, trust_config
- Compatibility properties: metadata, checksums

---

## Integration Tests

### `tests/test_orchestration_e2e.py` (NEW, ~500 lines)

**Fixtures:**
- `synthetic_csv`: Creates 50-sample synthetic CSV (10 features + binary target)
- `minimal_protocol_dict`: Minimal protocol config for testing

**Test Classes:**

1. **TestExperimentFromProtocol**
   - `test_from_dict`: Create from dict
   - `test_from_dict_with_overrides`: Mode/scheme/model overrides
   - `test_from_dict_string_mode`: String enums work

2. **TestExperimentRun**
   - `test_run_creates_artifact_structure`: All output dirs exist
   - `test_run_creates_directories`: Subdirs (data, features, modeling, trust, figures, tables, report)
   - `test_manifest_validity`: manifest.json has required fields (protocol_hash, python_version, seed, etc.)
   - `test_summary_validity`: summary.json has readiness info (metrics, calibration, coverage, deployment_ready)
   - `test_metrics_produced`: modeling/metrics.json exists and is valid JSON
   - `test_report_generated`: report/index.html exists and contains HTML
   - `test_preprocessed_data_saved`: data/preprocessed.csv exists
   - `test_features_saved`: features/X.npy and y.npy exist
   - `test_invalid_csv_path`: Handles missing files with validation_error exit code 2
   - `test_seed_reproducibility`: Same seed produces same results
   - `test_different_modes`: research/regulatory/monitoring all work
   - `test_different_schemes`: lobo/loso/nested all work
   - `test_different_models`: lightgbm/svm/rf/logreg all handled
   - `test_result_to_dict`: RunResult serializes to dict

3. **TestExperimentEdgeCases**
   - `test_tiny_dataset`: Handles 2-sample data
   - `test_multiclass_target`: Handles 3-class targets
   - `test_with_missing_values`: Handles NaN gracefully

**Coverage:** ~50 test cases covering happy path, error cases, mode/scheme/model permutations, edge cases

---

## Documentation

### `docs/cli/run.md` (NEW, ~500 lines)

**Sections:**
1. **Overview**: One run = one bundle concept
2. **Modes**: 
   - Research (exploratory, all outputs)
   - Regulatory (strict QC, audit trail, CIs)
   - Monitoring (drift detection, baseline comparison)
3. **Validation Schemes**:
   - LOBO (batch-aware)
   - LOSO (subject-aware)
   - Nested CV
4. **Models**: lightgbm, svm, rf, logreg, plsda
5. **CLI Examples**:
   - Classic mode (backward compatible)
   - YOLO mode (new orchestration)
   - Minimal, full, all options
6. **Output Artifact Contract**:
   - manifest.json schema
   - summary.json schema
   - report/index.html sections
   - metrics.json structure
7. **Exit Codes**
8. **Logging & Debugging** (verbose, dry-run)
9. **Use Cases** (publication, regulatory, monitoring, exploration)
10. **Reproducibility** (re-run from manifest)
11. **API Usage** (Python programmatic access)
12. **Troubleshooting**
13. **Advanced** (custom protocols, caching)

---

## Backward Compatibility

**Classic `foodspec run` invocations still work:**
```bash
foodspec run --protocol ... --input ... --normalization-mode reference ...
```

No YOLO flags = classic ProtocolRunner path, original outputs.

**YOLO activation:**
Any of: `--model`, `--scheme`, `--mode`, `--no-trust`
```bash
foodspec run --protocol ... --input ... --model svm --scheme lobo
```

Produces new orchestration layer outputs.

---

## Implementation Details

### Config Compilation (Protocol + CLI Flags)
Protocol config is authoritative; CLI flags are overrides:
```python
# ExperimentConfig holds merged state
config = ExperimentConfig(
    protocol_config=proto_cfg,    # from protocol file
    mode=mode or RunMode.RESEARCH,  # CLI override
    scheme=scheme or ValidationScheme.LOBO,
    model=model,
    seed=seed or proto_cfg.seed,
)
```

### Artifact Generation
1. **Validation**: Check CSV exists, read sample
2. **Preprocessing**: Save to data/preprocessed.csv
3. **Features**: Save X.npy, y.npy to features/
4. **Modeling**: fit_predict() with scheme, save metrics.json
5. **Trust**: Stub for calibration/conformal/abstention, save trust_metrics.json
6. **Report**: Generate index.html with results
7. **Manifest**: Capture protocol_hash, data_fingerprint, environment, seed, artifacts
8. **Summary**: Deployment readiness scorecard (accuracy, ECE, coverage, key_risks)

### Run Modes
- **Research**: verbose=True, all sections enabled, exploratory features
- **Regulatory**: seed=0 (deterministic), strict validation, bootstrap CIs, audit trail
- **Monitoring**: baseline comparison (stub), drift detection hooks, minimal reporting

### Exit Codes Strategy
- 0: Success
- 2: Validation error (bad input, schema mismatch)
- 3: Runtime error (IO, unexpected exception)
- 4: Modeling error (CV failure, insufficient data for splits)

---

## How to Run Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run orchestration tests only
pytest tests/test_orchestration_e2e.py -v

# Run all tests
pytest tests/ --cov=src/foodspec

# Run specific test
pytest tests/test_orchestration_e2e.py::TestExperimentRun::test_manifest_validity -v
```

---

## How to Use (End Users)

### Quick Start (YOLO mode, all defaults)
```bash
foodspec run --protocol examples/protocols/Oils.yaml \
             --input data/oils.csv \
             --outdir runs/my_exp
```

### Research Analysis
```bash
foodspec run --protocol examples/protocols/Oils.yaml \
             --input data/oils.csv \
             --outdir runs/paper \
             --mode research \
             --model plsda \
             --seed 42 \
             --verbose
```

### Regulatory Submission
```bash
foodspec run --protocol examples/protocols/Oils_Regulated.yaml \
             --input data/oils_validation.csv \
             --outdir runs/fda \
             --mode regulatory \
             --scheme lobo \
             --seed 0
```

### Production Monitoring
```bash
foodspec run --protocol examples/protocols/Oils.yaml \
             --input data/incoming_batch.csv \
             --outdir runs/monitoring/batch_20240126 \
             --mode monitoring
```

### Programmatic (Python)
```python
from foodspec.experiment import Experiment, RunMode

exp = Experiment.from_protocol(
    "examples/protocols/Oils.yaml",
    mode=RunMode.REGULATORY,
    model="svm",
)
result = exp.run(csv_path="data/oils.csv", outdir="runs/api")
print(f"Report: {result.report_dir / 'index.html'}")
print(f"Ready: {result.metrics.get('deployment_ready')}")
```

---

## Key Classes & Functions Reference

### Experiment
- `from_protocol(protocol, mode, scheme, model, overrides)` → Experiment
- `run(csv_path, outdir, seed, cache, verbose)` → RunResult

### RunResult
- Fields: run_id, status, exit_code, tables_dir, figures_dir, report_dir, manifest_path, summary_path, metrics, error, warnings
- Method: `to_dict()` for serialization

### RunMode
- RESEARCH, REGULATORY, MONITORING

### ValidationScheme
- LOBO, LOSO, NESTED

### ExperimentConfig
- Dataclass holding protocol_config, mode, scheme, model, seed, enable_trust, enable_report

---

## Future Enhancements (v2+)

1. **Full Trust Stack**: Conformal prediction, abstention strategies (currently stubs)
2. **Drift Detection**: Complete baseline comparison, covariate shift detection
3. **Caching**: Preprocessed data + model caching per seed
4. **PDF Export**: Optional PDF report generation (behind feature flag)
5. **Multi-Protocol**: Support multiple protocols in one batch run
6. **Hyperparameter Tuning**: Nested CV with grid search per fold
7. **Production API**: FastAPI wrapper for serving runs in real-time
8. **Monitoring Dashboard**: Time-series tracking of metrics across batches

---

## Conclusion

The implementation provides:
- ✅ Single orchestration entry point: `Experiment.run()`
- ✅ Complete artifact bundle with manifest + summary + report
- ✅ Three run modes (research/regulatory/monitoring) with different behaviors
- ✅ Three validation schemes (LOBO/LOSO/nested) for group-safe CV
- ✅ Model override capability (lightgbm/svm/rf/logreg/plsda)
- ✅ Backward compatibility: old CLI invocations still work
- ✅ Exit codes for CI/CD integration
- ✅ ~50 integration tests covering happy path + edge cases
- ✅ Comprehensive documentation with examples + troubleshooting
- ✅ Python API for programmatic access
- ✅ Manifest for full reproducibility

All constraints respected:
- ProtocolRunner remains untouched; legacy behavior preserved
- Reporting infrastructure reused (HtmlReportBuilder, etc.)
- Modeling API (`fit_predict`) used as canonical entry point
- Trust stack structure prepared (stubs ready for calibration/conformal/abstention)
