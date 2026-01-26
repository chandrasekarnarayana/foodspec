# Workflow Run (Phase 3)

Phase 3 is the **canonical workflow entry point** for FoodSpec. It provides deterministic, auditable execution with strict regulatory compliance modes.

## Quick Start

```bash
# Minimal research workflow
foodspec workflow-run \
  --protocol examples/protocols/minimal_phase3.yaml \
  --input data.csv \
  --output-dir ./results

# Strict regulatory workflow
foodspec workflow-run \
  --protocol examples/protocols/minimal_phase3.yaml \
  --input data.csv \
  --output-dir ./results \
  --mode regulatory \
  --strict
```

## Workflow Modes

### Research Mode (default)
- **Intent**: Development, exploration, fast iteration
- **QC**: Advisory (informational, non-blocking)
- **Trust**: Optional (if enabled, can return placeholder)
- **Reporting**: Optional
- **Exit Code**: 0 on success, non-zero on error
- **Use case**: Developing new methods, testing hypotheses

```bash
foodspec workflow-run --protocol proto.yaml --input data.csv --output-dir out
```

### Regulatory Mode
- **Intent**: Pre-submission, validation workflows
- **QC**: Enforced (must pass or workflow fails)
- **Trust**: Mandatory (if non-placeholder, required)
- **Reporting**: Mandatory
- **Modeling**: Required
- **Model approval**: Must be in approved registry
- **Exit Code**: 0 on success, specific codes on failure
- **Use case**: Before regulatory submission

```bash
foodspec workflow-run --protocol proto.yaml --input data.csv --output-dir out --mode regulatory
```

### Strict Regulatory Mode
- **Intent**: Production submission, strict audit trail
- **QC**: Enforced (must pass)
- **Trust**: Mandatory, **must not be skipped** (returns success even if placeholder)
- **Reporting**: Mandatory, **must not be skipped**
- **Modeling**: Mandatory
- **Model approval**: Checked, must be approved
- **Artifact contract**: Versioned (v3), digest validated
- **Exit Code**: Specific codes guarantee artifact contract satisfaction
- **Use case**: Regulatory submissions, audit compliance

```bash
foodspec workflow-run --protocol proto.yaml --input data.csv --output-dir out --mode regulatory --strict
```

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | SUCCESS | All stages passed, artifact contract satisfied |
| 4 | ProtocolError | Protocol invalid or model not approved |
| 5 | ModelingError | Modeling fit/predict failed |
| 6 | TrustError | Trust stack failed or skipped in strict mode |
| 7 | QCError | QC gates failed in enforce mode |
| 8 | ReportingError | Report generation failed or skipped in strict mode |
| 9 | ArtifactError | Artifact contract incomplete (e.g., missing files) |

### Interpreting Exit Codes in Strict Regulatory

- **Exit 0**: Artifact contract v3 fully satisfied. All required files present. Audit trail complete.
- **Exit 4**: Model not in approved registry. Retry with approved model.
- **Exit 7**: QC gates failed. Review data quality in `artifacts/qc_results.json`.
- **Exit 6/8**: Trust or reporting forced but failed/skipped. Use research mode to debug.

## Output Artifact Tree

```
output_dir/
├── manifest.json                 # Execution fingerprint + contract version
├── success.json                  # Run summary (exit 0 only)
├── error.json                    # Error details (exit ≠0)
├── logs/
│   └── run.log                  # Structured log of all stages
├── artifacts/
│   ├── qc_results.json          # QC gate results: data_integrity, spectral_quality
│   ├── trust_stack.json         # Trust placeholders (calibration, conformal, abstention)
│   ├── metrics.json             # Model metrics (accuracy, precision, recall, F1)
│   ├── report.html              # HTML report (for artifact contract)
│   └── data_fingerprint.json    # SHA256 hashes of inputs
├── report/
│   └── index.html               # Browsable HTML report (same as artifacts/report.html)
└── tables/
    ├── preprocessed.csv         # After preprocessing stage
    ├── features.csv             # After feature extraction
    └── predictions.csv          # Model predictions (if modeling enabled)
```

## Interpreting QC Results

File: `artifacts/qc_results.json`

```json
{
  "passed": true,
  "gates": {
    "data_integrity": {
      "status": "pass",
      "metrics": {
        "row_count": 100,
        "missing_per_column": {"feature_1": 0.0, "feature_2": 0.01},
        "duplicate_rows": 0
      }
    },
    "spectral_quality": {
      "status": "pass",
      "metrics": {
        "outlier_fraction": 0.02,
        "snr_median": 12.5
      }
    }
  }
}
```

**In research mode**: "pass" is informational; workflow continues regardless.  
**In regulatory mode**: Any "fail" halts workflow with exit code 7.

## Interpreting Trust Stack

File: `artifacts/trust_stack.json`

In research mode, may contain `"status": "skipped"` (trust not implemented yet).

In strict regulatory mode, will contain `"status": "success"` with placeholders:

```json
{
  "status": "success",
  "reason": "Placeholder (real trust stack TBD, strict regulatory requires non-skipped result)",
  "coverage": 1.0,
  "calibration": {"status": "placeholder"},
  "conformal": {"status": "placeholder"},
  "abstention": {"status": "placeholder"}
}
```

**Note**: In production, this will contain real trust metrics (calibration error, conformal prediction sets, etc.). Current implementation is a placeholder to satisfy artifact contract.

## Manifest + Contract Versioning

File: `manifest.json` includes:

```json
{
  "artifact_contract_version": "v3",
  "artifact_contract_digest": "sha256_hex_...",
  "cli_args": {...},
  "seed": 42,
  "mode": "regulatory",
  "artifacts": {...}
}
```

The digest is computed from the sorted list of required artifacts for the contract version. This enables:
1. **Reproducibility**: Same protocol + data → same manifest + digest
2. **Audit compliance**: Changes to artifact list are recorded in manifest
3. **Version tracking**: Workflows reference which contract version they satisfy

## CLI Options

```bash
foodspec workflow-run \
  --protocol <path>              # Protocol YAML (required)
  --input <path> [<path>...]     # Input CSV files (required, repeatable)
  --output-dir <path>            # Output directory (required)
  --mode {research,regulatory}   # Default: research
  --strict                       # Strict regulatory enforcement (requires --mode regulatory)
  --model <name>                 # Override protocol model (e.g., LogisticRegression)
  --scheme <name>                # Override protocol CV scheme (e.g., kfold, lobo)
  --seed <int>                   # Random seed
  --enable-modeling              # Force modeling (default: config-driven)
  --enable-trust                 # Force trust stack (default: config-driven)
  --enable-reporting             # Force reporting (default: config-driven)
```

## Examples

### Example 1: Quick Research Run
```bash
foodspec workflow-run \
  --protocol examples/protocols/minimal_phase3.yaml \
  --input examples/data/small_dataset.csv \
  --output-dir ./research_run \
  --seed 42
```
**Expected**: exit 0, advisory QC, optional trust/reporting

### Example 2: Regulatory Pre-Check
```bash
foodspec workflow-run \
  --protocol examples/protocols/minimal_phase3.yaml \
  --input examples/data/validated_dataset.csv \
  --output-dir ./regulatory_run \
  --mode regulatory \
  --seed 42
```
**Expected**: exit 0 (if QC passes), enforced QC, mandatory trust/reporting

### Example 3: Strict Audit Trail
```bash
foodspec workflow-run \
  --protocol examples/protocols/minimal_phase3.yaml \
  --input examples/data/validated_dataset.csv \
  --output-dir ./audit_run \
  --mode regulatory \
  --strict \
  --seed 42
```
**Expected**: exit 0, strict contract v3 validation, versioned manifest, all artifacts present

### Example 4: Model Approval Check
```bash
foodspec workflow-run \
  --protocol examples/protocols/minimal_phase3.yaml \
  --input examples/data/validated_dataset.csv \
  --output-dir ./model_check \
  --mode regulatory \
  --model WeirdCustomModel \
  --strict
```
**Expected**: exit 4 (ProtocolError - model not in approved registry)

## Troubleshooting

### Exit Code 7: QC Gate Failed
```bash
# 1. Check QC results
cat output_dir/artifacts/qc_results.json

# 2. Review failed gates (data_integrity, spectral_quality, model_reliability)
# 3. Investigate data quality issues in the CSV

# 4. In research mode, QC is informational (doesn't block)
# 5. In regulatory mode, fix data or use research mode for debugging
```

### Exit Code 4: Model Not Approved
```bash
# Check approved models (from model_registry.py)
# Approved: LogisticRegression, RandomForest, SVC, GradientBoosting, ExtraTreesClassifier

# Use approved model:
foodspec workflow-run ... --model LogisticRegression
```

### Missing Artifacts
```bash
# Check manifest for which artifacts are required
cat output_dir/manifest.json | grep artifact_contract

# If missing file, check run.log for which stage failed
cat output_dir/logs/run.log | grep ERROR
```

## See Also

- [Protocol Configuration](api/core.md) - Protocol schema and options
- [Model Registry](../src/foodspec/workflow/model_registry.py) - Approved models, aliases
- [QC System](concepts/qc_system.md) - Gate details, thresholds
- [Trust & Uncertainty](concepts/trust_uncertainty.md) - Trust stack architecture

