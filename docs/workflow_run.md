# Workflow Run (Phase 1/2/3)

The FoodSpec workflow system provides three orchestration phases with increasing regulatory rigor. Phase 3 is the **canonical entry point** for production workflows with strict regulatory compliance modes.

## Phase Overview

| Phase | QC | Trust | Reporting | Use Case |
|-------|-----|-------|-----------|----------|
| **Phase 1** | Advisory | Optional | Optional | Development, rapid prototyping |
| **Phase 2** | Enforced | Optional | Optional | QC validation, data quality checks |
| **Phase 3** | Enforced | Mandatory* | Mandatory* | Regulatory submissions, production |

*In strict regulatory mode

## Quick Start

```bash
# Phase 1: Minimal workflow (development)
python -m foodspec.cli.main workflow-run-strict \
  tests/fixtures/minimal_protocol_phase3.yaml \
  --input data.csv \
  --output-dir ./results \
  --phase 1

# Phase 2: QC enforcement
python -m foodspec.cli.main workflow-run-strict \
  tests/fixtures/minimal_protocol_phase3.yaml \
  --input data.csv \
  --output-dir ./results \
  --phase 2

# Phase 3: Full pipeline (default)
python -m foodspec.cli.main workflow-run-strict \
  tests/fixtures/minimal_protocol_phase3.yaml \
  --input data.csv \
  --output-dir ./results \
  --phase 3 \
  --allow-placeholder-trust  # Required for development until real trust implemented
```

## Placeholder Trust Governance (Phase 3)

**CRITICAL**: The trust stack is currently a **placeholder implementation**. It is NOT production-ready for regulatory submissions.

### Default Behavior (Strict Regulatory Mode)
- Placeholder trust is **REJECTED** by default
- Exit code: **6** (TrustError)
- Error message: "Placeholder trust stack not allowed in strict regulatory mode"

### Development Mode
- Use `--allow-placeholder-trust` flag to accept placeholder for development/testing
- Logs warning: "⚠️ Placeholder trust stack being used in strict regulatory mode"
- trust_stack.json includes: `"implementation": "placeholder"`, `"capabilities": []`

### Example Commands

```bash
# REJECTED (exit 6) - default behavior
python -m foodspec.cli.main workflow-run-strict \
  protocol.yaml \
  --input data.csv \
  --phase 3

# ACCEPTED (exit 0) - development only
python -m foodspec.cli.main workflow-run-strict \
  protocol.yaml \
  --input data.csv \
  --phase 3 \
  --allow-placeholder-trust
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
python -m foodspec.cli.main workflow-run-strict protocol.yaml --input data.csv --output-dir out --mode research
```

### Regulatory Mode
- **Intent**: Pre-submission, validation workflows
- **QC**: Enforced (must pass or workflow fails)
- **Trust**: Mandatory (placeholder rejected unless --allow-placeholder-trust)
- **Reporting**: Mandatory
- **Modeling**: Required
- **Model approval**: Must be in approved registry
- **Exit Code**: 0 on success, specific codes on failure (see below)
- **Use case**: Before regulatory submission

```bash
python -m foodspec.cli.main workflow-run-strict protocol.yaml --input data.csv --output-dir out --mode regulatory --phase 3 --allow-placeholder-trust
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

| Code | Meaning | What Failed | What To Do Next |
|------|---------|-------------|-----------------|
| **0** | SUCCESS | ✅ All stages passed | Artifact contract satisfied, proceed with review |
| **2** | ConfigError | Invalid configuration | Check WorkflowConfig parameters and protocol YAML |
| **4** | ProtocolError | Protocol/model invalid | Use approved model from registry or fix protocol syntax |
| **5** | ModelingError | Model fit/predict failed | Check data format, labels, feature matrix shape |
| **6** | TrustError | **Placeholder trust rejected** | Use `--allow-placeholder-trust` for development OR implement real trust stack |
| **7** | QCError | QC gates failed | Fix data quality issues (missing values, class imbalance, etc.) |
| **8** | ReportingError | Report generation failed | Check report template and data availability |
| **9** | ArtifactError | Required artifacts missing | Check artifact contract validation, may be internal error |

### Exit Code 6: Trust Error (Placeholder Governance)

**Most Common Cause**: Strict regulatory mode rejects placeholder trust by default.

```bash
# Problem: Exit 6 with "Placeholder trust stack not allowed"
python -m foodspec.cli.main workflow-run-strict protocol.yaml --input data.csv --phase 3
# Exit code: 6

# Solution 1: Development/Testing (allow placeholder)
python -m foodspec.cli.main workflow-run-strict protocol.yaml --input data.csv --phase 3 --allow-placeholder-trust
# Exit code: 0 (with warning log)

# Solution 2: Production (implement real trust)
# Implement calibration, conformal prediction, abstention mechanisms
# Then update _run_trust_stack_real() to return "implementation": "real"
```

### Interpreting Exit Codes in Strict Regulatory

- **Exit 0**: Artifact contract v3 fully satisfied. All required files present. Audit trail complete.
- **Exit 4**: Model not in approved registry. Retry with approved model from `src/foodspec/workflow/model_registry.py`.
- **Exit 6**: Trust implementation issue. Either allow placeholder (dev) or implement real trust (production).
- **Exit 7**: Data quality issue. Review QC report at `{run_dir}/artifacts/qc_results.json`.
- **Exit 8**: Reporting infrastructure issue. Check logs at `{run_dir}/logs/run.log`.
- **Exit 9**: Artifact contract violation. Check `{run_dir}/error.json` for details.
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
  --phase {1,2,3}                # Workflow phase (1=minimal, 2=QC, 3=full, default=3)
  --allow-placeholder-trust      # Allow placeholder trust in strict mode (dev only)
```

## Contract Digest Lock (Drift Prevention)

The artifact contract (`src/foodspec/workflow/contracts/contract_v3.json`) defines which artifacts are required for successful workflow completion. To prevent unintentional changes, the contract digest is locked in tests.

### Current Contract v3 Digest
```
61f345763075100e57f0ea0cbb9e098aabae15549aad43933a230ce1c4a9154f
```

### What is the Digest?
- SHA256 hash of all required artifact paths across contract sections
- Computed from sorted, deduplicated artifact keys
- Includes: required_always, required_qc, required_trust, required_reporting, required_modeling, etc.
- Does NOT include artifact descriptions (only keys)

### Test Protection
The test `test_contract_v3_digest_lock` in `tests/test_workflow_phase3_e2e_real.py` verifies:
```python
EXPECTED_DIGEST = "61f345763075100e57f0ea0cbb9e098aabae15549aad43933a230ce1c4a9154f"
assert current_digest == EXPECTED_DIGEST
```

### When to Update the Digest

**If test fails with "Contract digest mismatch":**

1. **Check if contract_v3.json was intentionally modified**
   - Did you add/remove required artifacts?
   - Did you change artifact paths?

2. **If intentional:**
   - Run: `python -c "from foodspec.workflow.artifact_contract import ArtifactContract; print(ArtifactContract.compute_digest(ArtifactContract._load_contract('v3')))"`
   - Copy new digest to test's `EXPECTED_DIGEST`
   - Commit with clear message explaining contract change

3. **If unintentional:**
   - Revert `contract_v3.json` to previous state
   - Re-run tests to verify digest matches

### Why This Matters
- Prevents accidental artifact requirement changes
- Ensures regulatory workflows have stable expectations
- Requires explicit developer acknowledgment for contract modifications

## Examples

### Example 1: Quick Research Run (Phase 1)
```bash
python -m foodspec.cli.main workflow-run-strict \
  tests/fixtures/minimal_protocol_phase3.yaml \
  --input data.csv \
  --output-dir ./research_run \
  --phase 1 \
  --seed 42
```
**Expected**: exit 0, advisory QC, minimal artifacts

### Example 2: QC Validation (Phase 2)
```bash
python -m foodspec.cli.main workflow-run-strict \
  tests/fixtures/minimal_protocol_phase3.yaml \
  --input data.csv \
  --output-dir ./qc_check \
  --phase 2
```
**Expected**: exit 0 if QC passes, exit 7 if QC fails

### Example 3: Full Pipeline Development (Phase 3 + Placeholder)
```bash
python -m foodspec.cli.main workflow-run-strict \
  tests/fixtures/minimal_protocol_phase3.yaml \
  --input data.csv \
  --output-dir ./dev_run \
  --phase 3 \
  --allow-placeholder-trust
```
**Expected**: exit 0, all artifacts created, warning about placeholder trust

### Example 4: Regulatory Pre-Check
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

