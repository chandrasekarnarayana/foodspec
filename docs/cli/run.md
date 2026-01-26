# FoodSpec `run` Command: Complete E2E Orchestration

The `foodspec run` command provides a complete end-to-end orchestration layer that produces one reproducible, self-contained artifact bundle per run. This guide covers modes, outputs, and usage.

## Overview

`foodspec run` wires together:
1. **Schema validation** – Ensures input data conforms to protocol spec
2. **Preprocessing** – Normalization, baseline correction, spike removal (protocol-driven)
3. **Feature engineering** – Extracts spectral features per protocol
4. **Group-safe modeling & validation** – Cross-validation with group awareness (LOBO, LOSO, nested)
5. **Trust stack** – Calibration, conformal prediction, abstention
6. **Visualizations** – Confusion matrices, ROC curves, feature importance
7. **HTML reporting** – Self-contained report with all results
8. **Manifest & summary** – Full reproducibility record

### One Run = One Complete Artifact Bundle

Every run creates a structured output directory containing:
```
run_<id>/
  manifest.json              # Full run metadata: versions, hashes, environment
  summary.json               # Deployment readiness scorecard
  data/
    preprocessed.csv         # Validated & processed data
  features/
    X.npy                    # Feature matrix
    y.npy                    # Target labels
  modeling/
    metrics.json             # CV fold metrics + aggregates
    model.pkl                # Trained model (optional)
  trust/
    trust_metrics.json       # Calibration ECE, conformal coverage, abstention rate
  figures/
    *.png, *.svg             # Plots (confusion matrix, ROC, etc.)
  tables/
    *.csv, *.parquet         # Detailed results per fold
  report/
    index.html               # Complete HTML report with embedded assets
```

## Modes

### Research (default)
**Goal:** Exploratory analysis, maximal debug outputs, interpretation focus.

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/exp1 \
  --mode research
```

**Behavior:**
- Includes all visualizations and detailed metrics
- Verbose logging
- Reports uncertainty & calibration metrics
- No enforcement of QC policies
- Allows experimental features

**Artifacts:** All available (figures, tables, uncertainty, limitations sections in report)

### Regulatory
**Goal:** Compliance-ready runs with audit trail, deterministic seeds, strict validation.

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/regulatory \
  --mode regulatory \
  --seed 0
```

**Behavior:**
- Enforces stricter QC policies
- Deterministic seed (default if not specified)
- Emphasizes reproducibility and traceability
- Includes confidence intervals on metrics
- Bootstrap resampling for robust estimates
- Audit trail of all configuration choices
- Stable naming conventions

**Artifacts:** Includes audit trail, confidence intervals, bootstrap distributions

### Monitoring
**Goal:** Drift detection, baseline comparison, ongoing surveillance.

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/new_batch.csv \
  --outdir runs/monitoring \
  --mode monitoring \
  --baseline runs/baseline_run/manifest.json
```

**Behavior:**
- Compares new data to baseline distribution
- Detects feature drift, covariate shift
- Minimal reporting (flagged changes only)
- Fast execution
- Hooks for change detection algorithms

**Note:** v1 includes structure; baseline comparison is minimal. v2 will add full drift detection.

**Artifacts:** Drift plots, change detection metrics, baseline comparison tables

## Validation Schemes

### LOBO (Leave-One-Batch-Out, default)
Assumes samples grouped by batch/instrument/collection-date.

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/exp1 \
  --scheme lobo
```

**Use when:** Instruments, collection batches, or temporal groups matter. Prevents information leakage.

### LOSO (Leave-One-Subject-Out)
Each subject (e.g., oil/plant sample) has multiple measurements; one subject held out per fold.

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/exp1 \
  --scheme loso
```

**Use when:** Each sample has multiple replicates and subject-level generalization is the goal.

### Nested CV
Outer loop stratified, inner loop for hyperparameter tuning.

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/exp1 \
  --scheme nested
```

**Use when:** Small data, need robust hyperparameter estimates, or competing models.

## Models

Override the default model specified in the protocol:

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/exp1 \
  --model svm
```

**Available models:**
- `lightgbm` – Gradient boosting (fast, interpretable)
- `svm` – Support Vector Machine (robust)
- `rf` – Random Forest (parallel, robust)
- `logreg` – Logistic Regression (linear baseline)
- `plsda` – PLS Discriminant Analysis (spectral native, interpretable)

Protocol config specifies the default; CLI `--model` overrides it.

## CLI Examples

### Classic Mode (Backward Compatible)
All existing scripts continue to work without changes:

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --normalization-mode reference \
  --baseline-method als \
  --outdir runs/classic
```

**Note:** Classic mode uses the legacy `ProtocolRunner`; outputs are in the old format.

### YOLO Mode (New E2E Orchestration)
Triggered by any of: `--model`, `--scheme`, `--mode`, `--no-trust`

```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/yolo \
  --model lightgbm \
  --scheme lobo \
  --mode research \
  --trust
```

**Produces:** New structured artifact bundle with manifest, summary, report.

### Minimal Command (YOLO defaults)
```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/minimal
```

**Defaults:**
- Mode: `research`
- Scheme: `lobo`
- Model: `lightgbm`
- Trust: enabled

### All Options
```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --input data/oils_test.csv \
  --outdir runs/full \
  --model svm \
  --scheme nested \
  --mode regulatory \
  --trust \
  --seed 42 \
  --verbose
```

## Output Artifact Contract

### `manifest.json`
Complete run metadata for reproducibility:

```json
{
  "protocol_hash": "abc123...",
  "protocol_snapshot": { "name": "Oils", "version": "1.0.0", "steps": [...] },
  "python_version": "3.11.0",
  "platform": "Linux 5.15.0",
  "dependencies": { "foodspec": "2.1.0", "scikit-learn": "1.3.0", ... },
  "seed": 42,
  "data_fingerprint": "def456...",
  "start_time": "2024-01-26T15:30:00Z",
  "end_time": "2024-01-26T15:32:15Z",
  "duration_seconds": 135.2,
  "artifacts": {
    "manifest": "manifest.json",
    "summary": "summary.json",
    "metrics": "modeling/metrics.json",
    "report": "report/index.html"
  },
  "validation_spec": {
    "scheme": "lobo",
    "mode": "research"
  },
  "warnings": [],
  "cache_hits": [],
  "cache_misses": []
}
```

### `summary.json`
Deployment readiness scorecard for decision-makers:

```json
{
  "dataset_summary": {
    "samples": 150,
    "classes": 3,
    "modality": "raman"
  },
  "scheme": "lobo",
  "model": "lightgbm",
  "mode": "research",
  "metrics": {
    "accuracy": 0.92,
    "f1_weighted": 0.91,
    "balanced_accuracy": 0.90
  },
  "calibration": {
    "ece": 0.032
  },
  "coverage": 0.95,
  "abstention_rate": 0.02,
  "deployment_readiness_score": 0.87,
  "deployment_ready": true,
  "key_risks": [
    "Feature drift on new instruments",
    "Class imbalance on rare oils",
    "Limited temperature robustness"
  ]
}
```

### `report/index.html`
Self-contained HTML report with embedded figures, tables, and interactive elements.

**Sections (mode-dependent):**
- **Summary** – Overview and key metrics
- **Dataset** – Input size, modality, splits
- **Methods** – Protocol configuration
- **Metrics** – Accuracy, F1, ROC, per-class stats
- **Calibration & Uncertainty** – ECE, conformal coverage, abstention
- **Readiness** – Deployment scorecard
- **Limitations** – Known risks and model assumptions

### `metrics.json` (under `modeling/`)
Detailed CV results:

```json
{
  "accuracy": 0.92,
  "f1_weighted": 0.91,
  "balanced_accuracy": 0.90,
  "folds": [
    {
      "fold": 0,
      "accuracy": 0.93,
      "f1_weighted": 0.92,
      "balanced_accuracy": 0.91
    },
    ...
  ],
  "confusion_matrix": [[..., ...], [..., ...]],
  "per_class_f1": [0.88, 0.95, 0.89]
}
```

## Exit Codes

- **0** – Success
- **2** – Validation error (bad input, schema mismatch)
- **3** – Runtime error (e.g., model training failed)
- **4** – Modeling/CV error (insufficient data, algorithm failure)

## Logging & Debugging

### Verbose Mode
```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/debug \
  --verbose
```

Outputs: execution trace, memory usage, CV fold details.

### Dry Run (schema validation only)
```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/validation \
  --dry-run
```

Validates protocol vs. data without executing steps.

## Use Cases

### Publication-Ready Analysis
```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils_discovery.csv \
  --outdir runs/paper \
  --mode research \
  --scheme nested \
  --model plsda \
  --seed 42 \
  --verbose
```

**Produces:** Reproducible results, publication figures, supplementary metrics.

### Regulatory Submission
```bash
foodspec run \
  --protocol examples/protocols/Oils_Regulated.yaml \
  --input data/oils_validation.csv \
  --outdir runs/fda_submission \
  --mode regulatory \
  --scheme lobo \
  --model svm \
  --seed 0
```

**Produces:** Audit trail, deterministic results, compliance-ready manifest, bootstrap CIs.

### Production Monitoring
```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/incoming_batch.csv \
  --outdir runs/monitoring/batch_2024_01_26 \
  --mode monitoring \
  --baseline runs/regulatory/baseline/manifest.json
```

**Produces:** Drift metrics, change flags, rapid feedback loop.

### Quick Exploration
```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/quick
```

**Produces:** All defaults (research mode, LOBO, LightGBM, trust enabled).

## Reproducibility

To exactly reproduce a prior run:

1. **From manifest:**
   ```bash
   # Extract seed and config from manifest.json
   export SEED=$(jq -r '.seed' runs/baseline/manifest.json)
   foodspec run \
     --protocol examples/protocols/Oils.yaml \
     --input data/oils.csv \
     --outdir runs/reproduction \
     --seed $SEED \
     --model lightgbm \
     --scheme lobo \
     --mode research
   ```

2. **Compare manifests:**
   ```bash
   # Both should have matching hashes and seeds
   jq '.protocol_hash, .seed, .data_fingerprint' runs/baseline/manifest.json
   jq '.protocol_hash, .seed, .data_fingerprint' runs/reproduction/manifest.json
   ```

## API Usage (Python)

For programmatic access:

```python
from pathlib import Path
from foodspec.experiment import Experiment, RunMode, ValidationScheme

# Create experiment from protocol dict or file
exp = Experiment.from_protocol(
    "examples/protocols/Oils.yaml",
    mode=RunMode.REGULATORY,
    scheme=ValidationScheme.LOBO,
    model="svm",
)

# Run on input data
result = exp.run(
    csv_path=Path("data/oils.csv"),
    outdir=Path("runs/api_run"),
    seed=0,
    verbose=True,
)

# Check results
print(f"Status: {result.status}")
print(f"Report: {result.report_dir / 'index.html'}")
print(f"Manifest: {result.manifest_path}")

# Access metrics
import json
summary = json.loads(result.summary_path.read_text())
print(f"Accuracy: {summary['metrics']['accuracy']}")
print(f"Deployment Ready: {summary['deployment_ready']}")
```

## Troubleshooting

### "Protocol not found"
```bash
# Protocol name must be in examples/protocols/ or a valid file path
ls examples/protocols/
foodspec run --protocol examples/protocols/ExistingProtocol.yaml ...
```

### "No inputs provided"
```bash
# Use --input or --input-dir
foodspec run --protocol ... --input data/file.csv ...
# OR
foodspec run --protocol ... --input-dir data/ --glob "*.csv" ...
```

### "Validation error"
```bash
# Check input CSV schema
foodspec run --protocol ... --input data/file.csv --dry-run --verbose
```

### "Model not available"
```bash
# Install additional dependencies
pip install scikit-learn xgboost lightgbm
```

### "Low accuracy / model seems broken"
Check:
1. Data preprocessing (normalization mode, spike removal)
2. Feature engineering pipeline (protocol-specific)
3. Class balance (use `--scheme loso` if imbalanced)
4. Sample size (CV needs ≥2 samples per class per fold)

## Advanced Configuration

### Custom Protocol with Preprocessing
See `examples/protocols/` for template YAML structure. Protocol can specify:
- Normalization method (snv, minmax, reference)
- Baseline correction (als, rubberband, none)
- Spike removal algorithm
- Feature extraction method
- Default model and CV strategy
- QC policy

Protocol config + CLI flags = merged final config (CLI takes precedence).

### Caching (Experimental)
```bash
foodspec run \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --outdir runs/cached \
  --cache
```

Caches preprocessed data and trained models per seed.

## Next Steps

- Explore [API documentation](../api/experiment.md) for programmatic usage
- Review [trust stack](../trust-uncertainty.md) for calibration and conformal methods
- See [workflows](../workflows/) for multi-protocol pipelines
- Check [deployment](../deployment.md) for production recommendations
