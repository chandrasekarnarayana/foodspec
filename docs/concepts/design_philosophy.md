# Design Philosophy & Execution Engine

FoodSpec is governed by **seven formal design principles**, enforced at runtime. Every execution must satisfy ALL principles.

---

## Canonical Design Principles

### 1. TASK_FIRST

Tasks must be one of: `authentication`, `adulteration`, `monitoring`

```python
enforce_task_first({"task": "authentication"})  # ✓ Pass
enforce_task_first({"task": "unknown"})          # ✗ PhilosophyError
```

### 2. QC_FIRST

Quality control runs BEFORE modeling. Pass rate ≥ 50%, no critical failures.

```python
enforce_qc_first({"status": "pass", "pass_rate": 0.95})  # ✓ Pass
enforce_qc_first({"status": "fail", "pass_rate": 0.3})    # ✗ PhilosophyError
```

### 3. TRUST_FIRST

Trust stack outputs required: `calibration`, `conformal` (minimum)

```python
enforce_trust_first({
    "calibration": {"ece": 0.042},
    "conformal": {"coverage": 0.92}
})  # ✓ Pass
```

### 4. PROTOCOL_IS_SOURCE_OF_TRUTH

Protocol is immutable, hashable, and defines: task, modality, model, validation

```python
enforce_protocol_truth(protocol)  # Must have all required attributes
```

### 5. REPRODUCIBILITY_REQUIRED

All randomness deterministic: seed, environment, versions, data fingerprint captured

```python
set_global_seed(42)
report = generate_reproducibility_report(seed=42, csv_path="data.csv")
```

### 6. DUAL_API

Both CLI and programmatic APIs fully supported

```bash
# CLI
foodspec run --protocol proto.yaml --input data.csv --output-dir runs/exp

# Programmatic
engine = ExecutionEngine()
engine.validate_protocol(protocol)
engine.execute_pipeline()
```

### 7. REPORT_FIRST

Reports auto-generated for every run (HTML, JSON, Markdown)

```python
enforce_report_first(artifacts)  # Must have report_html, card_json, or card_markdown
```

---

## Execution Engine

The **ExecutionEngine** orchestrates all 12 execution steps:

```python
from foodspec.engine.orchestrator import ExecutionEngine

engine = ExecutionEngine(run_id="exp_001")

# Step 1: Validate protocol
engine.validate_protocol(protocol, protocol_dict)

# Step 2: Validate data
engine.validate_data(Path("data.csv"))

# Step 3: Setup reproducibility
engine.setup_reproducibility(seed=42)

# Step 4: Setup pipeline DAG
dag = engine.setup_pipeline()

# Steps 5-9: Execute pipeline stages (caller provides functions)
engine.register_stage_function("preprocess", preprocess_func)
engine.register_stage_function("qc", qc_func)
engine.register_stage_function("features", features_func)
engine.register_stage_function("model", model_func)
engine.register_stage_function("trust", trust_func)
engine.register_stage_function("visualization", viz_func)
engine.register_stage_function("reporting", report_func)

results = engine.execute_pipeline()

# Step 10: Register artifacts
engine.register_artifact("metrics", ArtifactType.METRICS, Path("metrics.json"))
engine.finalize_artifacts()

# Step 11: Generate manifest
engine.generate_manifest(Path("output/"))

# Step 12: Philosophy enforcement (automatic)
manifest = engine.get_manifest()
print(manifest.summary())
```

---

## Philosophy Enforcement

All principles enforced at 12 checkpoints:

```
1. Protocol validation        → enforce_task_first()
2. Protocol truth check       → enforce_protocol_truth()
3. Data validation            → fingerprint_csv()
4. Reproducibility setup      → capture_environment()
5. QC stage execution         → qc_func()
6. QC first check             → enforce_qc_first()
7. Model & trust execution    → model_func(), trust_func()
8. Trust first check          → enforce_trust_first()
9. Report generation          → report_func()
10. Reproducibility check     → enforce_reproducibility()
11. Report first check        → enforce_report_first()
12. Manifest generation       → validate_all_principles()
```

### Violations Raise Exceptions

```python
from foodspec.core.philosophy import PhilosophyError

try:
    engine.run(protocol, protocol_dict, csv_path, out_dir)
except PhilosophyError as e:
    print(f"Philosophy violation: {e}")
    # Example output:
    # Philosophy enforcement failed (2 violations):
    #   ✗ QC-First: Pass rate 0.3 below 50% threshold
    #   ✗ Report-First: No report artifacts generated
```

---

## Pipeline DAG

Each run has a **directed acyclic graph** of stages:

```
preprocess → qc → features → model → trust → visualization → reporting
```

Nodes are ordered by dependencies and executed in order:

```python
from foodspec.engine.dag import build_standard_pipeline

dag = build_standard_pipeline()

# Validate DAG
dag.validate()

# Get execution order
order = dag.get_execution_order()  # ['preprocess', 'qc', 'features', ...]

# Save visualization
dag.to_svg(Path("output/dag.svg"))
```

---

## Artifact Registry

All outputs tracked centrally:

```python
from foodspec.engine.artifacts import ArtifactType, get_registry

registry = get_registry()

# Register artifacts
registry.register("metrics", ArtifactType.METRICS, Path("metrics.json"))
registry.register("report", ArtifactType.REPORTS, Path("report.html"))

# Query registry
artifacts_by_type = registry.list_by_type()  # {'metrics': [...], 'reports': [...]}
total_size = registry.total_size()  # bytes

# Export registry
registry.to_json(Path("artifacts.json"))
```

---

## Run Manifest

Comprehensive metadata for every run:

```python
from foodspec.core.run_manifest import ManifestBuilder, RunStatus

builder = ManifestBuilder("run_001")

builder.set_protocol(ProtocolSnapshot(
    protocol_hash="abc123...",
    task="authentication",
    modality="Raman",
    model="SVM",
))

builder.set_data(DataSnapshot(
    data_fingerprint="def456...",
    row_count=100,
    column_count=5,
))

builder.set_environment(EnvironmentSnapshot(
    seed=42,
    python_version="3.9.0",
    os_name="Linux",
    machine="x86_64",
    cpu_count=4,
))

manifest = builder.build()
manifest.mark_success()
manifest.to_json(Path("run_manifest.json"))
```

**Manifest Contents**:
- Run metadata (ID, timestamps, status)
- Protocol snapshot (hash, config, task)
- Data snapshot (fingerprint, row/col counts)
- Environment snapshot (seed, versions, OS)
- DAG snapshot (node list, execution order)
- Artifact snapshot (count, types, size)
- Philosophy checks (all principle validations)

---

## Deterministic Execution

All randomness controlled:

```python
from foodspec.utils.determinism import (
    set_global_seed,
    capture_environment,
    capture_versions,
    fingerprint_csv,
    fingerprint_protocol,
)

# Set seed (controls numpy, random, sklearn)
set_global_seed(42)

# Capture reproducibility metadata
env = capture_environment()
vers = capture_versions()
data_hash = fingerprint_csv(Path("data.csv"))
proto_hash = fingerprint_protocol(protocol_dict)

# Same seed → same outputs (bitwise identical)
```

---

## Testing Philosophy

Unit tests for each principle:

```python
from foodspec.core.philosophy import enforce_task_first, PhilosophyError

def test_task_first_valid():
    enforce_task_first({"task": "authentication"})  # ✓

def test_task_first_invalid():
    with pytest.raises(PhilosophyError):
        enforce_task_first({"task": "unknown"})
```

---

## Benefits

### For Users
✅ Reproducible by default (same seed = exact same outputs)  
✅ Quality gates (QC blocks bad data automatically)  
✅ Uncertainty quantified (calibration & conformal outputs)  
✅ Full audit trail (manifest records everything)

### For Developers  
✅ Clear contracts (know what's required at each step)  
✅ Early validation (errors caught at protocol/data load)  
✅ Orchestration provided (ExecutionEngine handles coordination)  
✅ Extensibility (register custom stage functions)

### For Compliance
✅ Deterministic execution (reproducibility certified)  
✅ Protocol versioning (config hashes tracked)  
✅ Data provenance (fingerprints recorded)  
✅ Artifact tracking (all outputs registered)

---

## Reference

| Component | Module | Key Classes | Purpose |
| --- | --- | --- | --- |
| Philosophy | `foodspec.core.philosophy` | `DESIGN_PRINCIPLES`, `PhilosophyError` | Principle definitions & enforcement |
| Orchestrator | `foodspec.engine.orchestrator` | `ExecutionEngine` | 12-step orchestration |
| DAG | `foodspec.engine.dag` | `PipelineDAG`, `Node`, `NodeType` | Dependency graph & execution |
| Registry | `foodspec.engine.artifacts` | `ArtifactRegistry`, `Artifact` | Artifact tracking & provenance |
| Manifest | `foodspec.core.run_manifest` | `RunManifest`, `ManifestBuilder` | Comprehensive run metadata |
| Determinism | `foodspec.utils.determinism` | `set_global_seed`, fingerprinting | Reproducibility infrastructure |

---

## Feature map (mindmap → module)

| Mindmap node | Module path | Public API | CLI command | Artifacts |
| --- | --- | --- | --- | --- |
| Data Objects | `foodspec.data_objects` | `Spectrum`, `SpectraSet`, `SpectralDataset` | `foodspec io validate` | `protocol.yaml`, `run_summary.json` |
| Data Extraction | `foodspec.io` | `read_spectra`, `detect_format` | `foodspec io validate` | ingest logs |
| Programming Engine | `foodspec.engine` | preprocessing pipeline | `foodspec preprocess run` | preprocessing logs |
| Quality Control | `foodspec.qc` | QC engine, dataset checks | `foodspec qc spectral|dataset` | `qc_results.json` |
| Feature Engineering | `foodspec.features` | peaks, ratios, chemometrics | `foodspec features extract` | feature tables |
| Modeling & Validation | `foodspec.modeling` | model factories, validation | `foodspec train`, `foodspec evaluate` | `metrics/metrics.json`, `validation/folds.json` |
| Trust & Uncertainty | `foodspec.trust` | calibration, conformal, abstention | `foodspec trust fit|conformal|abstain` | `trust/calibration.json`, `trust/conformal.json`, `trust/abstention.json`, `trust/readiness.json`, `trust_outputs.json` |
| Visualization & Reporting | `foodspec.viz`, `foodspec.reporting` | plots, reports | `foodspec report` | HTML/PDF reports |
