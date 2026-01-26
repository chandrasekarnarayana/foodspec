# FoodSpec Execution Engine

The **ExecutionEngine** is FoodSpec's central orchestrator that governs all runs with formal design principles enforced at runtime.

---

## Overview

Every FoodSpec run passes through a **12-step orchestration pipeline**:

```
1. Protocol Validation      → TASK_FIRST + PROTOCOL_TRUTH checks
2. Data Validation          → Data fingerprinting + statistics
3. Reproducibility Setup    → Global seed + environment capture
4. Pipeline DAG Setup       → Dependency graph creation
5. Preprocessing Stage      → Data transformation
6. QC Stage                 → QC_FIRST enforcement
7. Feature Stage            → Feature extraction
8. Model Stage              → Training + validation
9. Trust Stage              → TRUST_FIRST enforcement
10. Visualization Stage     → Figure generation
11. Reporting Stage         → REPORT_FIRST enforcement
12. Manifest Generation     → Complete provenance record
```

Each step is **mandatory** and **validated**. Failures propagate with clear error messages.

---

## Architecture

### Components

```
ExecutionEngine (orchestrator.py)
    ├── PhilosophyEnforcement (philosophy.py)
    ├── PipelineDAG (dag.py)
    ├── ArtifactRegistry (artifacts.py)
    ├── RunManifest (run_manifest.py)
    └── Determinism (determinism.py)
```

### Data Flow

```
Protocol + CSV
    ↓
ExecutionEngine
    ↓
[Validate] → [Setup] → [Execute DAG] → [Register] → [Manifest]
    ↓
Output Directory
    ├── metrics.csv
    ├── predictions.csv
    ├── qc_report.json
    ├── trust/
    ├── plots/
    ├── report.html
    ├── artifacts.json
    └── run_manifest.json
```

---

## Usage

### Basic Example

```python
from pathlib import Path
from foodspec.engine.orchestrator import ExecutionEngine
from foodspec.engine.artifacts import ArtifactType

# 1. Create engine
engine = ExecutionEngine(run_id="exp_001")

# 2. Validate protocol
protocol = ProtocolConfig.from_file("protocol.yaml")
protocol_dict = protocol.to_dict()
engine.validate_protocol(protocol, protocol_dict)

# 3. Validate data
engine.validate_data(Path("data.csv"))

# 4. Setup reproducibility
engine.setup_reproducibility(seed=42)

# 5. Setup pipeline
dag = engine.setup_pipeline()

# 6. Register stage functions
def preprocess_stage(context, **params):
    # Your preprocessing logic
    return {"X_processed": processed_data}

def qc_stage(context, **params):
    # Your QC logic
    return {"qc_results": {"status": "pass", "pass_rate": 0.95}}

# ... register all stages ...
engine.register_stage_function("preprocess", preprocess_stage)
engine.register_stage_function("qc", qc_stage)

# 7. Execute pipeline
results = engine.execute_pipeline()

# 8. Register artifacts
engine.register_artifact(
    "metrics",
    ArtifactType.METRICS,
    Path("output/metrics.csv"),
    source_node="model",
)

# 9. Finalize
engine.finalize_artifacts()
manifest_path = engine.generate_manifest(Path("output/"))

print(f"✓ Run complete: {manifest_path}")
```

---

## Pipeline DAG

### Standard Pipeline

```python
from foodspec.engine.dag import build_standard_pipeline

dag = build_standard_pipeline()

# Node structure:
# preprocess → qc → features → model → trust → visualization → reporting
```

### Custom Pipeline

```python
from foodspec.engine.dag import PipelineDAG, NodeType

dag = PipelineDAG()

# Add nodes
dag.add_node("load", NodeType.PREPROCESS, inputs=[], outputs=["data"])
dag.add_node("clean", NodeType.QC, inputs=["load"], outputs=["clean_data"])
dag.add_node("extract", NodeType.FEATURES, inputs=["clean"], outputs=["features"])
dag.add_node("train", NodeType.MODEL, inputs=["extract"], outputs=["model"])

# Validate
dag.validate()

# Execute
results = dag.execute(context={})

# Visualize
dag.to_svg(Path("pipeline.svg"))
```

### Topological Ordering

The DAG ensures nodes execute in dependency order:

```python
dag = PipelineDAG()
dag.add_node("c", NodeType.MODEL, inputs=["b"])
dag.add_node("a", NodeType.PREPROCESS, inputs=[])
dag.add_node("b", NodeType.QC, inputs=["a"])

order = dag.topological_sort()
# Result: ["a", "b", "c"] (automatically sorted)
```

---

## Artifact Registry

### Registration

```python
from foodspec.engine.artifacts import ArtifactRegistry, ArtifactType

registry = ArtifactRegistry()

# Register artifacts
registry.register(
    name="metrics",
    artifact_type=ArtifactType.METRICS,
    path=Path("output/metrics.csv"),
    description="Cross-validation metrics",
    source_node="model",
    metadata={"cv_folds": 5},
)

registry.register(
    name="confusion_matrix",
    artifact_type=ArtifactType.PLOTS,
    path=Path("output/confusion.png"),
    source_node="visualization",
)
```

### Querying

```python
# Resolve by name
artifact = registry.resolve("metrics")
print(artifact.path)  # PosixPath('output/metrics.csv')

# List by type
plots = registry.resolve_by_type(ArtifactType.PLOTS)
for plot in plots:
    print(plot.name, plot.path)

# Summary statistics
summary = registry.summary()
print(summary["total_artifacts"])  # 2
print(summary["count_by_type"])    # {'metrics': 1, 'plots': 1}
```

### Export

```python
# Save registry to JSON
registry.to_json(Path("output/artifacts.json"))

# Export manifest
registry.export_manifest(Path("output/artifact_manifest.json"))
```

---

## Run Manifest

### Structure

```json
{
  "metadata": {
    "run_id": "exp_001",
    "timestamp_start": "2026-01-26T10:30:00.000Z",
    "timestamp_end": "2026-01-26T10:35:42.123Z",
    "status": "success",
    "foodspec_version": "2.0.0"
  },
  "protocol": {
    "protocol_hash": "abc123def456...",
    "task": "authentication",
    "modality": "Raman",
    "model": "SVM",
    "validation": "cross-validation"
  },
  "data": {
    "data_fingerprint": "789xyz...",
    "row_count": 150,
    "column_count": 1024,
    "size_bytes": 614400
  },
  "environment": {
    "seed": 42,
    "python_version": "3.9.0",
    "os_name": "Linux",
    "cpu_count": 8,
    "package_versions": {
      "numpy": "1.24.0",
      "scikit-learn": "1.3.0"
    }
  },
  "dag": {
    "node_count": 7,
    "execution_order": ["preprocess", "qc", "features", "model", "trust", "visualization", "reporting"]
  },
  "artifacts": {
    "artifact_count": 12,
    "by_type": {"metrics": 1, "plots": 5, "reports": 1},
    "total_size_bytes": 5242880
  },
  "philosophy_checks": {
    "task_first": true,
    "qc_first": true,
    "trust_first": true,
    "reproducibility": true,
    "report_first": true
  }
}
```

### Building Manifests

```python
from foodspec.core.run_manifest import (
    ManifestBuilder,
    ProtocolSnapshot,
    DataSnapshot,
    EnvironmentSnapshot,
)

builder = ManifestBuilder("exp_001")

# Add snapshots
builder.set_protocol(ProtocolSnapshot(
    protocol_hash="abc123",
    task="authentication",
))

builder.set_data(DataSnapshot(
    data_fingerprint="xyz789",
    row_count=100,
))

builder.set_environment(EnvironmentSnapshot(
    seed=42,
    python_version="3.9.0",
    os_name="Linux",
    os_version="5.10",
    machine="x86_64",
    cpu_count=4,
))

# Build and save
manifest = builder.build()
manifest.mark_success()
manifest.to_json(Path("run_manifest.json"))

# Print summary
print(manifest.summary())
```

---

## Deterministic Execution

### Global Seed

```python
from foodspec.utils.determinism import set_global_seed, get_global_seed

# Set seed (affects numpy, random, sklearn)
set_global_seed(42)

# Verify
assert get_global_seed() == 42

# Same seed → identical outputs
```

### Environment Capture

```python
from foodspec.utils.determinism import capture_environment, capture_versions

# Capture OS, Python, machine info
env = capture_environment()
print(env["os"]["name"])          # "Linux"
print(env["python"]["version"])   # "3.9.0"
print(env["machine"]["cpu_count"]) # 8

# Capture package versions
versions = capture_versions()
print(versions["critical_packages"]["numpy"])  # "1.24.0"
```

### Fingerprinting

```python
from foodspec.utils.determinism import fingerprint_csv, fingerprint_protocol

# CSV fingerprint (SHA256)
data_hash = fingerprint_csv(Path("data.csv"))
print(data_hash)  # "a3f2d9b8c1e..."

# Protocol fingerprint (SHA256 of JSON)
protocol_dict = {"task": "auth", "model": "svm"}
proto_hash = fingerprint_protocol(protocol_dict)
print(proto_hash)  # "7d4e2f1a..."
```

---

## Philosophy Enforcement

### Automatic Validation

Philosophy checks run automatically at specific checkpoints:

```python
engine = ExecutionEngine()

# Checkpoint 1: Protocol validation
engine.validate_protocol(protocol, protocol_dict)  # ✓ TASK_FIRST checked

# Checkpoint 2: QC validation
qc_results = {"status": "pass", "pass_rate": 0.95}
# ✓ QC_FIRST enforced after QC stage

# Checkpoint 3: Trust validation
trust_outputs = {"calibration": {}, "conformal": {}}
# ✓ TRUST_FIRST enforced after trust stage

# Checkpoint 4: Reproducibility
# ✓ REPRODUCIBILITY_REQUIRED enforced before manifest generation

# Checkpoint 5: Reports
# ✓ REPORT_FIRST enforced before completion
```

### Manual Validation

```python
from foodspec.core.philosophy import validate_all_principles

validate_all_principles(
    config=protocol_dict,
    protocol=protocol,
    qc_results=qc_results,
    trust_outputs=trust_outputs,
    manifest=manifest.to_dict(),
    artifacts=artifact_dict,
)
# Raises PhilosophyError if any principle violated
```

---

## Error Handling

### Philosophy Violations

```python
from foodspec.core.philosophy import PhilosophyError

try:
    engine.validate_protocol(protocol, {"task": "invalid"})
except PhilosophyError as e:
    print(f"Philosophy error: {e}")
    # Output: Task 'invalid' not in TASK_FIRST: ['authentication', 'adulteration', 'monitoring']
```

### DAG Cycles

```python
dag = PipelineDAG()
dag.add_node("a", NodeType.PREPROCESS, inputs=["b"])
dag.add_node("b", NodeType.QC, inputs=["a"])

try:
    dag.topological_sort()
except ValueError as e:
    print(f"DAG error: {e}")
    # Output: Cycle detected involving node 'a'
```

### Missing Artifacts

```python
try:
    artifact = registry.resolve("nonexistent")
    if artifact is None:
        print("Artifact not found")
except KeyError:
    print("Registry lookup failed")
```

---

## Output Structure

Standard run output directory:

```
output/
├── run_manifest.json          # Comprehensive metadata
├── artifacts.json             # Artifact registry
├── dag.json                   # Pipeline DAG
├── dag.svg                    # DAG visualization (if graphviz available)
│
├── data/
│   ├── X_processed.npy        # Preprocessed data
│   └── features.csv           # Extracted features
│
├── qc/
│   ├── qc_report.json         # QC results
│   └── qc_summary.json        # QC summary
│
├── metrics/
│   ├── metrics.csv            # Cross-validation metrics
│   └── predictions.csv        # Model predictions
│
├── trust/
│   ├── calibration.json       # Calibration metrics
│   ├── conformal.json         # Conformal predictions
│   ├── abstention.json        # Abstention rates
│   └── drift.json             # Drift detection
│
├── plots/
│   ├── confusion_matrix.png
│   ├── calibration_curve.png
│   ├── coverage_efficiency.png
│   └── abstention_by_class.png
│
└── reports/
    ├── report.html            # Full HTML report
    ├── card.json              # Experiment card (JSON)
    └── card.md                # Experiment card (Markdown)
```

---

## Integration with CLI

The execution engine integrates seamlessly with the CLI:

```bash
# CLI run (uses ExecutionEngine internally)
foodspec run \
    --protocol protocol.yaml \
    --input data.csv \
    --output-dir runs/exp_001 \
    --seed 42 \
    --report

# Generates:
# runs/exp_001/run_manifest.json
# runs/exp_001/artifacts.json
# runs/exp_001/report.html
# ... all outputs ...
```

---

## Testing

### Unit Tests

```bash
# Test philosophy enforcement
pytest tests/engine/test_execution_engine.py::TestPhilosophyEnforcement -v

# Test DAG functionality
pytest tests/engine/test_execution_engine.py::TestPipelineDAG -v

# Test artifact registry
pytest tests/engine/test_execution_engine.py::TestArtifactRegistry -v

# Test determinism
pytest tests/engine/test_execution_engine.py::TestDeterministicExecution -v
```

### Integration Tests

```bash
# End-to-end orchestration
pytest tests/engine/test_execution_engine.py::TestEndToEnd -v
```

---

## Performance

### Overhead

- **Protocol validation**: ~10-50 ms
- **DAG setup**: ~5-20 ms
- **Environment capture**: ~50-100 ms
- **Fingerprinting**: ~100-500 ms (CSV size dependent)
- **Manifest generation**: ~10-50 ms

**Total overhead**: ~200-700 ms per run (negligible for ML workflows)

### Scalability

- DAG supports 100+ nodes efficiently
- Artifact registry handles 1000+ artifacts
- Manifest size: ~10-100 KB

---

## Best Practices

### 1. Always Set Seed

```python
engine.setup_reproducibility(seed=42)  # Explicit is better
```

### 2. Register All Artifacts

```python
# Register everything produced
engine.register_artifact("metrics", ArtifactType.METRICS, path)
engine.register_artifact("plots", ArtifactType.PLOTS, path)
```

### 3. Validate Early

```python
# Validate protocol BEFORE loading data
engine.validate_protocol(protocol, protocol_dict)

# Catch errors early
engine.validate_data(csv_path)
```

### 4. Use Standard Pipeline

```python
# Start with standard DAG
dag = build_standard_pipeline()

# Customize if needed
dag.add_node("custom_stage", NodeType.FEATURES, inputs=["features"])
```

### 5. Check Philosophy

```python
# Explicitly check philosophy before finalization
validate_all_principles(config, protocol, qc, trust, manifest, artifacts)
```

---

## References

| Component | File | Key Classes |
| --- | --- | --- |
| Orchestrator | `src/foodspec/engine/orchestrator.py` | `ExecutionEngine` |
| Philosophy | `src/foodspec/core/philosophy.py` | `DESIGN_PRINCIPLES`, `PhilosophyError` |
| DAG | `src/foodspec/engine/dag.py` | `PipelineDAG`, `Node` |
| Registry | `src/foodspec/engine/artifacts.py` | `ArtifactRegistry`, `Artifact` |
| Manifest | `src/foodspec/core/run_manifest.py` | `RunManifest`, `ManifestBuilder` |
| Determinism | `src/foodspec/utils/determinism.py` | `set_global_seed`, fingerprinting |
| Tests | `tests/engine/test_execution_engine.py` | 35+ test methods |
| Docs | `docs/concepts/design_philosophy.md` | Philosophy documentation |

---

## See Also

- [Design Philosophy](design_philosophy.md) - Formal principles
- [Pipeline DAG](../user-guide/workflows.md) - Workflow design
- [Reproducibility](../foundations/reproducibility.md) - Deterministic execution
- [Trust & Uncertainty](../concepts/trust_uncertainty.md) - Trust stack integration
