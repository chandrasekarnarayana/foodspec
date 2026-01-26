# FoodSpec Execution Engine - Quick Reference

## Files Created/Modified

### New Modules (6 files)
1. `src/foodspec/core/philosophy.py` - Design principles & enforcement (330 lines)
2. `src/foodspec/utils/determinism.py` - Reproducibility infrastructure (250 lines)
3. `src/foodspec/engine/dag.py` - Pipeline DAG system (370 lines)
4. `src/foodspec/engine/artifacts.py` - Artifact registry (300 lines)
5. `src/foodspec/core/run_manifest.py` - Run metadata (350 lines)
6. `src/foodspec/engine/orchestrator.py` - Central orchestrator (400 lines)

### Updated Files (2 files)
7. `src/foodspec/engine/__init__.py` - Export orchestration classes
8. `src/foodspec/core/__init__.py` - Export philosophy & manifest classes

### Tests (1 file)
9. `tests/engine/test_execution_engine.py` - 35+ tests (520 lines)

### Documentation (2 files)
10. `docs/concepts/design_philosophy.md` - Philosophy guide (600 lines)
11. `docs/concepts/execution_engine.md` - Engine guide (700 lines)

### Summary (1 file)
12. `EXECUTION_ENGINE_SUMMARY.md` - Complete implementation summary

---

## 7 Design Principles (Enforced at Runtime)

| Principle | Check | Enforcement |
| --- | --- | --- |
| **TASK_FIRST** | Task ∈ {authentication, adulteration, monitoring} | `enforce_task_first()` |
| **QC_FIRST** | QC runs before model, pass_rate ≥ 50% | `enforce_qc_first()` |
| **TRUST_FIRST** | Calibration + conformal outputs required | `enforce_trust_first()` |
| **PROTOCOL_TRUTH** | Protocol immutable, hashable | `enforce_protocol_truth()` |
| **REPRODUCIBILITY** | Seed, versions, env captured | `enforce_reproducibility()` |
| **DUAL_API** | CLI + programmatic both work | N/A (architectural) |
| **REPORT_FIRST** | Reports auto-generated | `enforce_report_first()` |

---

## 12-Step Orchestration

```python
engine = ExecutionEngine(run_id="exp_001")

1.  engine.validate_protocol(protocol, protocol_dict)   # TASK_FIRST check
2.  engine.validate_data(csv_path)                      # Data fingerprinting
3.  engine.setup_reproducibility(seed=42)               # Global seed
4.  engine.setup_pipeline()                             # DAG creation
5.  engine.register_stage_function("preprocess", func)  # Register stages
6.  engine.register_stage_function("qc", func)          # ...
7.  engine.register_stage_function("model", func)       # ...
8.  engine.register_stage_function("trust", func)       # ...
9.  engine.execute_pipeline()                           # Execute all stages
10. engine.finalize_artifacts()                         # Validate registry
11. engine.generate_manifest(out_dir)                   # Create manifest.json
12. # Philosophy checks happen automatically at checkpoints
```

---

## Quick Start

### Verify Installation
```bash
python -c "from foodspec.engine.orchestrator import ExecutionEngine; print('✓ OK')"
```

### Run Tests
```bash
pytest tests/engine/test_execution_engine.py -v
```

### Minimal Example
```python
from foodspec.engine.orchestrator import ExecutionEngine
from foodspec.engine.artifacts import ArtifactType

engine = ExecutionEngine()
engine.setup_reproducibility(seed=42)
manifest = engine.generate_manifest(Path("output/"))
print(f"✓ Manifest: {manifest}")
```

---

## Key Classes

| Class | Module | Purpose |
| --- | --- | --- |
| `ExecutionEngine` | `orchestrator` | Central orchestrator |
| `PipelineDAG` | `dag` | Dependency graph |
| `ArtifactRegistry` | `artifacts` | Output tracking |
| `RunManifest` | `run_manifest` | Metadata record |
| `DESIGN_PRINCIPLES` | `philosophy` | Principle singleton |

---

## Import Shortcuts

```python
# Orchestration
from foodspec.engine.orchestrator import ExecutionEngine

# Philosophy
from foodspec.core.philosophy import DESIGN_PRINCIPLES, PhilosophyError, validate_all_principles

# DAG
from foodspec.engine.dag import PipelineDAG, build_standard_pipeline

# Artifacts
from foodspec.engine.artifacts import ArtifactRegistry, ArtifactType

# Manifest
from foodspec.core.run_manifest import ManifestBuilder, RunStatus

# Determinism
from foodspec.utils.determinism import set_global_seed, fingerprint_csv
```

---

## Output Structure

```
runs/exp_001/
├── run_manifest.json      # ← Complete metadata
├── artifacts.json         # ← Registry
├── metrics.csv
├── predictions.csv
├── qc/qc_report.json
├── trust/calibration.json
├── trust/conformal.json
├── plots/*.png
└── reports/report.html
```

---

## Philosophy Violations

All violations raise `PhilosophyError`:

```python
try:
    engine.validate_protocol(protocol, {"task": "invalid"})
except PhilosophyError as e:
    print(e)  # "Task 'invalid' not in TASK_FIRST: [...]"
```

---

## Next Steps

1. ✅ Verify imports: `python -c "from foodspec.engine.orchestrator import ExecutionEngine"`
2. ✅ Run tests: `pytest tests/engine/test_execution_engine.py -v`
3. ⬜ Integrate with CLI: Update `foodspec run` to use ExecutionEngine
4. ⬜ Add to CI/CD: Ensure philosophy tests run automatically
5. ⬜ Update examples: Show ExecutionEngine usage

---

## Documentation

- **Philosophy**: `docs/concepts/design_philosophy.md`
- **Engine Guide**: `docs/concepts/execution_engine.md`
- **Summary**: `EXECUTION_ENGINE_SUMMARY.md`
- **Tests**: `tests/engine/test_execution_engine.py`

---

## Status

✅ **All 10 tasks complete**  
✅ **2500+ lines of production code**  
✅ **35+ comprehensive tests**  
✅ **1300+ lines of documentation**  
✅ **Ready for integration**
