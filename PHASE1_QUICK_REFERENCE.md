# Phase 1 Quick Reference Card

## 🚀 Quick Start

```bash
# Run workflow
foodspec workflow-run \
    tests/fixtures/minimal_protocol.yaml \
    --input tests/fixtures/minimal.csv \
    --output-dir runs/exp1 \
    --seed 42

# Run tests
pytest tests/test_workflow_phase1.py -v --no-cov

# Check CLI help
foodspec workflow-run --help
```

## 📁 New Files (9 files, ~1500 lines)

**Core Orchestrator**:
- `src/foodspec/workflow/config.py` - WorkflowConfig validation
- `src/foodspec/workflow/fingerprint.py` - Fingerprinting + Manifest
- `src/foodspec/workflow/errors.py` - Exit codes (0, 2-9)
- `src/foodspec/workflow/artifact_contract.py` - Artifact validation
- `src/foodspec/workflow/phase1_orchestrator.py` - Main orchestrator (450 lines)

**CLI**:
- `src/foodspec/cli/commands/workflow.py` - Updated with new command

**Tests**:
- `tests/test_workflow_phase1.py` - 24 comprehensive tests
- `tests/fixtures/minimal.csv` - Test data (10 rows)
- `tests/fixtures/minimal_protocol.yaml` - Test protocol

## 📊 Test Results

```
✅ 24 passed, 8 warnings in 1.66s
├─ 5 config validation tests
├─ 5 fingerprinting tests
├─ 4 error handling tests
├─ 3 artifact contract tests
├─ 4 integration tests
└─ 3 parametrized tests (with variants)
```

## 🔄 Pipeline Steps

```
1. Setup run dir + logging
   └─ logs/run.log, logs/run.jsonl
   
2. Validate config
   
3. Load protocol
   └─ ProtocolConfig.from_file()
   
4. Read + fingerprint data
   └─ SHA256 + metadata
   
5. Preprocessing (Phase 2 stub)
   
6. Features (Phase 2 stub)
   
7. Modeling (if enabled)
   └─ fit_predict() with CV
   
8. Build manifest
   └─ Versions, fingerprints, timing
   
9. Validate artifacts
   └─ manifest.json, logs/run.log
   
10. Write success.json
    └─ exit code 0
    
OR on error:

10. Write error.json
    └─ exit code 2-9
```

## 📋 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | ✅ Success |
| 2 | ❌ CLI error |
| 3 | ❌ Validation error |
| 4 | ❌ Protocol error |
| 5 | ❌ Modeling error |
| 6 | ❌ Trust error |
| 7 | ❌ QC error |
| 8 | ❌ Reporting error |
| 9 | ❌ Artifact error |

## 🎯 Key Features

✅ **Input Validation**
- Config validation with helpful errors
- Protocol YAML/JSON loading
- CSV schema validation
- Dataset fingerprinting (SHA256)

✅ **Deterministic Execution**
- Random seed control (numpy, random, sklearn)
- Reproducible split generation
- Version tracking

✅ **Logging**
- Human-readable logs (run.log)
- Structured JSON logs (run.jsonl)
- Per-line JSON entries with timestamp, level, stage

✅ **Error Handling**
- 8 exception classes with exit codes
- error.json with hints and remediation
- Graceful failure paths

✅ **Artifact Tracking**
- Manifest.json with environment capture
- success.json on completion
- Artifact contract validation

✅ **CLI Integration**
- `foodspec workflow-run` command
- Full option support
- Backward compatible (no breaking changes)

## 🧪 Run a Test

```python
from pathlib import Path
from foodspec.workflow.config import WorkflowConfig
from foodspec.workflow.phase1_orchestrator import run_workflow

cfg = WorkflowConfig(
    protocol=Path("tests/fixtures/minimal_protocol.yaml"),
    inputs=[Path("tests/fixtures/minimal.csv")],
    output_dir=Path("/tmp/test_run"),
    mode="research",
    seed=42,
    enable_modeling=False,
)

exit_code = run_workflow(cfg)
print(f"Exit code: {exit_code}")
# Check /tmp/test_run/ for artifacts
```

## 📊 Manifest Contents

```json
{
  "protocol_fingerprint": {
    "sha256": "abc123...",
    "path": "protocol.yaml"
  },
  "dataset_fingerprints": [{
    "sha256": "def456...",
    "rows": 100,
    "columns": ["col1", "col2"],
    "missing_per_column": {"col1": 0.0}
  }],
  "foodspec_version": "2.0.0",
  "python_version": "3.12.9",
  "platform_info": "Linux-6.8.0-90-generic-x86_64-with-glibc2.35",
  "git_commit": "abc123def456...",
  "seed": 42,
  "mode": "research",
  "start_time": "2026-01-26T14:50:32Z",
  "end_time": "2026-01-26T14:50:34Z",
  "duration_seconds": 2.1,
  "artifacts": {...}
}
```

## 🔧 Typical Workflow

### Local Development
```bash
# Run one test
pytest tests/test_workflow_phase1.py::test_manifest_build -xvs

# Run all Phase 1 tests
pytest tests/test_workflow_phase1.py -v

# Test CLI manually
foodspec workflow-run tests/fixtures/minimal_protocol.yaml \
    --input tests/fixtures/minimal.csv \
    --verbose
```

### Production Use
```bash
# Execute workflow in research mode
foodspec workflow-run \
    data/protocol.yaml \
    --input data/oils.csv \
    --output-dir runs/production \
    --seed 42 \
    --model LogisticRegression

# Check exit code
echo $?

# Review results
cat runs/production/manifest.json
cat runs/production/logs/run.log
```

## 🐛 Debugging

### View logs
```bash
cat runs/run_1/logs/run.log
cat runs/run_1/logs/run.jsonl | python -m json.tool
```

### Check error
```bash
cat runs/run_1/error.json | python -m json.tool
```

### Inspect manifest
```bash
python -c "
import json
with open('runs/run_1/manifest.json') as f:
    manifest = json.load(f)
    print(f'Mode: {manifest[\"mode\"]}')
    print(f'Seed: {manifest[\"seed\"]}')
    print(f'Duration: {manifest[\"duration_seconds\"]}s')
    print(f'Dataset hash: {manifest[\"dataset_fingerprints\"][0][\"sha256\"][:8]}...')
"
```

## 📚 Documentation

- **README**: [PHASE1_IMPLEMENTATION_SUMMARY.md](PHASE1_IMPLEMENTATION_SUMMARY.md)
- **Code**: Comprehensive docstrings in all modules
- **Tests**: 24 tests demonstrating usage
- **CLI**: `foodspec workflow-run --help`

## ✨ What's Next (Phase 2)

1. QC gates (data, spectral, model)
2. Regulatory mode enforcement
3. Trust stack integration
4. Compliance statements
5. Full feature/preprocessing integration
6. Multi-input support

---

**Status**: ✅ Phase 1 Complete (24/24 tests passing)
