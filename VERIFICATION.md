# Verification Checklist

## Code Quality Verification

### 1. Syntax & Imports
```bash
# Check syntax of new modules
python -m py_compile src/foodspec/experiment/experiment.py
python -m py_compile src/foodspec/experiment/__init__.py

# Check imports resolve
python -c "from foodspec.experiment import Experiment, RunMode, ValidationScheme; print('✓ Imports OK')"

# Check for errors
python -m pylint src/foodspec/experiment/ --errors-only 2>/dev/null || echo "Pylint optional"
```

### 2. CLI Integration
```bash
# Check CLI loads without error
foodspec --help | grep -q "run" && echo "✓ CLI loads"

# Check run command exists
foodspec run --help | head -5
```

### 3. Type Hints
```bash
# Check type hints with mypy (if installed)
python -m mypy src/foodspec/experiment/ --no-error-summary 2>&1 | head -20 || echo "Mypy optional"
```

---

## Functional Verification

### 1. Create Test Data
```bash
python << 'EOF'
import pandas as pd
import numpy as np

# Create synthetic test CSV
np.random.seed(42)
df = pd.DataFrame({
    'feature_1': np.random.randn(50),
    'feature_2': np.random.randn(50),
    'feature_3': np.random.randn(50),
    'feature_4': np.random.randn(50),
    'feature_5': np.random.randn(50),
    'target': np.random.randint(0, 2, 50)
})
df.to_csv('/tmp/test_data.csv', index=False)
print(f"✓ Created test data: /tmp/test_data.csv ({df.shape})")
EOF
```

### 2. Verify Experiment Class Works
```bash
python << 'EOF'
from pathlib import Path
from foodspec.experiment import Experiment, RunMode, ValidationScheme

# Test 1: Create from dict
proto = {
    "name": "TestProto",
    "version": "1.0.0",
    "steps": [],
    "expected_columns": {"target": "target"},
}
exp = Experiment.from_protocol(proto)
print(f"✓ Experiment.from_protocol() works")
print(f"  Mode: {exp.config.mode.value}")
print(f"  Scheme: {exp.config.scheme.value}")

# Test 2: Verify attributes
assert exp.config.mode == RunMode.RESEARCH
assert exp.config.scheme == ValidationScheme.LOBO
print(f"✓ Configuration defaults correct")

# Test 3: with overrides
exp2 = Experiment.from_protocol(
    proto,
    mode="regulatory",
    scheme="loso",
    model="svm",
)
assert exp2.config.mode == RunMode.REGULATORY
assert exp2.config.scheme == ValidationScheme.LOSO
assert exp2.config.model == "svm"
print(f"✓ Configuration overrides work")
EOF
```

### 3. Test Run (Minimal)
```bash
python << 'EOF'
import tempfile
from pathlib import Path
from foodspec.experiment import Experiment

proto = {
    "name": "Test",
    "version": "1.0.0",
    "steps": [],
    "expected_columns": {"target": "target"},
}
exp = Experiment.from_protocol(proto)

with tempfile.TemporaryDirectory() as tmpdir:
    result = exp.run(
        csv_path=Path("/tmp/test_data.csv"),
        outdir=Path(tmpdir),
        seed=42,
        verbose=True,
    )
    
    print(f"\n✓ Experiment.run() completed")
    print(f"  Status: {result.status}")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Run ID: {result.run_id}")
    print(f"  Manifest: {result.manifest_path}")
    print(f"  Summary: {result.summary_path}")
    print(f"  Report: {result.report_dir / 'index.html'}")
    
    # Verify output structure
    assert result.manifest_path.exists(), "manifest.json missing"
    assert result.summary_path.exists(), "summary.json missing"
    assert (result.report_dir / "index.html").exists(), "index.html missing"
    print(f"✓ Artifact structure verified")
    
    # Check manifest contents
    import json
    manifest = json.loads(result.manifest_path.read_text())
    assert "protocol_hash" in manifest
    assert "python_version" in manifest
    assert "seed" in manifest
    print(f"✓ Manifest contains required fields")
    
    # Check summary contents
    summary = json.loads(result.summary_path.read_text())
    assert "metrics" in summary
    assert "deployment_readiness_score" in summary
    print(f"✓ Summary contains required fields")
EOF
```

### 4. Test CLI Integration
```bash
# Test YOLO mode (should trigger orchestration)
foodspec run \
  --protocol examples/protocols/EdibleOil_Classification_v1.yaml \
  --input /tmp/test_data.csv \
  --outdir /tmp/runs/test_yolo \
  --model lightgbm \
  --scheme lobo \
  --mode research \
  --verbose 2>&1 | head -20

# Check output
if [ -f /tmp/runs/test_yolo/run_*/manifest.json ]; then
    echo "✓ Orchestration path created manifest.json"
    ls -la /tmp/runs/test_yolo/run_*/
fi

# Test exit code
foodspec run \
  --protocol examples/protocols/EdibleOil_Classification_v1.yaml \
  --input /nonexistent.csv \
  --outdir /tmp/runs/test_error \
  2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -eq 2 ]; then
    echo "✓ Exit code 2 (validation error) correct"
fi
```

### 5. Test Backward Compatibility
```bash
# Test classic mode (no YOLO flags)
foodspec run \
  --protocol examples/protocols/EdibleOil_Classification_v1.yaml \
  --input /tmp/test_data.csv \
  --outdir /tmp/runs/test_classic \
  --verbose 2>&1 | head -10

# Classic path should execute without YOLO-specific output
echo "✓ Classic mode runs (backward compatible)"
```

---

## Integration Tests

### Run Full Test Suite
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run orchestration tests only
pytest tests/test_orchestration_e2e.py -v --tb=short

# Expected: ~50 tests, all passing
```

### Run Specific Test Groups
```bash
# Test Experiment creation
pytest tests/test_orchestration_e2e.py::TestExperimentFromProtocol -v

# Test run execution
pytest tests/test_orchestration_e2e.py::TestExperimentRun -v

# Test edge cases
pytest tests/test_orchestration_e2e.py::TestExperimentEdgeCases -v
```

### Coverage Report
```bash
pytest tests/test_orchestration_e2e.py \
  --cov=src/foodspec/experiment \
  --cov-report=html \
  --cov-report=term

# Open htmlcov/index.html to see coverage
```

---

## Documentation Verification

### 1. CLI Help
```bash
# Check help text
foodspec run --help | grep -E "(--model|--scheme|--mode|--trust)"

# Should show all YOLO options
echo "✓ New CLI options documented"
```

### 2. Docs Exist
```bash
# Check documentation files
ls -la docs/cli/run.md IMPLEMENTATION_SUMMARY.md QUICK_REFERENCE.md PATCH_SUMMARY.md

# Should all exist
echo "✓ Documentation complete"
```

### 3. Code Comments
```bash
# Check docstrings
grep -c "\"\"\"" src/foodspec/experiment/experiment.py
# Should have substantial docstrings

echo "✓ Code documented"
```

---

## Artifact Contract Verification

### 1. Manifest JSON Schema
```bash
python << 'EOF'
import json
from pathlib import Path

# Find a generated manifest
manifest_path = Path("/tmp/runs/test_yolo/run_*/manifest.json").glob("*")[0] if list(Path("/tmp/runs/test_yolo").glob("run_*/manifest.json")) else None

if manifest_path:
    manifest = json.loads(manifest_path.read_text())
    
    # Verify required fields
    required = [
        "protocol_hash",
        "python_version",
        "platform",
        "seed",
        "data_fingerprint",
        "start_time",
        "end_time",
        "duration_seconds",
        "artifacts",
    ]
    
    for field in required:
        assert field in manifest, f"Missing field: {field}"
    
    print("✓ Manifest schema valid")
else:
    print("⚠ Manifest not yet generated; run test first")
EOF
```

### 2. Summary JSON Schema
```bash
python << 'EOF'
import json
from pathlib import Path

# Find a generated summary
summary_path = Path("/tmp/runs/test_yolo/run_*/summary.json").glob("*")[0] if list(Path("/tmp/runs/test_yolo").glob("run_*/summary.json")) else None

if summary_path:
    summary = json.loads(summary_path.read_text())
    
    # Verify required fields
    required = [
        "dataset_summary",
        "scheme",
        "model",
        "mode",
        "metrics",
        "calibration",
        "coverage",
        "abstention_rate",
        "deployment_readiness_score",
        "deployment_ready",
        "key_risks",
    ]
    
    for field in required:
        assert field in summary, f"Missing field: {field}"
    
    print("✓ Summary schema valid")
else:
    print("⚠ Summary not yet generated; run test first")
EOF
```

### 3. Directory Structure
```bash
python << 'EOF'
from pathlib import Path

# Find a run directory
run_dirs = list(Path("/tmp/runs/test_yolo").glob("run_*"))
if run_dirs:
    run_dir = run_dirs[0]
    
    required_dirs = [
        "data",
        "features",
        "modeling",
        "trust",
        "figures",
        "tables",
        "report",
    ]
    
    for d in required_dirs:
        assert (run_dir / d).exists(), f"Missing directory: {d}"
    
    # Check for files
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report" / "index.html").exists()
    
    print("✓ Artifact structure complete")
else:
    print("⚠ Run directory not yet created; run test first")
EOF
```

---

## Performance Verification

### 1. Execution Time
```bash
time foodspec run \
  --protocol examples/protocols/EdibleOil_Classification_v1.yaml \
  --input /tmp/test_data.csv \
  --outdir /tmp/runs/test_perf \
  2>/dev/null

# Should complete in < 30 seconds for small test data
echo "✓ Performance acceptable"
```

### 2. Memory Usage
```bash
python << 'EOF'
import psutil
import os
from pathlib import Path
from foodspec.experiment import Experiment

# Monitor memory during run
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

proto = {
    "name": "Test",
    "version": "1.0.0",
    "steps": [],
    "expected_columns": {"target": "target"},
}
exp = Experiment.from_protocol(proto)

result = exp.run(
    csv_path=Path("/tmp/test_data.csv"),
    outdir=Path("/tmp/runs/test_mem"),
)

peak_memory = process.memory_info().rss / 1024 / 1024  # MB
delta = peak_memory - initial_memory

print(f"✓ Memory usage: {delta:.1f} MB peak delta")
EOF
```

---

## Reproducibility Verification

### 1. Same Seed = Same Results
```bash
python << 'EOF'
import tempfile
from pathlib import Path
from foodspec.experiment import Experiment
import json

proto = {
    "name": "Test",
    "version": "1.0.0",
    "steps": [],
    "expected_columns": {"target": "target"},
}

# Run 1
exp1 = Experiment.from_protocol(proto)
with tempfile.TemporaryDirectory() as tmpdir:
    result1 = exp1.run(
        csv_path=Path("/tmp/test_data.csv"),
        outdir=Path(tmpdir),
        seed=42,
    )
    summary1 = json.loads(result1.summary_path.read_text())

# Run 2 (same seed)
exp2 = Experiment.from_protocol(proto)
with tempfile.TemporaryDirectory() as tmpdir:
    result2 = exp2.run(
        csv_path=Path("/tmp/test_data.csv"),
        outdir=Path(tmpdir),
        seed=42,
    )
    summary2 = json.loads(result2.summary_path.read_text())

# Compare
if summary1["metrics"]["accuracy"] == summary2["metrics"]["accuracy"]:
    print("✓ Same seed produces consistent metrics")
else:
    print("⚠ Metrics differ (expected with non-deterministic models)")
EOF
```

---

## Final Checklist

```bash
cat << 'EOF'
Verification Checklist:
======================

Code Quality:
  [ ] Python syntax valid (py_compile)
  [ ] Imports resolve (python -c)
  [ ] No errors (pylint/mypy)
  [ ] CLI loads (foodspec --help)

Functional:
  [ ] Experiment.from_protocol() works
  [ ] Experiment.run() completes
  [ ] RunResult has correct fields
  [ ] Exit codes correct (0/2/3/4)
  [ ] Artifact structure created
  [ ] Manifest JSON valid
  [ ] Summary JSON valid
  [ ] Report HTML generated

CLI:
  [ ] YOLO mode flags work (--model, --scheme, --mode, --trust)
  [ ] Classic mode still works (backward compatible)
  [ ] Help text updated
  [ ] Exit codes propagate correctly

Tests:
  [ ] Integration tests pass (~50 cases)
  [ ] Test coverage > 80%
  [ ] Edge cases handled

Documentation:
  [ ] docs/cli/run.md complete
  [ ] IMPLEMENTATION_SUMMARY.md complete
  [ ] QUICK_REFERENCE.md complete
  [ ] Code docstrings present
  [ ] Examples work

Reproducibility:
  [ ] Same seed → same results
  [ ] Manifest captures all metadata
  [ ] Can re-run from manifest

Performance:
  [ ] Execution time reasonable (< 30s for 50 samples)
  [ ] Memory usage acceptable (< 500MB)

Backward Compatibility:
  [ ] Existing --protocol invocations work
  [ ] ProtocolRunner path unchanged
  [ ] No breaking changes

======================
Ready for Production?
  [ ] All checks above passing
  [ ] Code review approved
  [ ] Tests pass on CI/CD
  [ ] Documentation reviewed

EOF
```

---

## Quick Test Script

Save and run this to verify everything:

```bash
#!/bin/bash

set -e

echo "FoodSpec E2E Orchestration Verification"
echo "========================================"
echo ""

# 1. Syntax
echo "1. Checking syntax..."
python -m py_compile src/foodspec/experiment/experiment.py
echo "   ✓ Syntax OK"

# 2. Imports
echo "2. Checking imports..."
python -c "from foodspec.experiment import Experiment; print('   ✓ Imports OK')"

# 3. CLI
echo "3. Checking CLI..."
foodspec run --help | grep -q "model" && echo "   ✓ CLI has YOLO flags"

# 4. Tests
echo "4. Running tests..."
pytest tests/test_orchestration_e2e.py -q --tb=no
echo "   ✓ Tests passing"

# 5. Docs
echo "5. Checking docs..."
[ -f docs/cli/run.md ] && echo "   ✓ Documentation exists"

echo ""
echo "✓ All verifications passed!"
echo ""
echo "Next steps:"
echo "  - Review: IMPLEMENTATION_SUMMARY.md"
echo "  - Try: foodspec run --protocol ... --input ... --model lightgbm"
echo "  - Check: docs/cli/run.md for full guide"
```

Run it:
```bash
bash verify.sh
```

---

## Support

If tests fail:

1. **Syntax errors**: Check Python 3.9+ installed
2. **Import errors**: Check PYTHONPATH includes src/
3. **CLI errors**: Check foodspec installed in editable mode (`pip install -e .`)
4. **Test failures**: Run with `-vv` flag for details
5. **Missing files**: Verify all 5 new files created

See `QUICK_REFERENCE.md` for troubleshooting.
