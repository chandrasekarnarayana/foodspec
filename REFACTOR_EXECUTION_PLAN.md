# FoodSpec Architecture Refactor Execution Plan

**Status**: Ready for Execution  
**Date**: January 25, 2026  
**Phases**: 5 (0 = Safety, 1-4 = Refactor, 5 = Verify)  
**Estimated Time**: 1-2 hours (execution) + 30 min (verification)  
**Risk Level**: **MEDIUM** (git history preserved, full rollback available)  

---

## PHASE 0: CREATE SAFETY BRANCH + SNAPSHOT

**Purpose**: Create backup branch and document current state before any changes.

**Commands**:
```bash
# 1. Create snapshot of current state (backup branch)
git checkout -b backup/pre-refactor-$(date +%Y%m%d-%H%M%S)
git push origin backup/pre-refactor-$(date +%Y%m%d-%H%M%S)

# 2. Return to main working branch
git checkout phase-1/protocol-driven-core

# 3. Create working branch for refactor
git checkout -b refactor/single-source-tree

# 4. Record starting state
git log --oneline -1 > /tmp/refactor_start.txt
echo "Starting commit: $(cat /tmp/refactor_start.txt)"
```

**Expected Outcome**:
```
On branch refactor/single-source-tree
Your branch is up to date with 'origin/phase-1/protocol-driven-core'
```

**Rollback** (if needed before Phase 1 completion):
```bash
git checkout phase-1/protocol-driven-core
git branch -D refactor/single-source-tree
git branch -D backup/pre-refactor-*  # Optional: keep backup
```

---

## PHASE 1: ELIMINATE DUAL SOURCE TREES

**Purpose**: Canonicalize all code to `src/foodspec/`, delete `foodspec_rewrite/`.

**Key Operations**:
- Move core modules from rewrite to src/ (protocol, registry, orchestrator, validation)
- Delete foodspec_rewrite/
- Fix all imports
- Record manifest

### Phase 1A: Dry Run (Preview)

**Command**:
```bash
python scripts/refactor_executor.py \
  --phase 1 \
  --dry-run \
  --manifest-output /tmp/refactor_manifest_phase1.json
```

**Expected Output**:
```
Phase 1: Eliminate Dual Source Trees (DRY RUN)
============================================
[Preview] MOVE: foodspec_rewrite/foodspec/core/protocol.py → src/foodspec/core/protocol.py
[Preview] MOVE: foodspec_rewrite/foodspec/core/registry.py → src/foodspec/core/registry.py
[Preview] MOVE: foodspec_rewrite/foodspec/core/orchestrator.py → src/foodspec/core/orchestrator.py
[Preview] MOVE: foodspec_rewrite/foodspec/validation/ → src/foodspec/validation/
[Preview] MOVE: foodspec_rewrite/foodspec/preprocess/recipes.py → src/foodspec/preprocess/recipes.py
[Preview] REMOVE: foodspec_rewrite/
[Preview] UPDATE: Import references (21 files)

Dry run complete. Review changes and run with --execute to proceed.
```

### Phase 1B: Execute

**Command**:
```bash
python scripts/refactor_executor.py \
  --phase 1 \
  --execute \
  --manifest-output /tmp/refactor_manifest_phase1.json
```

**Expected Outcome**:
```
✓ MOVE: foodspec_rewrite/foodspec/core/protocol.py → src/foodspec/core/protocol.py
✓ MOVE: foodspec_rewrite/foodspec/core/registry.py → src/foodspec/core/registry.py
✓ MOVE: foodspec_rewrite/foodspec/core/orchestrator.py → src/foodspec/core/orchestrator.py
✓ MOVE: foodspec_rewrite/foodspec/validation/ → src/foodspec/validation/
✓ MOVE: foodspec_rewrite/foodspec/preprocess/recipes.py → src/foodspec/preprocess/recipes.py
✓ REMOVE: foodspec_rewrite/
✓ UPDATE: 21 import references

Manifest saved to /tmp/refactor_manifest_phase1.json
```

**Git State**:
```bash
git status
# Changes not staged for commit:
#   renamed: foodspec_rewrite/foodspec/core/protocol.py → src/foodspec/core/protocol.py
#   renamed: foodspec_rewrite/foodspec/core/registry.py → src/foodspec/core/registry.py
#   ...
#   deleted: foodspec_rewrite/

git add -A
git commit -m "refactor: consolidate to single source tree (src/foodspec/)"
```

### Phase 1C: Verify Imports

**Command**:
```bash
python -c "from foodspec.core.protocol import ProtocolV2; print('✓ Protocol OK')"
python -c "from foodspec.core.registry import ComponentRegistry; print('✓ Registry OK')"
python -c "from foodspec.core.orchestrator import ExecutionEngine; print('✓ Engine OK')"
python -c "from foodspec.validation.evaluation import evaluate_model_cv; print('✓ Eval OK')"
```

**Expected Outcome**:
```
✓ Protocol OK
✓ Registry OK
✓ Engine OK
✓ Eval OK
```

### Phase 1D: Rollback (if needed)

**Command**:
```bash
python scripts/refactor_executor.py \
  --rollback /tmp/refactor_manifest_phase1.json

# OR manual rollback
git reset --hard HEAD~1
git checkout phase-1/protocol-driven-core
```

---

## PHASE 2: CONSOLIDATE CONFIGS

**Purpose**: Single pyproject.toml, single mkdocs.yml, consistent metadata.

**Key Operations**:
- Delete `foodspec_rewrite/pyproject.toml`
- Update `./pyproject.toml` with merged dependencies
- Update `./pyproject.toml` with new CLI command: `foodspec = "foodspec.cli:main:run"`
- Delete `foodspec_rewrite/mkdocs.yml` (if separate)
- Record manifest

### Phase 2A: Dry Run

**Command**:
```bash
python scripts/refactor_executor.py \
  --phase 2 \
  --dry-run \
  --manifest-output /tmp/refactor_manifest_phase2.json
```

**Expected Output**:
```
Phase 2: Consolidate Configs (DRY RUN)
=====================================
[Preview] DELETE: foodspec_rewrite/pyproject.toml
[Preview] UPDATE: ./pyproject.toml
  - Add dependencies: pydantic>=2.0, jinja2>=3.0 (from rewrite)
  - ADD CLI script: foodspec = "foodspec.cli.main:run"
  - SET version: 1.1.0

Dry run complete.
```

### Phase 2B: Execute

**Command**:
```bash
python scripts/refactor_executor.py \
  --phase 2 \
  --execute \
  --manifest-output /tmp/refactor_manifest_phase2.json
```

**Verify**:
```bash
grep -A 5 "foodspec =" ./pyproject.toml
# [project.scripts]
# foodspec = "foodspec.cli.main:run"

grep "version" ./pyproject.toml | head -1
# version = "1.1.0"

pip install -e . --no-deps  # Test syntax
```

---

## PHASE 3: ARCHIVE INTERNAL DOCS + CLEAN BUILD ARTIFACTS

**Purpose**: Move phase history to _internal/, remove tracked build outputs.

**Key Operations**:
- Move foodspec_rewrite/*.md → _internal/phase-history/
- Remove site/, outputs/, __pycache__, .coverage, etc. from git tracking
- Add to .gitignore
- Update .gitignore to prevent future leakage

### Phase 3A: Dry Run

**Command**:
```bash
python scripts/refactor_executor.py \
  --phase 3 \
  --dry-run
```

**Expected Output**:
```
Phase 3: Archive Docs + Clean Artifacts (DRY RUN)
================================================
[Preview] MOVE: foodspec_rewrite/ARCHITECTURE.md → _internal/phase-history/rewrite-architecture.md
[Preview] GIT-RM: site/
[Preview] GIT-RM: outputs/
[Preview] GIT-RM: __pycache__/
[Preview] UPDATE: .gitignore

Dry run complete.
```

### Phase 3B: Execute

**Command**:
```bash
python scripts/refactor_executor.py \
  --phase 3 \
  --execute

# Commit separately for clarity
git add .gitignore
git commit -m "build: prevent tracking of generated files"

git add _internal/phase-history/
git commit -m "docs: archive phase-1 implementation docs"
```

**Verify**:
```bash
# Should show removals
git status | grep "deleted:"

# Should be empty
git ls-files | grep -E "^site/|^__pycache__|^outputs/" | wc -l
# 0
```

---

## PHASE 4: REORGANIZE EXAMPLES + SCRIPTS

**Purpose**: Organize by use case (not by file type).

**Directory Structure**:
```
examples/
├── quickstarts/          # Single-file, <5 min examples
│   ├── oil_authentication.py
│   ├── aging.py
│   ├── heating_quality.py
│   └── mixture_analysis.py
├── protocols/            # Full protocol YAML files
│   ├── oil_authentication_full.yaml
│   ├── ...
├── notebooks/            # Interactive notebooks (keep)
│   ├── ...
├── advanced/             # Multi-file, complex examples
│   ├── multimodal_fusion_demo.py
│   ├── vip_demo.py
│   └── ...
└── fixtures/             # Small test data
    ├── olive_oil_sample.csv
    └── ...

scripts/
├── refactor_executor.py        # This refactor script
├── validate_architecture.py    # Architecture enforcement
└── (archive old scripts to _internal/scripts-archive/)
```

### Phase 4A: Dry Run

**Command**:
```bash
python scripts/refactor_executor.py \
  --phase 4 \
  --dry-run
```

**Expected Output**:
```
Phase 4: Reorganize Examples + Scripts (DRY RUN)
===============================================
[Preview] MOVE: examples/oil_authentication_quickstart.py → examples/quickstarts/oil_authentication.py
[Preview] MOVE: examples/heating_quality_quickstart.py → examples/quickstarts/heating_quality.py
[Preview] CREATE: examples/protocols/
[Preview] MOVE: examples/configs/*.yaml → examples/protocols/
[Preview] CREATE: examples/advanced/
[Preview] MOVE: examples/*_demo.py → examples/advanced/

Dry run complete.
```

### Phase 4B: Execute

**Command**:
```bash
python scripts/refactor_executor.py \
  --phase 4 \
  --execute

git add examples/
git add scripts/
git commit -m "refactor: reorganize examples and scripts by use case"
```

---

## PHASE 5: VERIFICATION + TESTS

**Purpose**: Verify single source tree works, run CI tests, minimal integration test.

### Phase 5A: Architecture Validation

**Command**:
```bash
python scripts/validate_architecture.py --strict
```

**Expected Output**:
```
Architecture Validation Results
================================
✓ Single package root: src/foodspec/
✓ Only one pyproject.toml: ./pyproject.toml
✓ No foodspec_rewrite/ directory
✓ No conflicting imports
✓ All test_*.py files present
✓ CLI entrypoint callable

Result: PASS (All checks passed)
```

### Phase 5B: Run Unit Tests

**Command**:
```bash
pytest tests/test_architecture.py -v
pytest tests/test_architecture_ci.py -v
```

**Expected Output**:
```
tests/test_architecture.py::test_single_package_root PASSED
tests/test_architecture.py::test_one_pyproject_toml PASSED
tests/test_architecture.py::test_no_foodspec_rewrite PASSED
tests/test_architecture.py::test_imports_resolve PASSED
tests/test_architecture.py::test_cli_callable PASSED
tests/test_architecture_ci.py::test_manifest_completeness PASSED
tests/test_architecture_ci.py::test_minimal_e2e_run PASSED

====== 7 passed in 1.23s ======
```

### Phase 5C: Minimal End-to-End Run

**Command**:
```bash
# Create minimal test protocol
foodspec run examples/protocols/test_minimal.yaml \
  --output-dir ./test_run_final \
  --no-viz  # Skip slow viz

# Verify outputs
test ! -f ./test_run_final/manifest.json && echo "✗ Manifest missing" || echo "✓ Manifest OK"
test ! -f ./test_run_final/metrics.json && echo "✗ Metrics missing" || echo "✓ Metrics OK"
test ! -f ./test_run_final/predictions.json && echo "✗ Predictions missing" || echo "✓ Predictions OK"
test -d ./test_run_final && echo "✓ Run artifacts OK"
```

**Expected Outcome**:
```
✓ Manifest OK
✓ Metrics OK
✓ Predictions OK
✓ Run artifacts OK

Duration: ~15-20 seconds
```

---

## ROLLBACK PROCEDURES

### Full Rollback to Pre-Refactor State

**Command**:
```bash
# Go back to backup branch
git checkout backup/pre-refactor-*

# Or reset to commit before refactor
git reset --hard $(cat /tmp/refactor_start.txt)

# Delete working branch
git branch -D refactor/single-source-tree
```

### Selective Rollback (by Phase)

**Command**:
```bash
# Rollback to end of Phase N-1
python scripts/refactor_executor.py \
  --rollback /tmp/refactor_manifest_phase{N-1}.json

# Then commit
git add -A
git commit -m "refactor: rollback to phase N-1"
```

---

## SUMMARY TABLE

| Phase | Operations | Files Moved/Deleted | Expected Time | Risk |
|-------|-----------|-------------------|---|---|
| 0 | Create branches, snapshot | 0 | 2 min | 🟢 None |
| 1 | Move core modules, delete rewrite/ | 5 moved, 1 deleted | 10 min | 🟡 Medium (imports) |
| 2 | Consolidate configs | 1 deleted, 1 updated | 5 min | 🟢 Low |
| 3 | Archive docs, clean artifacts | 5-10 files | 5 min | 🟢 Low |
| 4 | Reorganize examples | ~20 files | 5 min | 🟢 Low |
| 5 | Verify + tests | 0 | 15 min | 🟢 None |

**Total**: 1-2 hours execution + 30 min verification

---

## CI ENFORCEMENT (Prevent Regression)

These tests must be added to `.github/workflows/architecture-enforce.yml`:

```yaml
name: Architecture Enforcement

on: [push, pull_request]

jobs:
  enforce:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Check single package root
        run: |
          count=$(find . -name "__init__.py" -path "*/foodspec/__init__.py" | wc -l)
          if [ $count -ne 1 ]; then
            echo "ERROR: Found $count foodspec package roots (expected 1)"
            exit 1
          fi
      
      - name: Check single pyproject.toml
        run: |
          count=$(find . -name "pyproject.toml" -not -path "./.git/*" | wc -l)
          if [ $count -ne 1 ]; then
            echo "ERROR: Found $count pyproject.toml files (expected 1)"
            exit 1
          fi
      
      - name: Check no foodspec_rewrite
        run: |
          if [ -d "foodspec_rewrite" ]; then
            echo "ERROR: foodspec_rewrite/ should not exist"
            exit 1
          fi
      
      - name: Test imports
        run: |
          python -c "from foodspec.core.protocol import ProtocolV2"
          python -c "from foodspec.core.registry import ComponentRegistry"
          python -c "from foodspec.core.orchestrator import ExecutionEngine"
```

---

## POST-REFACTOR CHECKLIST

- [ ] All phases executed without error
- [ ] Git history preserved (no force-push)
- [ ] Unit tests pass
- [ ] Minimal E2E test passes
- [ ] README updated to reflect new structure
- [ ] CI enforcement tests added
- [ ] Backup branch tagged
- [ ] Release notes prepared
- [ ] Team notified of import changes

---

**END OF EXECUTION PLAN**
