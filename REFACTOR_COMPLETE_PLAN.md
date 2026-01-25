# FoodSpec Architecture Refactor: Complete Execution Plan

**Status**: Ready for Execution  
**Date**: January 25, 2026  
**Scope**: Consolidate dual-tree architecture to single coherent source  
**Risk Level**: MEDIUM (git history preserved, full rollback available)  
**Estimated Duration**: 2-3 hours (execution) + 30 min (verification) + continuous enforcement

---

## EXECUTIVE SUMMARY

The FoodSpec repository currently has a **dangerous split architecture**:
- **Legacy**: `src/foodspec/` (installed, contains old APIs + merged modules)
- **Rewrite**: `foodspec_rewrite/foodspec/` (shadow duplicate, contains new architecture)

This creates:
- ❌ Import ambiguity (which foodspec gets imported?)
- ❌ Maintenance burden (two copies of same code)
- ❌ Silent failures (different APIs in each tree)
- ❌ Unclear upgrade path for users

**Solution**: Consolidate to **single source tree** with **phased, safe refactoring**:
1. Phase 0: Create safety branches + snapshot
2. Phase 1: Move rewrite modules to src/, delete foodspec_rewrite/
3. Phase 2: Single pyproject.toml, single CLI entrypoint
4. Phase 3: Archive internal docs, clean build artifacts
5. Phase 4: Reorganize examples by use case
6. Phase 5: Verify + run enforcement tests

**Result**: Single coherent architecture, one-command full run: `foodspec run protocol.yaml`

---

## DELIVERABLES

### 1. Documentation ✅
- **ARCHITECTURE_AUDIT_REPORT.md**: Evidence-based audit of current state
- **REFACTOR_EXECUTION_PLAN.md**: Exact commands for each phase
- **CANONICAL_MODULE_MAP.md**: Final target layout (authoritative)
- **This document**: Overview + enforcement strategy

### 2. Scripts ✅
- **scripts/refactor_executor.py**: Automated refactor with dry-run + manifest
- **scripts/validate_architecture.py**: Validation tool (can be run anytime)

### 3. Tests ✅
- **tests/test_architecture.py**: Architecture enforcement (8 tests)
- **tests/test_architecture_ci.py**: CI-level integration tests (10+ tests)
- **.github/workflows/architecture-enforce.yml**: CI enforcement workflow

### 4. Test Data ✅
- **examples/fixtures/TEST_PROTOCOL_MINIMAL.md**: Minimal E2E test spec
- **test_minimal.yaml**: Protocol for fast CI runs

---

## PHASE EXECUTION GUIDE

### PHASE 0: SAFETY & SNAPSHOT (2 minutes)

**Purpose**: Create unbreakable backup before any changes.

**Commands**:
```bash
# Create dated backup branch
git checkout -b backup/pre-refactor-$(date +%Y%m%d-%H%M%S)
git push origin backup/pre-refactor-$(date +%Y%m%d-%H%M%S)

# Return to working branch
git checkout phase-1/protocol-driven-core

# Create refactor branch
git checkout -b refactor/single-source-tree

# Document starting state
git log --oneline -1 > /tmp/refactor_start.txt
```

**Rollback**: `git checkout phase-1/protocol-driven-core && git branch -D refactor/single-source-tree`

---

### PHASE 1: ELIMINATE DUAL TREES (10 minutes)

**Purpose**: Canonicalize all code to `src/foodspec/`.

**Dry Run**:
```bash
python scripts/refactor_executor.py --phase 1 --dry-run
# Shows what would be moved without doing it
```

**Execute**:
```bash
python scripts/refactor_executor.py \
  --phase 1 \
  --execute \
  --manifest-output /tmp/manifest_phase1.json

# Commit
git add -A
git commit -m "refactor: consolidate to single source tree (src/foodspec/)"
```

**Verify**:
```bash
# Should pass
python -c "from foodspec.core.protocol import ProtocolV2; print('✓ OK')"
python -c "from foodspec.core.registry import ComponentRegistry; print('✓ OK')"
python -c "from foodspec.core.orchestrator import ExecutionEngine; print('✓ OK')"

# Should fail
python -c "import foodspec_rewrite" 2>&1 | grep "ModuleNotFoundError"
```

**Rollback**:
```bash
git reset --hard HEAD~1
git checkout phase-1/protocol-driven-core
```

---

### PHASE 2: CONSOLIDATE CONFIGS (5 minutes)

**Purpose**: Single pyproject.toml, single CLI entrypoint.

**Dry Run**:
```bash
python scripts/refactor_executor.py --phase 2 --dry-run
```

**Execute**:
```bash
python scripts/refactor_executor.py \
  --phase 2 \
  --execute \
  --manifest-output /tmp/manifest_phase2.json

git add pyproject.toml
git commit -m "build: consolidate to single pyproject.toml and version 1.1.0"
```

**Verify**:
```bash
grep -A 3 "foodspec =" ./pyproject.toml
# Should show: foodspec = "foodspec.cli:app" or "foodspec.cli.main:run"

pip install -e . --no-deps  # Verify syntax
```

---

### PHASE 3: ARCHIVE & CLEAN (5 minutes)

**Purpose**: Move phase docs to _internal/, remove build artifacts from git tracking.

**Execute**:
```bash
python scripts/refactor_executor.py \
  --phase 3 \
  --execute

git add .gitignore
git commit -m "build: prevent tracking of generated files"

git add _internal/phase-history/
git commit -m "docs: archive phase-1 implementation"
```

**Verify**:
```bash
ls -la _internal/phase-history/
# Should contain rewrite-*.md files

git ls-files | grep "^site/" | wc -l  # Should be 0
```

---

### PHASE 4: REORGANIZE EXAMPLES (5 minutes)

**Purpose**: Group examples by use case (quickstarts, protocols, advanced).

**Execute**:
```bash
python scripts/refactor_executor.py \
  --phase 4 \
  --execute

git add examples/
git add scripts/
git commit -m "refactor: reorganize examples and scripts by use case"
```

**Verify**:
```bash
ls examples/quickstarts/
# oil_authentication.py, heating_quality.py, aging.py, mixture_analysis.py

ls examples/protocols/
# *.yaml files
```

---

### PHASE 5: VERIFY + ENFORCE (30 minutes)

**Purpose**: Verify refactoring complete, establish CI enforcement.

#### 5A: Architecture Validation

```bash
# Run validation script
python scripts/validate_architecture.py --strict

# Expected output:
# ✓ Single package root: src/foodspec/
# ✓ No foodspec_rewrite directory
# ✓ Single pyproject.toml at repo root
# ✓ Import Protocol
# ✓ Import Registry
# ... (all checks pass)
```

#### 5B: Run Architecture Tests

```bash
# Install dependencies
pip install -e . --no-deps
pip install pytest

# Run tests
pytest tests/test_architecture.py -v
pytest tests/test_architecture_ci.py -v

# Expected: All pass
```

#### 5C: Minimal E2E Run

```bash
# This is what CI will run on every commit
# (Runs in <30 seconds)

foodspec run examples/protocols/test_minimal.yaml \
  --output-dir ./test_run_final \
  --no-viz \
  --no-report

# Verify outputs
test -f ./test_run_final/manifest.json && echo "✓ Manifest OK"
test -f ./test_run_final/metrics.json && echo "✓ Metrics OK"
test -f ./test_run_final/predictions.json && echo "✓ Predictions OK"
```

#### 5D: Enable CI Enforcement

```bash
# Create workflow (already done in .github/workflows/architecture-enforce.yml)
# It will run automatically on every push/PR

# Verify it runs
git push origin refactor/single-source-tree
# Check GitHub Actions tab for "Architecture Enforcement" workflow
```

---

## CANONICAL MODULE MAP (FINAL STATE)

After refactoring, the structure MUST be:

```
src/foodspec/
├── core/
│   ├── protocol.py          # ProtocolV2 (Pydantic-based)
│   ├── registry.py          # ComponentRegistry + defaults
│   ├── orchestrator.py      # ExecutionEngine (stage orchestration)
│   ├── artifacts.py         # ArtifactRegistry (standard paths)
│   └── manifest.py          # RunManifest (provenance)
├── validation/
│   ├── evaluation.py        # evaluate_model_cv, nested_cv
│   ├── splits.py            # CV splitters
│   ├── metrics.py           # 20+ metrics
│   ├── nested.py            # Nested CV
│   └── leakage.py           # Leakage detection
├── trust/
│   ├── conformal.py         # Split conformal, ICP
│   ├── calibration.py       # Platt, isotonic
│   ├── abstain.py           # Confidence-based
│   └── evaluator.py         # TrustEvaluator orchestrator
├── preprocess/
│   ├── baseline.py          # 6 baseline methods
│   ├── smoothing.py         # Savitzky-Golay, MA
│   ├── normalization.py     # SNV, MSC, vector
│   └── recipes.py           # Protocol-driven chains
├── features/
│   ├── peaks.py             # Peak detection
│   ├── bands.py             # Band integration
│   ├── chemometrics.py      # PCA, PLS, VIP
│   └── selection.py         # Feature selection
├── models/
│   ├── classical.py         # PLS, SVM, RF, Logistic
│   └── boosting.py          # XGBoost, LightGBM
├── viz/
│   ├── compare.py           # Multi-run comparison
│   ├── uncertainty.py       # Confidence bands
│   ├── coefficients.py      # Model coeff heatmaps
│   └── paper.py             # Publication presets
├── reporting/
│   ├── dossier.py           # HTML dossier
│   ├── pdf.py               # PDF export
│   └── export.py            # Archive bundles
├── deploy/
│   ├── bundle.py            # DeploymentBundle
│   ├── predict.py           # Inference
│   └── migration.py         # v1.0 → v2.0
└── cli/
    └── main.py              # Entry: foodspec run ...
```

**NO OTHER LAYOUT IS ACCEPTABLE.** CI tests will fail if violated.

---

## ENFORCEMENT STRATEGY

### CI Enforcement (Automatic)

On every push/PR to main or phase-1/:

1. **Architecture Checks** (pass/fail):
   - ✅ Exactly 1 foodspec/__init__.py
   - ✅ No foodspec_rewrite/ directory
   - ✅ Exactly 1 pyproject.toml
   - ✅ All critical imports work
   - ✅ No imports from removed paths
   - ✅ All core modules exist

2. **Test Checks** (pass/fail):
   - ✅ tests/test_architecture.py passes
   - ✅ tests/test_architecture_ci.py passes (key tests)

3. **Integration Checks** (optional):
   - ⚠ Minimal E2E run completes
   - ⚠ Manifest has required fields

**If any MUST-PASS check fails, PR will be blocked.**

### Local Pre-Commit Checks (Optional)

Can add `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python scripts/validate_architecture.py --strict || exit 1
```

### Regression Prevention

Tests will fail if:
- ❌ foodspec_rewrite/ is recreated
- ❌ Multiple pyproject.toml files appear
- ❌ Imports from old paths work (should fail)
- ❌ Duplicate class implementations found
- ❌ Missing core modules
- ❌ Manifest lacks required fields

---

## ROLLBACK PROCEDURES

### Full Rollback (to pre-refactor state)

```bash
# Option A: Go back to backup branch
git checkout backup/pre-refactor-YYYYMMDD-HHMMSS
git push origin backup/pre-refactor-YYYYMMDD-HHMMSS

# Option B: Reset commit by commit
git reset --hard HEAD~5  # Adjust N as needed
git checkout phase-1/protocol-driven-core

# Option C: Discard working branch entirely
git checkout phase-1/protocol-driven-core
git branch -D refactor/single-source-tree
```

### Selective Rollback (e.g., stop after Phase 2)

```bash
# Use saved manifest from previous phase
python scripts/refactor_executor.py --rollback /tmp/manifest_phase1.json
git add -A
git commit -m "refactor: rollback to end of phase 1"
```

---

## TIMELINE & EXPECTATIONS

| Phase | Duration | Critical? | Rollback Cost |
|-------|----------|-----------|---|
| 0 (Safety) | 2 min | ❌ No | N/A (no changes) |
| 1 (Consolidate) | 10 min | ⚠ Yes | LOW (easy git reset) |
| 2 (Configs) | 5 min | ⚠ Yes | LOW (edit pyproject) |
| 3 (Archive) | 5 min | ❌ No | LOW (git rm --cached) |
| 4 (Reorganize) | 5 min | ❌ No | LOW (git mv) |
| 5 (Verify) | 30 min | ✅ CRITICAL | N/A (tests only) |

**Total**: ~1 hour hands-on + ~30 min verification + continuous CI enforcement

---

## SUCCESS CRITERIA

After refactoring, these MUST be true:

✅ **Single Source Tree**
- One foodspec/__init__.py at src/foodspec/
- One pyproject.toml at ./
- No foodspec_rewrite/ directory
- All imports from src/foodspec/

✅ **Protocol-Driven Architecture**
- ProtocolV2 in src/foodspec/core/protocol.py
- ComponentRegistry in src/foodspec/core/registry.py
- ExecutionEngine in src/foodspec/core/orchestrator.py

✅ **One-Command Full Run**
- `foodspec run protocol.yaml --output-dir ./run` completes
- All stages execute: data → preprocess → qc → features → model → evaluate → trust → report
- Produces: manifest.json, metrics.json, predictions.json, trust results

✅ **No Regressions**
- All existing tests still pass
- Architecture tests pass
- CI enforcement passes
- Minimal E2E test passes

✅ **Clean Git History**
- Used git mv (history preserved)
- Clear commit messages
- Rollback-able at any point

---

## NEXT STEPS

1. **Review this plan** with team
2. **Create safety branch** (Phase 0)
3. **Execute phases 1-4** (30-40 minutes)
4. **Run verification** (30 minutes)
5. **Enable CI enforcement** (automated from then on)
6. **Update documentation** for users
7. **Tag release** v1.1.0 (or v2.0.0 if major change)

---

## QUESTIONS?

- **What if Phase 1 fails?** → Rollback: `git reset --hard $(cat /tmp/refactor_start.txt)`
- **Do I lose git history?** → No, used `git mv` to preserve history
- **What about existing forks?** → They'll need to rebase on refactored main
- **When should users upgrade?** → Only after complete verification in Phase 5
- **How do I know it worked?** → Run `python scripts/validate_architecture.py --strict` and `pytest tests/test_architecture.py`

---

**END OF REFACTOR PLAN**

**Prepared By**: Strict Refactor Engineer  
**Date**: January 25, 2026  
**Status**: Ready for Execution
