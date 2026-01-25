# Refactor Quick Start Guide

**TL;DR**: Execute this to consolidate FoodSpec to single source tree (30 min + verification)

---

## PREREQUISITES ✅

- [ ] In `/home/cs/FoodSpec` directory
- [ ] Git repo with no uncommitted changes
- [ ] Python 3.10+ available
- [ ] Read REFACTOR_EXECUTION_PLAN.md first

---

## SAFETY FIRST (Do This First!)

```bash
# Create backup branch
git checkout -b backup/pre-refactor-$(date +%Y%m%d-%H%M%S)
git push origin backup/pre-refactor-$(date +%Y%m%d-%H%M%S)

# Create working branch
git checkout phase-1/protocol-driven-core
git checkout -b refactor/single-source-tree

# Record starting state
git log --oneline -1 > /tmp/refactor_start.txt
echo "Backup created at backup/pre-refactor-*"
```

---

## PHASE 1: Preview Changes (2 min)

```bash
# Preview what Phase 1 will do
python scripts/refactor_executor.py --phase 1 --dry-run

# Should show:
# [Preview] MOVE: foodspec_rewrite/foodspec/core/protocol.py → src/foodspec/core/protocol.py
# ... (other moves)
# [Preview] REMOVE: foodspec_rewrite/
```

---

## PHASE 1: Execute (5 min)

```bash
# Execute Phase 1
python scripts/refactor_executor.py \
  --phase 1 \
  --execute \
  --manifest-output /tmp/manifest_phase1.json

# Commit
git add -A
git commit -m "refactor: consolidate to single source tree (src/foodspec/)"

# Verify imports work
python -c "from foodspec.core.protocol import ProtocolV2; print('✓')"
python -c "from foodspec.core.registry import ComponentRegistry; print('✓')"
python -c "from foodspec.core.orchestrator import ExecutionEngine; print('✓')"
```

---

## PHASE 2: Consolidate Configs (2 min)

```bash
# Execute Phase 2
python scripts/refactor_executor.py \
  --phase 2 \
  --execute \
  --manifest-output /tmp/manifest_phase2.json

# Commit
git add pyproject.toml
git commit -m "build: consolidate to single pyproject.toml, version 1.1.0"

# Verify
grep "version = " ./pyproject.toml | head -1
```

---

## PHASE 3: Archive & Clean (2 min)

```bash
# Execute Phase 3
python scripts/refactor_executor.py \
  --phase 3 \
  --execute

# Commit
git add .gitignore _internal/
git commit -m "docs: archive internal docs and clean build artifacts"
```

---

## PHASE 4: Reorganize Examples (2 min)

```bash
# Execute Phase 4
python scripts/refactor_executor.py \
  --phase 4 \
  --execute

# Commit
git add examples/ scripts/
git commit -m "refactor: reorganize examples by use case"
```

---

## PHASE 5: Verify Everything Works (15 min)

```bash
# 1. Run validation script
python scripts/validate_architecture.py --strict

# Should show all checks ✓

# 2. Install and test imports
pip install -e . --no-deps 2>&1 | grep -E "Successfully|error"
pip install pytest 2>&1 | tail -2

# 3. Run architecture tests
pytest tests/test_architecture.py -v

# Should see: ===== 20 passed =====

# 4. Run CI tests
pytest tests/test_architecture_ci.py -v --tb=short

# 5. Quick end-to-end test (optional, slow)
foodspec run examples/protocols/test_minimal.yaml \
  --output-dir ./test_run_verify \
  --no-viz --no-report

test -f ./test_run_verify/manifest.json && echo "✓ E2E passed"
```

---

## PUSH & ENABLE CI (2 min)

```bash
# Push refactored branch
git push origin refactor/single-source-tree

# Create PR on GitHub
# Watch .github/workflows/architecture-enforce.yml run automatically

# Once CI passes, merge to phase-1/protocol-driven-core
git checkout phase-1/protocol-driven-core
git merge refactor/single-source-tree
git push origin phase-1/protocol-driven-core
```

---

## TOTAL TIME: ~30 minutes (execution) + 15 min (verification)

---

## IF SOMETHING BREAKS 🔴

```bash
# Option A: Reset to before refactor
git reset --hard $(cat /tmp/refactor_start.txt)
git checkout phase-1/protocol-driven-core

# Option B: Reset to specific phase
git reset --hard HEAD~5  # Adjust number

# Option C: Restore from backup branch
git checkout backup/pre-refactor-YYYYMMDD-HHMMSS
```

---

## AFTER REFACTORING

Users will run:
```bash
foodspec run protocol.yaml --output-dir ./my_run
```

Instead of separate commands. Perfect! ✅

---

**Questions?** See REFACTOR_EXECUTION_PLAN.md for details.

**Current Status?** Run `python scripts/validate_architecture.py` anytime.
