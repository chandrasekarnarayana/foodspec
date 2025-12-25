# PHASE 0 — REPOSITORY AUDIT TABLE

**Date:** December 25, 2025  
**Auditor:** Senior Python OSS Maintainer  
**Scope:** Complete FoodSpec repository structure, files, and organization

---

## Executive Summary

- **Total Python files in src/:** 193
- **Total Markdown files:** 179
- **__pycache__ directories:** 26
- **Test files:** 157
- **Files > 600 lines:** 2 (core/api.py: 986, features/rq.py: 871)
- **Generated artifacts:** site/ (15MB), .pytest_cache/ (32KB), examples/sample_runs/ (208KB)
- **Documentation streams:** 3 (root *.md, docs/, docs/archive/)

---

## AUDIT TABLE

### A. Generated/Artifact Files (TO CLEAN)

| Path | Type | Size | Risk | Action | Reason |
|------|------|------|------|--------|--------|
| `site/` | Generated (mkdocs) | 15MB | LOW | DELETE + .gitignore | Build output, regenerable |
| `.pytest_cache/` | Generated (pytest) | 32KB | LOW | DELETE + .gitignore | Test cache, regenerable |
| `**/__pycache__/` | Generated (Python) | Various | LOW | DELETE + .gitignore | Bytecode cache |
| `examples/sample_runs/` | Artifact | 208KB | MEDIUM | KEEP | Sample data for tests/docs |

**Action Required:** Remove generated files, enhance .gitignore

---

### B. Documentation Files (CONSOLIDATION NEEDED)

#### Root-Level Documentation

| Path | Type | Duplicated? | Used-by | Risk | Action |
|------|------|-------------|---------|------|--------|
| `README.md` | Primary | No | GitHub, docs index | HIGH | **KEEP** - Entry point |
| `CHANGELOG.md` | Primary | No | Release process | HIGH | **KEEP** - Version history |
| `CONTRIBUTING.md` | Contributor | YES (docs/) | GitHub, contributors | MEDIUM | **KEEP ROOT** - Delete docs/CONTRIBUTING.md |
| `CODE_OF_CONDUCT.md` | Policy | No | GitHub | HIGH | **KEEP** - Community policy |
| `LICENSE` | Legal | No | GitHub, pypi | HIGH | **KEEP** - Legal requirement |
| `RELEASING.md` | Process | No | Maintainers | LOW | **MOVE** → docs/06-developer-guide/ |
| `RELEASE_CHECKLIST.md` | Process | No | Maintainers | LOW | **MOVE** → docs/06-developer-guide/ |
| `FEATURE_AUDIT.md` | Audit | No | Current session | LOW | **ARCHIVE** → docs/archive/ |
| `IMPLEMENTATION_AUDIT.md` | Audit | No | Historical | LOW | **ARCHIVE** → docs/archive/ |
| `MOATS_IMPLEMENTATION.md` | Implementation | No | Design docs | MEDIUM | **MOVE** → docs/05-advanced-topics/ |
| `PHASE0_DISCOVERY_REPORT.md` | Audit | No | Historical | LOW | **ARCHIVE** → docs/archive/ |
| `PROJECT_STRUCTURE_AUDIT.md` | Audit | No | Historical | LOW | **ARCHIVE** → docs/archive/ |
| `CODEBASE_STATUS_SUMMARY.md` | Status | No | Current session | LOW | **ARCHIVE** → docs/archive/ |
| `AUDIT_DOCUMENTATION_INDEX.md` | Index | No | Historical | LOW | **ARCHIVE** → docs/archive/ |
| `CLI_REFACTORING_COMPLETE.md` | Report | No | Historical | LOW | **ARCHIVE** → docs/archive/ |
| `REFACTORING_PLAN.md` | Plan | No | Current refactor | MEDIUM | **KEEP** - Active work |

**Count:** 16 root markdown files  
**Action Required:** Move 8 to docs/, keep 8 at root

#### docs/ Directory Structure

| Path | Type | Purpose | Status | Action |
|------|------|---------|--------|--------|
| `docs/index.md` | Entry | Main docs landing | EXISTS | **KEEP** - Update nav |
| `docs/01-getting-started/` | Tutorial | User onboarding | EXISTS | **KEEP** - Verify complete |
| `docs/02-tutorials/` | Tutorial | Hands-on guides | EXISTS | **KEEP** - Verify complete |
| `docs/03-cookbook/` | Recipes | Solution patterns | EXISTS | **KEEP** - Verify complete |
| `docs/04-user-guide/` | Guide | Feature docs | EXISTS | **KEEP** - Verify complete |
| `docs/05-advanced-topics/` | Advanced | Deep dives | EXISTS | **KEEP** - Add MOATS_IMPLEMENTATION |
| `docs/06-developer-guide/` | Developer | Contributing | EXISTS | **KEEP** - Add RELEASING |
| `docs/07-theory-and-background/` | Theory | Scientific basis | EXISTS | **KEEP** - Verify complete |
| `docs/api/` | API | Auto-generated | EXISTS | **KEEP** - Regenerate |
| `docs/archive/` | Archive | Old docs | EXISTS | **KEEP** - Add ARCHIVED banners |
| `docs/assets/` | Media | Images/CSS | EXISTS | **KEEP** - Verify usage |
| `docs/CONTRIBUTING.md` | Duplicate | Redundant | EXISTS | **DELETE** - Use root version |

**Documentation Streams:** Currently 3 (root, docs/, docs/archive) → Target: 1 (docs/ with curated root)

---

### C. Source Code Organization

#### src/foodspec/ Module Structure

| Module | Files | Purpose | Issues | Risk | Action |
|--------|-------|---------|--------|------|--------|
| `cli/` | 9 | CLI interface | ✅ Recently refactored | LOW | **VERIFY** imports |
| `cli/main_old.py` | 1 | Backup | Obsolete backup | LOW | **DELETE** - No longer needed |
| `core/` | 9 | Core data structures | api.py > 600 lines | HIGH | **SPLIT** api.py |
| `features/` | 5 | Feature extraction | rq.py > 600 lines | HIGH | **SPLIT** rq.py |
| `io/` | 10 | Data I/O | Well organized | LOW | **KEEP** |
| `preprocess/` | 11 | Preprocessing | Well organized | LOW | **KEEP** |
| `chemometrics/` | 8 | Chemometric models | Well organized | LOW | **KEEP** |
| `ml/` | 7 | ML workflows | Well organized | LOW | **KEEP** |
| `qc/` | 5 | Quality control | Well organized | LOW | **KEEP** |
| `stats/` | 6 | Statistics | Well organized | LOW | **KEEP** |
| `viz/` | 6 | Visualization | Well organized | LOW | **KEEP** |
| `workflows/` | 5 | Workflow orchestration | Well organized | LOW | **KEEP** |
| `apps/` | 6 | Application templates | Well organized | LOW | **KEEP** |
| `deploy/` | 3 | Deployment | Well organized | LOW | **KEEP** |
| `repro/` | 3 | Reproducibility | Well organized | LOW | **KEEP** |
| `protocol/` | 3 | Protocol engine | Well organized | LOW | **KEEP** |
| `exp/` | 2 | Experiment runner | Well organized | LOW | **KEEP** |
| `hyperspectral/` | 3 | HSI processing | Well organized | LOW | **KEEP** |
| `data/` | 2 | Data loading | Well organized | LOW | **KEEP** |
| `synthetic/` | 2 | Synthetic data | Well organized | LOW | **KEEP** |
| `gui/` | 1 | GUI support | Minimal | LOW | **KEEP** |
| `plugins/` | 1 | Plugin system | Minimal | LOW | **KEEP** |
| `predict/` | 1 | Prediction | Minimal | LOW | **KEEP** |
| `report/` | 1 | Reporting | Minimal | LOW | **KEEP** |
| `utils/` | 1 | Utilities | Well organized | LOW | **KEEP** |

**Total Modules:** 25  
**Well Organized:** 23  
**Need Attention:** 2 (cli/main_old.py deletion, split 2 large files)

---

### D. Test Organization

#### Test Directory Structure

| Path | Purpose | Files | Issues | Risk | Action |
|------|---------|-------|--------|------|--------|
| `tests/apps/` | App tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/chemometrics/` | Chemometric tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/core/` | Core tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/io_tests/` | I/O tests | Multiple | Name inconsistent | LOW | **RENAME** → tests/io/ |
| `tests/data_tests/` | Data tests | Multiple | Name inconsistent | LOW | **RENAME** → tests/data/ |
| `tests/test_chemometrics/` | Chemometric tests | Multiple | DUPLICATE structure | MEDIUM | **CONSOLIDATE** w/ tests/chemometrics/ |
| `tests/test_deploy/` | Deploy tests | Multiple | DUPLICATE structure | MEDIUM | **CONSOLIDATE** w/ tests/deploy/ |
| `tests/preprocess/` | Preprocess tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/ml/` | ML tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/qc/` | QC tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/stats/` | Stats tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/viz/` | Viz tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/workflows/` | Workflow tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/deploy/` | Deploy tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/repro/` | Repro tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/exp/` | Exp tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/features/` | Feature tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/gui/` | GUI tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/hyperspectral/` | HSI tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/plugins/` | Plugin tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/predict/` | Predict tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/synthetic/` | Synthetic tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |
| `tests/utils/` | Utility tests | Multiple | ✅ Mirrors src | LOW | **KEEP** |

**Total Test Directories:** 24  
**Well Organized:** 20  
**Need Renaming:** 2 (io_tests, data_tests)  
**Need Consolidation:** 2 (test_chemometrics, test_deploy)

---

### E. Files Exceeding 600 Lines (SPLIT REQUIRED)

| File | Lines | Purpose | Split Strategy | Risk |
|------|-------|---------|----------------|------|
| `src/foodspec/core/api.py` | 986 | FoodSpec main API | Split into: api.py (150), api_io.py (200), api_preprocess.py (200), api_modeling.py (250), api_workflows.py (150) | HIGH |
| `src/foodspec/features/rq.py` | 871 | Ratio Quality engine | Convert to package: rq/__init__.py (50), engine.py (300), analysis.py (250), reporting.py (200), types.py (50) | HIGH |

**Action Required:** Both files need careful splitting with backward-compatible imports

---

## DUPLICATION & CONFUSION ANALYSIS

### 1. Documentation Duplication

**Issue:** `CONTRIBUTING.md` exists in both root and docs/  
**Resolution:** Keep root version (GitHub displays it), delete docs/CONTRIBUTING.md  
**Risk:** LOW - docs/ version is likely symlink or copy

### 2. Test Directory Naming Inconsistency

**Issue:** Mixed patterns: `tests/io/` vs `tests/io_tests/`, `tests/chemometrics/` vs `tests/test_chemometrics/`  
**Resolution:** Standardize on `tests/<module>/` pattern  
**Risk:** MEDIUM - May require test configuration updates

### 3. Archived Documentation Visibility

**Issue:** `docs/archive/` exists but files lack "ARCHIVED" warnings  
**Resolution:** Add visible banner to all archived docs  
**Risk:** LOW - Clarification only

---

## PRIORITY ACTIONS BY PHASE

### PHASE 1 — Safe Cleanup (Risk: LOW)
1. ✅ Delete `site/` directory (15MB mkdocs output)
2. ✅ Delete `.pytest_cache/` directory
3. ✅ Delete all `__pycache__/` directories (26 total)
4. ✅ Delete `src/foodspec/cli/main_old.py` (obsolete backup)
5. ✅ Update `.gitignore` to prevent regeneration

**Estimated Time:** 5 minutes  
**Verification:** Check git status, run pytest collection

---

### PHASE 2 — Docs Reorganization (Risk: MEDIUM)
1. Move to docs/06-developer-guide/:
   - RELEASING.md
   - RELEASE_CHECKLIST.md
2. Move to docs/05-advanced-topics/:
   - MOATS_IMPLEMENTATION.md
3. Move to docs/archive/:
   - FEATURE_AUDIT.md
   - IMPLEMENTATION_AUDIT.md
   - PHASE0_DISCOVERY_REPORT.md
   - PROJECT_STRUCTURE_AUDIT.md
   - CODEBASE_STATUS_SUMMARY.md
   - AUDIT_DOCUMENTATION_INDEX.md
   - CLI_REFACTORING_COMPLETE.md
4. Delete: docs/CONTRIBUTING.md (use root)
5. Add "ARCHIVED" banners to all docs/archive/* files
6. Create docs/README_DOCS_STRUCTURE.md
7. Update mkdocs.yml navigation

**Estimated Time:** 20 minutes  
**Verification:** mkdocs build, check nav links

---

### PHASE 3 — Code Reorganization (Risk: LOW)
1. Rename tests/io_tests/ → tests/io/
2. Rename tests/data_tests/ → tests/data/
3. Consolidate tests/test_chemometrics/ → tests/chemometrics/
4. Consolidate tests/test_deploy/ → tests/deploy/
5. Verify all import paths still work
6. Update __init__.py re-exports as needed

**Estimated Time:** 15 minutes  
**Verification:** pytest -q, import checks

---

### PHASE 4 — 600-Line Rule (Risk: HIGH)
1. Split core/api.py (986 lines) into 5 modules:
   - Analyze class structure and method groupings
   - Create submodules with clear responsibilities
   - Implement with backward-compatible imports in api.py
   - Add comprehensive docstrings
   
2. Split features/rq.py (871 lines) into package:
   - Convert file to directory rq/
   - Split into engine, analysis, reporting, types
   - Re-export from rq/__init__.py for compatibility
   - Add comprehensive docstrings

**Estimated Time:** 2 hours  
**Verification:** pytest -q, import tests, CLI smoke tests

---

### PHASE 5 — Test Organization (Risk: LOW)
1. Verify tests/ mirrors src/ structure
2. Ensure all test files follow test_*.py convention
3. Clean up any remaining test artifacts
4. Verify pytest discovery: `pytest --collect-only`

**Estimated Time:** 10 minutes  
**Verification:** pytest collection count matches expectations

---

### PHASE 6 — Verification (Risk: MEDIUM)
1. Create docs/SMOKE_TEST.md with 5 essential commands
2. Run full test suite: `pytest -q`
3. Build documentation: `mkdocs build`
4. Test CLI: `foodspec --help` and all subcommands
5. Test installation: `pip install -e .`
6. Verify imports: Test key public APIs

**Estimated Time:** 30 minutes  
**Verification:** All checks pass

---

## RISK MATRIX

| Risk Level | Count | Description |
|------------|-------|-------------|
| **HIGH** | 2 | File splits (api.py, rq.py) - require careful refactoring |
| **MEDIUM** | 5 | Test consolidation, docs moves - may break references |
| **LOW** | 18 | Generated file deletion, renames - easily reversible |

---

## GITIGNORE ADDITIONS NEEDED

```gitignore
# Build outputs
site/
dist/
build/
*.egg-info/

# Test artifacts
.pytest_cache/
.coverage
htmlcov/
.tox/

# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so

# Runtime outputs
foodspec_runs/
protocol_runs_test/
*_demo_output/
examples/sample_runs/*.h5
examples/sample_runs/*.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
```

---

## SUCCESS CRITERIA

After all phases complete, the repository should have:

1. ✅ **Zero generated files tracked** in git
2. ✅ **Single documentation source** (docs/) with clear structure
3. ✅ **No Python files > 600 lines** in src/
4. ✅ **Consistent test organization** mirroring src/
5. ✅ **Clean .gitignore** preventing artifact commits
6. ✅ **All tests passing** (pytest)
7. ✅ **Documentation builds** (mkdocs)
8. ✅ **CLI functional** (all commands work)
9. ✅ **API stability preserved** (public imports unchanged)
10. ✅ **Clear smoke test procedure** documented

---

## NEXT STEPS

**Ready to proceed with PHASE 1 — Safe Cleanup**

Awaiting approval to begin cleanup operations. Recommend reviewing this audit table before proceeding to ensure alignment with project goals.
