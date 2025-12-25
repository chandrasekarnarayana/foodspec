# FoodSpec Repository Refactoring Plan

**Date:** December 25, 2025  
**Status:** Implementation in Progress

---

## Executive Summary

FoodSpec is being refactored into a professional, scalable, release-ready open-source Python package. The codebase already uses src/ layout but needs:
1. Splitting oversized files (>600 lines)
2. Removing generated artifacts from version control
3. Consolidating duplicate documentation
4. Improving test coverage (currently 14% → target 75%)
5. Adding professional tooling (.gitignore, pre-commit, linting)

---

## Current State Inventory

### Large Files Requiring Split (>600 lines)

| File | Lines | Action Required |
|------|-------|-----------------|
| `cli/main.py` | 1175 | Split into subcommands |
| `core/api.py` | 986 | Split into modules (io, preprocess, modeling, workflows) |
| `features/rq.py` | 871 | Split into engine, analysis, reporting |
| `preprocess/matrix_correction.py` | 564 | Keep as-is (under threshold) |
| `preprocess/engine.py` | 553 | Keep as-is |
| `core/dataset.py` | 534 | Keep as-is |
| `workflows/heating_trajectory.py` | 531 | Keep as-is |
| `preprocess/calibration_transfer.py` | 515 | Keep as-is |
| `core/spectral_dataset.py` | 511 | Keep as-is |

**Total files >600 lines:** 3 (cli/main.py, core/api.py, features/rq.py)

### Generated Artifacts (Not in Version Control)

| Directory/File | Size | Action |
|----------------|------|--------|
| `htmlcov/` | 8.1 MB | Add to .gitignore, delete from repo |
| `protocol_runs_test/` | 3.6 MB | Add to .gitignore, move to examples/sample_runs/ (minimal) |
| `moats_demo_output/` | 20 KB | Add to .gitignore, delete |
| `foodspec_runs/` | 4 KB | Add to .gitignore, delete |
| `__pycache__/` | Various | Already in .gitignore, verify cleanup |
| `.pytest_cache/` | Various | Add to .gitignore |
| `site/` | Generated | Add to .gitignore (mkdocs build output) |

### Documentation Structure

- **Total .md files:** 152
- **Root-level docs:** 55
- **Subdirectories:** docs/ has extensive nested structure
- **Issues identified:**
  - Potential duplicates between docs/ and root-level
  - docs/archive/ may contain outdated content
  - Need to verify all docs are referenced in mkdocs.yml

---

## Target Folder Structure

```
FoodSpec/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI
│
├── src/foodspec/
│   ├── __init__.py
│   ├── _version.py                   # Version management
│   │
│   ├── cli/                          # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py                   # ✂️ Split from 1175 → <600 lines
│   │   ├── commands/                 # NEW: Subcommands extracted
│   │   │   ├── __init__.py
│   │   │   ├── preprocess.py
│   │   │   ├── qc.py
│   │   │   ├── model.py
│   │   │   └── workflow.py
│   │   ├── library_search.py
│   │   ├── plugin.py
│   │   ├── predict.py
│   │   ├── protocol.py
│   │   ├── publish.py
│   │   └── registry.py
│   │
│   ├── core/                         # Core data structures
│   │   ├── __init__.py
│   │   ├── api.py                    # ✂️ Split from 986 → <600 lines
│   │   ├── api_io.py                 # NEW: I/O methods extracted
│   │   ├── api_preprocess.py         # NEW: Preprocessing methods
│   │   ├── api_modeling.py           # NEW: Modeling methods
│   │   ├── api_workflows.py          # NEW: Workflow methods
│   │   ├── dataset.py
│   │   ├── spectral_dataset.py
│   │   ├── spectrum.py
│   │   ├── run_record.py
│   │   ├── output_bundle.py
│   │   └── summary.py
│   │
│   ├── features/                     # Feature extraction & RQ
│   │   ├── __init__.py
│   │   ├── rq.py                     # ✂️ Split from 871 → <600 lines
│   │   ├── rq_engine.py              # NEW: Core RQ engine
│   │   ├── rq_analysis.py            # NEW: Statistical analysis
│   │   ├── rq_reporting.py           # NEW: Results formatting
│   │   ├── peaks.py
│   │   ├── ratios.py
│   │   ├── bands.py
│   │   ├── fingerprint.py
│   │   ├── interpretation.py
│   │   ├── library.py
│   │   └── specs.py
│   │
│   ├── preprocess/                   # Preprocessing pipeline
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── baseline.py
│   │   ├── normalization.py
│   │   ├── smoothing.py
│   │   ├── spikes.py
│   │   ├── matrix_correction.py
│   │   ├── calibration_transfer.py
│   │   └── ...
│   │
│   ├── io/                           # Import/export
│   ├── ml/                           # Machine learning
│   ├── chemometrics/                 # Chemometrics methods
│   ├── qc/                           # Quality control
│   ├── metrics/                      # Metrics
│   ├── stats/                        # Statistics
│   ├── viz/                          # Visualization
│   ├── workflows/                    # End-to-end workflows
│   ├── deploy/                       # Deployment utilities
│   ├── apps/                         # Domain applications
│   ├── synthetic/                    # Synthetic data generation
│   ├── utils/                        # Utilities
│   └── _internal/                    # Private helpers (non-public API)
│
├── tests/                            # Test suite (mirrors src structure)
│   ├── conftest.py
│   ├── cli/
│   ├── core/
│   ├── features/
│   ├── preprocess/
│   ├── io_tests/
│   ├── ml/
│   ├── qc/
│   ├── ...
│   └── data_tests/                   # Test fixtures
│
├── docs/                             # Documentation (mkdocs)
│   ├── index.md
│   ├── 01-getting-started/
│   ├── 02-tutorials/
│   ├── 03-cookbook/
│   ├── 04-user-guide/
│   ├── 05-advanced-topics/
│   ├── 06-developer-guide/
│   ├── 07-theory-and-background/
│   ├── api/                          # API reference
│   ├── MIGRATION_GUIDE.md            # ✅ Created
│   └── archive/                      # ⚠️ Add banner, exclude from nav
│
├── examples/                         # Example scripts & notebooks
│   ├── quickstart/
│   ├── protocols/
│   ├── notebooks/
│   ├── sample_runs/                  # NEW: Minimal protocol run examples
│   └── data/
│
├── benchmarks/                       # Performance benchmarks
│
├── tools/                            # Development utilities
│   └── (if any)
│
├── .gitignore                        # ✅ Enhanced
├── .pre-commit-config.yaml          # 🔧 NEW: Pre-commit hooks
├── pyproject.toml                    # ✅ Enhanced with linting config
├── mkdocs.yml                        # ✅ Verify nav
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── FEATURE_AUDIT.md
├── PROJECT_STRUCTURE_AUDIT.md
└── REFACTORING_PLAN.md              # This file
```

---

## File Rename/Move Map

### Split Operations (Files >600 Lines)

#### 1. cli/main.py (1175 lines) → cli/main.py + cli/commands/*.py

```
BEFORE:
src/foodspec/cli/main.py (1175 lines, all commands in one file)

AFTER:
src/foodspec/cli/main.py (~200 lines, main app + imports)
src/foodspec/cli/commands/preprocess.py (~200 lines)
src/foodspec/cli/commands/qc.py (~150 lines)
src/foodspec/cli/commands/model.py (~200 lines)
src/foodspec/cli/commands/workflow.py (~200 lines)
```

#### 2. core/api.py (986 lines) → core/api.py + core/api_*.py

```
BEFORE:
src/foodspec/core/api.py (986 lines, all FoodSpec methods)

AFTER:
src/foodspec/core/api.py (~150 lines, main class + __init__)
src/foodspec/core/api_io.py (~200 lines, load/save methods)
src/foodspec/core/api_preprocess.py (~200 lines, preprocessing methods)
src/foodspec/core/api_modeling.py (~250 lines, modeling methods)
src/foodspec/core/api_workflows.py (~150 lines, workflow methods)
```

#### 3. features/rq.py (871 lines) → features/rq/ package

```
BEFORE:
src/foodspec/features/rq.py (871 lines, all RQ engine code)

AFTER:
src/foodspec/features/rq/__init__.py (~50 lines, re-exports)
src/foodspec/features/rq/engine.py (~300 lines, core RatioQualityEngine)
src/foodspec/features/rq/analysis.py (~250 lines, statistical analysis)
src/foodspec/features/rq/reporting.py (~200 lines, results formatting)
src/foodspec/features/rq/types.py (~50 lines, dataclasses)
```

### Backward Compatibility Shims

```
OLD LOCATION                         → NEW LOCATION (with shim at old)
src/foodspec/features/rq.py          → src/foodspec/features/rq/__init__.py (shim remains)
```

---

## Deletion Plan

### ✅ Safe to Delete (Generated Artifacts)

These should be deleted from version control and added to .gitignore:

```bash
# Coverage reports
rm -rf htmlcov/
rm -f coverage.xml
rm -f .coverage

# pytest cache
rm -rf .pytest_cache/

# mkdocs build output
rm -rf site/

# Runtime outputs (keep minimal examples only)
rm -rf moats_demo_output/
rm -rf foodspec_runs/
mv protocol_runs_test/20251212_042014_run examples/sample_runs/example_oil_auth/
mv protocol_runs_test/20251212_042234_run examples/sample_runs/example_heating/
rm -rf protocol_runs_test/

# Python caches
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Ruff cache
rm -rf .ruff_cache/
```

### ⚠️ Move to Archive (Potentially Outdated Docs)

Need manual review before deletion:

```bash
# Review docs/archive/ for outdated content
# If truly outdated, add banner: "ARCHIVED - Historical documentation only"
# Exclude from mkdocs.yml nav
```

### ❌ Do Not Delete

Keep these files (essential):

- All `.py` source files in `src/foodspec/`
- All test files in `tests/`
- Active documentation in `docs/` (after review)
- Examples in `examples/`
- Configuration files: `pyproject.toml`, `mkdocs.yml`, etc.
- Root-level markdown: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, etc.

---

## Implementation Plan (Step-by-Step)

### ✅ Commit 1: Add Tooling & Cleanup (COMPLETED PARTIALLY)

**Changes:**
- [x] Enhanced .gitignore
- [ ] Add .pre-commit-config.yaml
- [ ] Add ruff config to pyproject.toml
- [ ] Add pytest config (already exists)
- [ ] Delete generated artifacts

**Commands:**
```bash
# Delete artifacts
rm -rf htmlcov/ .pytest_cache/ moats_demo_output/ foodspec_runs/
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .ruff_cache/

# Move protocol runs to examples
mkdir -p examples/sample_runs
mv protocol_runs_test/20251212_042014_run examples/sample_runs/example_oil_auth/
rm -rf protocol_runs_test/
```

### 🔧 Commit 2: Split Oversized Files

**Order:**
1. Split cli/main.py → cli/main.py + cli/commands/*.py
2. Split core/api.py → core/api.py + core/api_*.py
3. Split features/rq.py → features/rq/ package

**Testing after each split:**
```bash
python -m pip install -e .
pytest tests/cli/ -v
pytest tests/core/ -v
pytest tests/features/ -v
```

### 🔧 Commit 3: Documentation Cleanup

**Actions:**
1. Review docs/archive/ - add "ARCHIVED" banner if outdated
2. Remove duplicate docs (keep canonical version only)
3. Update mkdocs.yml nav to exclude archived docs
4. Fix broken links after moves

### 🔧 Commit 4: Add Pre-commit Hooks

**Setup:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

### 🔧 Commit 5: Validation & Testing

**Run:**
```bash
python -m pip install -e .
pytest -q
ruff check .
ruff format --check .
mkdocs build
python -m pytest --cov=src/foodspec --cov-report=term
```

---

## Success Criteria

- ✅ No Python file >600 lines
- ✅ All generated artifacts removed from git and in .gitignore
- ✅ Tests passing (pytest -q)
- ✅ Linting passing (ruff check .)
- ✅ Formatting consistent (ruff format .)
- ✅ Docs buildable (mkdocs build)
- ✅ Examples runnable
- ✅ Coverage >20% (incremental improvement toward 75%)
- ✅ Backward compatibility maintained (shims working)

---

## Risk Mitigation

1. **Breaking Changes:** All file moves use backward-compatible shims
2. **Test Failures:** Run tests after each commit
3. **Documentation Breakage:** Validate mkdocs build after doc changes
4. **Import Errors:** Test import paths after splits

---

## Timeline

- **Phase 1 (Tooling):** 30 minutes
- **Phase 2 (File Splits):** 2-3 hours
- **Phase 3 (Doc Cleanup):** 1 hour
- **Phase 4 (Validation):** 30 minutes

**Total:** ~4-5 hours

---

## Next Steps

1. Implement Commit 1 (tooling + cleanup)
2. Implement Commit 2 (split oversized files)
3. Run validation suite
4. Update FEATURE_AUDIT.md with completion status
