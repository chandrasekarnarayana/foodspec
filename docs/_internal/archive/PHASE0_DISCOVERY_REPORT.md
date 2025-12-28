---
**🗄️ ARCHIVED DOCUMENT**

This document is archived for historical reference and is no longer actively maintained. 
For current documentation, see [docs/README_DOCS_STRUCTURE.md](README_DOCS_STRUCTURE.md).

---

# PHASE 0 — DISCOVERY REPORT
**Date:** December 25, 2025  
**Engineer:** Lead Maintainer & Release Engineer

---

## EXECUTIVE SUMMARY

FoodSpec has **161 Python source files** organized into **24 top-level modules**. The codebase is moderately organized with a good test structure (152 test files in 20 directories). However, there are **CRITICAL ISSUES** requiring immediate attention to meet Ultralytics-level standards:

### 🔴 **CRITICAL ISSUES**

1. **4 FILES EXCEED 600 LINES** (violates rule #1)
   - `cli.py` (1,175 lines) - MASSIVE monolithic CLI
   - `core/api.py` (986 lines) - God object with too many responsibilities
   - `rq.py` (871 lines) - Ratio-Quality engine with multiple concerns
   - `protocol_engine.py` (743 lines) - Protocol execution with schema + runner + steps

2. **SCATTERED CLI FILES** (violates rule #7)
   - Main: `cli.py` (1,175 lines)
   - Scattered: `cli_library_search.py`, `cli_predict.py`, `cli_protocol.py`, `cli_publish.py`, `cli_plugin.py`, `cli_registry.py`
   - No centralized `src/foodspec/cli/` directory

3. **INCONSISTENT TOP-LEVEL STRUCTURE**
   - 30+ files directly in `src/foodspec/` (should be in submodules)
   - Mix of CLIs, workflows, engines, and utilities at root level
   - Unclear separation of concerns

4. **NAMING CONFUSION**
   - `engine.py` exists in multiple places (preprocess/engine.py, qc/engine.py, protocol_engine.py at root)
   - `rq.py` vs `features/ratios.py` - unclear distinction
   - Workflow files scattered at root vs `workflows/` directory

---

## DETAILED FINDINGS

### A) FILES > 600 LINES (MUST SPLIT)

| File | Lines | Issues | Proposed Split |
|------|-------|--------|----------------|
| **cli.py** | 1,175 | Monolithic CLI with 20+ commands, mixing concerns | Split into `cli/` module with subcommands |
| **core/api.py** | 986 | FoodSpec class doing everything (ingestion, preprocess, QC, RQ, export) | Split: core/dataset.py, core/preprocessing.py, core/qc_integration.py, core/rq_integration.py |
| **rq.py** | 871 | RatioQualityEngine with peak extraction, stability, discriminative analysis, clustering | Split: rq/engine.py, rq/stability.py, rq/discrimination.py, rq/clustering.py |
| **protocol_engine.py** | 743 | Protocol schema validation, step execution, runner logic all mixed | Split: protocol/engine.py, protocol/schema.py, protocol/runner.py, protocol/steps/ |

### B) SCATTERED TOP-LEVEL FILES (SHOULD MOVE)

**Current root files that should move:**
```plaintext
src/foodspec/
├── cli.py (1175) → cli/main.py + cli/preprocess.py + cli/qc.py etc.
├── cli_*.py (7 files) → cli/
├── protocol_engine.py → protocol/engine.py
├── artifact.py → deploy/artifact.py
├── output_bundle.py → repro/output_bundle.py
├── rq.py → features/rq/engine.py (or keep at root but split)
├── library_search.py → features/library_search.py
├── calibration_transfer.py → workflows/calibration_transfer.py
├── harmonization.py → workflows/harmonization.py
├── heating_trajectory.py → workflows/heating_trajectory.py
├── matrix_correction.py → workflows/matrix_correction.py
├── validation.py → utils/validation.py
├── config.py → (keep at root)
├── registry.py → plugins/registry.py
├── logging_utils.py → utils/logging.py
├── logo.py → utils/logo.py
└── spectral_io.py → io/high_level.py
```

### C) MODULE STRUCTURE ALIGNMENT

✅ **GOOD STRUCTURE** (already organized):
- `apps/` - Domain applications (dairy, heating, oils, qc)
- `chemometrics/` - Multivariate methods
- `core/` - Core data structures (but api.py too large)
- `features/` - Feature extraction
- `hyperspectral/` - HSI support
- `io/` - Import/export
- `ml/` - Machine learning
- `preprocess/` - Preprocessing
- `qc/` - Quality control
- `stats/` - Statistical analysis
- `viz/` - Visualization
- `workflows/` - End-to-end workflows

⚠️ **NEEDS ORGANIZATION**:
- No `cli/` directory - CLIs scattered at root
- No `protocol/` directory - protocol_engine.py at root
- `deploy/` exists but empty - artifact.py at root
- `repro/` exists but output_bundle.py at root
- Many workflow files at root should be in `workflows/`

### D) PUBLIC API ENTRY POINTS

**Current exports in `__init__.py`** (149 lines):
- Phase 1: FoodSpec, Spectrum, RunRecord, OutputBundle
- Phase 0: 40+ legacy exports
- Issue: Too many exports, unclear API surface

**Current exports in `core/api.py`**:
- FoodSpec class (986 lines) - MONOLITHIC

### E) PYTEST & COVERAGE

**Current setup:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "--cov=foodspec --cov-report=term-missing --cov-fail-under=75"

[tool.coverage.run]
omit = [
  "src/foodspec/cli.py",  # Excluded from coverage!
  "src/foodspec/preprocess/derivatives.py",
]
```

⚠️ **Issues:**
- `cli.py` excluded from coverage requirement (75% minimum)
- No coverage for CLI means untested commands
- After splitting, all new CLI modules MUST be tested

### F) DOCS STRUCTURE

**Organization:**
```plaintext
docs/
├── 01-getting-started/
├── 02-tutorials/
├── 03-cookbook/
├── 04-user-guide/
├── 05-advanced-topics/
├── 06-developer-guide/
├── 07-theory-and-background/
├── archive/           # ← Good: has archive folder
├── examples/
└── ... (20+ subdirectories)
```

**Potential issues (need inspection):**
- Multiple overlapping directories (user_guide/, 04-user-guide/)
- Archive may have outdated content that should be deleted
- Need to verify CLI docs match actual commands
- Need to check for broken cross-references after refactor

---

## PROPOSED PLAN

### PHASE 1: PROFESSIONAL RESTRUCTURE (2-3 days)

#### Step 1.1: Create CLI Module Structure
```plaintext
src/foodspec/cli/
├── __init__.py        # Main CLI entry point (typer app)
├── main.py            # foodspec main command
├── preprocess.py      # foodspec preprocess
├── qc.py              # foodspec qc
├── predict.py         # foodspec predict
├── protocol.py        # foodspec protocol
├── library.py         # foodspec library-search
├── publish.py         # foodspec publish
├── plugin.py          # foodspec plugin
├── registry.py        # foodspec registry
└── benchmark.py       # NEW: foodspec benchmark
```

**Migration:**
- Move `cli.py` → split into cli/*.py (preserve all commands)
- Move `cli_*.py` → cli/*.py (consolidate)
- Create backward-compat shim: `cli.py` imports from `cli/main.py`
- Update all imports
- Add tests for each CLI module

#### Step 1.2: Create Protocol Module Structure
```plaintext
src/foodspec/protocol/
├── __init__.py
├── engine.py          # ProtocolEngine class
├── schema.py          # Protocol schema validation
├── runner.py          # Protocol execution runner
├── steps/
│   ├── __init__.py
│   ├── base.py        # Base step classes
│   ├── preprocess.py  # Preprocessing steps
│   ├── qc.py          # QC steps
│   ├── modeling.py    # Modeling steps
│   └── rq.py          # RQ analysis steps
```yaml

**Migration:**
- Split `protocol_engine.py` → protocol/*.py
- Preserve public API: import ProtocolEngine from protocol module
- Update tests
- Add step registry and plugin support

#### Step 1.3: Reorganize Top-Level Files
```
MOVE:
artifact.py → deploy/artifact.py
output_bundle.py → repro/output_bundle.py
calibration_transfer.py → workflows/calibration_transfer.py
harmonization.py → workflows/harmonization.py
heating_trajectory.py → workflows/heating_trajectory.py
matrix_correction.py → workflows/matrix_correction.py
library_search.py → features/library_search.py
validation.py → utils/validation.py
logging_utils.py → utils/logging.py
logo.py → utils/logo.py
spectral_io.py → io/high_level.py
registry.py → plugins/registry.py

KEEP AT ROOT (legitimate):
__init__.py
config.py
check_env.py
metrics.py (or move to stats/metrics.py)
```yaml

**For each move:**
- Add backward-compat re-export in original location
- Update all imports
- Update tests
- Update docs references

#### Step 1.4: Update __init__.py Public API
- Remove internal exports
- Keep only high-level user-facing API
- Add deprecation warnings for old import paths
- Document migration path

### PHASE 2: SPLIT LARGE FILES (3-4 days)

#### 2.1: Split cli.py (1,175 lines)
**Current structure analysis needed:**
- Commands: preprocess, qc, predict, protocol, heating, oil-auth, mixture, etc.
- Shared utilities
- Typer app configuration

**Target:** 10-12 files, each <150 lines

#### 2.2: Split core/api.py (986 lines)
**Current structure:**
- FoodSpec class with 20+ methods
- Ingestion, preprocessing, QC, RQ, export logic

**Target split:**
```
core/
├── api.py (< 200 lines) # Main FoodSpec class
├── dataset.py           # Dataset management
├── preprocessing.py     # Preprocessing integration
├── qc_integration.py    # QC integration
├── rq_integration.py    # RQ integration
└── export.py            # Export functionality
```yaml

#### 2.3: Split rq.py (871 lines)
**Target:**
```
features/rq/
├── __init__.py
├── engine.py (< 300 lines)    # Main RQ engine
├── stability.py               # Stability analysis
├── discrimination.py          # Discriminative analysis
├── clustering.py              # Clustering analysis
├── peak_extraction.py         # Peak extraction utilities
└── reporting.py               # Results formatting
```yaml

#### 2.4: Split protocol_engine.py (743 lines)
See Phase 1 Step 1.2 above

### PHASE 3: TRUST LAYER + PAPER READINESS (5-7 days)

#### 3.1: Implement VIP for PLS/PLS-DA
**New file:** `src/foodspec/chemometrics/vip.py`
```
def calculate_vip(pls_model, X, y) -> np.ndarray:
    """Calculate Variable Importance in Projection (VIP) scores.
    
    Parameters
    ----------
    pls_model : PLSRegression or Pipeline
        Fitted PLS model
    X : ndarray (n_samples, n_features)
        Training data
    y : ndarray (n_samples,) or (n_samples, n_targets)
        Training targets
        
    Returns
    -------
    vip_scores : ndarray (n_features,)
        VIP score for each feature (>1 = important)
        
    References
    ----------
    Wold et al. (2001) PLS-regression: a basic tool of chemometrics.
    """
```yaml

**Integration points:**
- chemometrics/models.py - add VIP to PLS output
- report/methods.py - include VIP in reports
- viz/ - add VIP bar plots

**Tests:**
- test_vip_calculation_shape()
- test_vip_sanity_checks()
- test_vip_threshold_interpretation()
- test_vip_integration_with_pls()

#### 3.2: Artifact Version Compatibility
**Enhance:** `src/foodspec/deploy/artifact.py`

Add:
- manifest_schema_version: "1.0"
- foodspec_version: __version__
- python_version: sys.version
- dependencies: {pkg: version}
- artifact_hash: SHA256
- created_timestamp: ISO8601

**Load behavior:**
- Check major version compatibility
- Warn on minor version mismatch
- Migrate on schema version changes
- Fail gracefully with clear error messages

**Tests:**
- test_artifact_version_compatible()
- test_artifact_version_incompatible_major()
- test_artifact_version_missing_fields()
- test_artifact_migration()

#### 3.3: Benchmark & Dataset Contract
**New:** `src/foodspec/benchmark/`
```
benchmark/
├── __init__.py
├── runner.py          # Benchmark execution
├── datasets.py        # Dataset contracts
├── suites/
│   ├── __init__.py
│   ├── preprocessing.py   # Baseline, smoothing benchmarks
│   ├── classification.py  # Oil auth, dairy auth
│   └── regression.py      # Heating quality
└── reporting.py       # Benchmark reports
```yaml

**CLI:** `foodspec benchmark <suite> [--output-dir]`

**Tests:**
- test_benchmark_preprocessing_suite()
- test_benchmark_classification_suite()
- test_dataset_contract_validation()

#### 3.4: Pre-flight Validation
**New:** `src/foodspec/utils/preflight.py`
```
@dataclass
class ValidationReport:
    ok: bool
    issues: List[str]
    warnings: List[str]
    suggested_fixes: List[str]
```yaml

**Integration points:**
- io/ - validate on load
- preprocess/ - check assumptions before transform
- qc/ - validate thresholds
- ml/ - check train data quality
- protocol/ - validate config before run

**Tests:**
- test_preflight_missing_wavenumbers()
- test_preflight_insufficient_samples()
- test_preflight_class_imbalance()
- test_preflight_suggested_fixes()

### PHASE 4: DOCS CLEANUP (2-3 days)

#### 4.1: Audit Docs for Accuracy
- [ ] Check CLI docs match `--help` output
- [ ] Verify all code examples work
- [ ] Update import paths after refactor
- [ ] Remove/archive outdated content

#### 4.2: Create Smoke Test Doc
**New:** `docs/01-getting-started/smoke_test.md`

5 commands that validate installation:
1. `foodspec --version`
2. `foodspec preprocess --example`
3. `foodspec qc --example`
4. `foodspec protocol --list-examples`
5. `foodspec benchmark preprocessing-quick`

#### 4.3: Update Cross-References
- Search and replace old import paths
- Update file structure diagrams
- Fix broken internal links

### PHASE 5: QUALITY GATES & AUDIT TABLE (1-2 days)

#### 5.1: Create Feature Stocktake
**New:** `docs/FEATURE_STOCKTAKE.md`

Columns:
- Feature Name, Feature ID, Module Path
- Public API, CLI Entry, YAML/Config
- Input/Output Types
- Deterministic/Seeded/Hashable
- Assumptions Checked, Failure Modes Documented
- Tests Exist, Example Exists, Docs Page
- Status (Stable/Experimental/Deprecated)

#### 5.2: Add Quality Gates
**New:** `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
```

**Update:** `pyproject.toml`
```toml
[tool.ruff]
line-length = 100
target-version = "py39"
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]  # Line too long (handled by formatter)
```

---

## RISKS & MITIGATION

### HIGH RISK
1. **Breaking changes in refactor**
   - Mitigation: Backward-compat shims, deprecation warnings, thorough testing
   
2. **Import path changes breaking user code**
   - Mitigation: Keep old imports working with deprecation warnings for 2 versions

3. **Test failures after file moves**
   - Mitigation: Fix imports incrementally, run pytest after each batch

### MEDIUM RISK
4. **CLI behavior changes**
   - Mitigation: Preserve exact CLI syntax, add tests for all commands

5. **Performance regression**
   - Mitigation: Benchmark before/after, profile critical paths

### LOW RISK
6. **Documentation outdated**
   - Mitigation: Automated link checking, manual review

---

## SUCCESS CRITERIA

### Phase 1 Complete When:
- [ ] All files moved to correct modules
- [ ] No file >600 lines at src/foodspec/ root
- [ ] CLI in `cli/` directory
- [ ] Protocol in `protocol/` directory
- [ ] All tests pass (577 tests, 0 errors)
- [ ] Backward compatibility maintained

### Phase 2 Complete When:
- [ ] cli.py split into <10 files, each <150 lines
- [ ] core/api.py split into 5-6 files, each <200 lines
- [ ] rq.py split into 6-7 files, each <200 lines
- [ ] protocol_engine.py split (already in Phase 1)
- [ ] All tests pass
- [ ] Coverage maintained >75%

### Phase 3 Complete When:
- [ ] VIP implemented with 4+ tests
- [ ] Artifact versioning enforced with 4+ tests
- [ ] Benchmark suite runs with 3+ tests
- [ ] Pre-flight validation integrated with 4+ tests
- [ ] All integration points tested

### Phase 4 Complete When:
- [ ] 0 broken links in docs
- [ ] CLI docs match --help
- [ ] Smoke test doc validated
- [ ] Old docs archived/removed

### Phase 5 Complete When:
- [ ] Feature stocktake complete (80+ features)
- [ ] pre-commit config working
- [ ] ruff linting passes
- [ ] All quality gates green

---

## ESTIMATED EFFORT

| Phase | Days | Dependencies |
|-------|------|--------------|
| Phase 0 (Discovery) | 0.5 | - |
| Phase 1 (Restructure) | 2-3 | Phase 0 |
| Phase 2 (Split Files) | 3-4 | Phase 1 |
| Phase 3 (Trust Layer) | 5-7 | Phase 2 |
| Phase 4 (Docs) | 2-3 | Phase 1, 2, 3 |
| Phase 5 (Quality) | 1-2 | Phase 4 |
| **TOTAL** | **14-20 days** | Sequential |

---

## NEXT STEPS

**IMMEDIATE (Phase 1 Step 1.1):**
1. Create `src/foodspec/cli/` directory structure
2. Extract commands from `cli.py` into separate files
3. Create backward-compat shim in `cli.py`
4. Run tests
5. Commit

**FOLLOW-ON:**
- Phase 1 Steps 1.2, 1.3, 1.4
- Phase 2 file splits
- Phase 3 feature implementations

---

**DECISION REQUIRED:** Approve this plan before proceeding to Phase 1 implementation.
