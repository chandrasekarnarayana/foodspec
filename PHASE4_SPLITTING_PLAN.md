# PHASE 4: File Splitting Plan (600-Line Rule)

**Status:** In Progress  
**Date:** December 25, 2025  
**Risk Level:** HIGH

---

## Overview

Two files exceed 600 lines and require refactoring:
- `src/foodspec/core/api.py`: 986 lines (single FoodSpec class)
- `src/foodspec/features/rq.py`: 871 lines (RQ engine with dataclasses + engine class)

---

## Strategy: Mixin-Based Refactoring (Lowest Risk)

Instead of fully splitting into separate files (which risks breaking imports), we use **mixin classes** to organize code while maintaining a single entry point. This ensures:
- ✅ Backward compatibility preserved
- ✅ All existing imports continue to work
- ✅ IDE autocomplete/type hints unchanged
- ✅ Tests don't need modification
- ✅ Easy to revert if issues arise

---

## Plan for core/api.py (986 lines → 5 modules)

### Method Distribution

**Total methods:** 19 methods in FoodSpec class

**Logical grouping:**

1. **api.py** (main file, ~150 lines) - Core orchestration
   - `__init__` (lines 64-115)
   - `summary()` (lines 518-534)
   - `__repr__()` (lines 536-545)
   - Imports from all mixins

2. **api_io.py** (~120 lines) - Data loading
   - `_load_data()` (lines 118-157)
   - `_dataframe_to_spectrum_set()` (lines 503-516)

3. **api_preprocess.py** (~190 lines) - Preprocessing
   - `qc()` (lines 159-206)
   - `preprocess()` (lines 208-291)
   - `apply_matrix_correction()` (lines 547-606)
   - `apply_calibration_transfer()` (lines 688-758)

4. **api_modeling.py** (~220 lines) - Modeling
   - `features()` (lines 293-349)
   - `train()` (lines 351-409)
   - `library_similarity()` (lines 411-478)

5. **api_workflows.py** (~200 lines) - Analysis workflows
   - `analyze_heating_trajectory()` (lines 608-686)
   - `export()` (lines 480-501)

6. **api_diagnostics.py** (~180 lines) - Dataset diagnostics
   - `summarize_dataset()` (lines 760-799)
   - `check_class_balance()` (lines 801-841)
   - `assess_replicate_consistency()` (lines 843-879)
   - `detect_leakage()` (lines 881-930)
   - `compute_readiness_score()` (lines 932-987)

### Implementation Steps

1. Create 5 new mixin files with extracted methods
2. Have each mixin inherit necessary dependencies
3. Modify api.py to inherit from all mixins
4. Add re-export in `__init__.py` for backward compatibility
5. Run full test suite
6. Verify all CLI commands work
7. Check API documentation builds

---

## Plan for features/rq.py (871 lines → 4 modules)

### Current structure

**Dataclasses** (lines 1-100):
- PeakDefinition
- RatioDefinition
- RQConfig
- RatioQualityResult

**RatioQualityEngine class** (lines 101-871):
- ~25 methods handling computation, analysis, reporting

### Refactoring approach

Create a package: `features/rq/`

1. **`__init__.py`** (~50 lines)
   - Re-export all public API
   - Backward compatibility

2. **types.py** (~80 lines)
   - All dataclasses (PeakDefinition, RatioDefinition, RQConfig, RatioQualityResult)
   - Type definitions

3. **engine.py** (~300 lines)
   - RatioQualityEngine class core
   - Main computation methods

4. **analysis.py** (~250 lines)
   - Statistical analysis methods
   - Discriminative power, stability analysis

5. **reporting.py** (~150 lines)
   - Text report generation
   - Visualization helpers
   - Export functions

---

## Risk Mitigation

### Before Starting

- [x] Full test suite passes (642 tests)
- [ ] Create git branch for PHASE4 work
- [ ] Document current test coverage baseline

### During Refactoring

- [ ] Refactor one file at a time (api.py first, then rq.py)
- [ ] Run tests after each module creation
- [ ] Verify CLI still works
- [ ] Check import patterns don't break

### After Completion

- [ ] Full pytest run (all 642 tests must pass)
- [ ] CLI smoke test: `foodspec --help` and all subcommands
- [ ] Import test: verify all public API imports work
- [ ] Documentation build: `mkdocs build`
- [ ] Create comprehensive commit message with rollback instructions

---

## Testing Checklist

### Core Tests (api.py)

```bash
# All core tests
pytest tests/core/ -v

# API-specific tests
pytest tests/core/test_api.py -v
pytest tests/core/test_api_extended.py -v
pytest tests/core/test_foodspec_integration.py -v

# Integration tests using FoodSpec class
pytest tests/workflows/ -v
```

### RQ Tests (rq.py)

```bash
# RQ engine tests
pytest tests/features/test_rq*.py -v

# CLI that uses RQ
foodspec analysis rq-analysis --help
```

### Smoke Tests

```bash
# All CLI commands
foodspec --help
foodspec data load --help
foodspec preprocess run --help
foodspec modeling train --help
foodspec analysis rq-analysis --help
foodspec workflow heating-quality --help
foodspec utils validate --help

# Python API imports
python -c "from foodspec import FoodSpec; print('✓ FoodSpec')"
python -c "from foodspec.features.rq import RatioQualityEngine; print('✓ RQ')"
python -c "from foodspec.core.api import FoodSpec; print('✓ core.api')"
```

---

## Rollback Plan

If any tests fail or imports break:

```bash
# Revert to pre-PHASE4 state
git reset --hard HEAD~1

# Or revert specific commit
git revert <commit-hash>
```

---

## Success Criteria

- ✅ All 642 tests pass
- ✅ No file exceeds 600 lines
- ✅ All imports work (backward compatibility)
- ✅ CLI functions normally
- ✅ Documentation builds
- ✅ Type hints preserved
- ✅ Code quality maintained or improved

---

## Notes

- This is a **code organization** refactoring, not a feature change
- **No behavior changes** - outputs must be identical
- **Backward compatibility** is critical - all existing code must work
- Focus on **safe, incremental** changes with testing between each step

---

**Next Step:** Create git branch and begin with api.py mixin extraction
