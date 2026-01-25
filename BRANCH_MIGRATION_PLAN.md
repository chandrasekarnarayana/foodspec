# FoodSpec Branch Migration & Legacy Code Deprecation Plan

**Date**: January 25, 2026  
**Current Branch**: `phase-1/protocol-driven-core`  
**Target**: Merge to `main` and deprecate legacy code  
**Version**: 1.0.0 → 2.0.0 (Major version bump due to breaking changes)

---

## Executive Summary

This document outlines the plan to:
1. Merge `phase-1/protocol-driven-core` into `main`
2. Identify and deprecate legacy code
3. Create a phased migration path for users
4. Remove deprecated code in v2.0.0

**Timeline**: 3-phase approach over 6 months
- Phase 1 (Month 1-2): Merge and mark deprecations
- Phase 2 (Month 3-4): Migration support and warnings
- Phase 3 (Month 5-6): Remove deprecated code (v2.0.0)

---

## Current State Analysis

### Branch Statistics

```
Current Branch: phase-1/protocol-driven-core
Total Python Files: 234
Major Modules: 28
New Implementations: 8 phases completed
Test Coverage: 12% overall (new modules: 88%+)
```

### New Architecture (Completed in Current Branch)

#### ✅ Phase 1-8 Implementations (KEEP - Production Ready)

1. **Core API (`core/`)** - NEW
   - `FoodSpec` unified API
   - `Spectrum`, `RunRecord`, `OutputBundle`
   - Modern protocol-driven architecture
   
2. **Trust Subsystem (`trust/`)** - NEW
   - Calibration, conformal prediction, reliability
   - Abstention logic, coverage guarantees
   - Uncertainty quantification

3. **Reporting (`reporting/`)** - NEW
   - Dossier generation
   - PDF export with WeasyPrint
   - Archive export for reproducibility
   - Paper presets (JOSS, Nature, Science)

4. **Visualization (`viz/`)** - NEW
   - Multi-run comparison utilities
   - Uncertainty visualizations
   - Coefficient/stability plots
   - Embeddings and processing stages

5. **Protocol System (`protocol/`)** - NEW
   - Protocol-driven execution
   - Step orchestration
   - Reproducible workflows

6. **QC Engine (`qc/`)** - MODERNIZED
   - Dataset quality checks
   - Drift detection
   - Leakage detection
   - Readiness scoring

7. **Features (`features/`)** - MODERNIZED
   - RQ (Ratio Quality) engine
   - Peak detection and ratios
   - Feature interpretation

8. **Preprocessing (`preprocess/`)** - MODERNIZED
   - Engine-based architecture
   - Baseline correction
   - Normalization, smoothing
   - Matrix correction

---

## Legacy Code Audit

### 🔴 CRITICAL: Modules to Deprecate

#### 1. **Duplicate/Old Core Modules**

##### `spectral_dataset.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Superseded by `core/spectral_dataset.py`
- **Action**: Add deprecation warning, redirect imports
- **Remove**: v2.0.0
- **Migration**: Use `from foodspec.core import SpectralDataset`

##### `output_bundle.py` (ROOT LEVEL) - DEPRECATED  
- **Status**: Superseded by `core/output_bundle.py`
- **Action**: Add deprecation warning
- **Remove**: v2.0.0
- **Migration**: Use `from foodspec.core import OutputBundle`

##### `model_lifecycle.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Replaced by `ml/lifecycle.py`
- **Action**: Add deprecation warning
- **Remove**: v2.0.0
- **Migration**: Use `from foodspec.ml import ModelLifecycle`

##### `model_registry.py` (ROOT LEVEL) - DEPRECATED
- **Status**: No longer used, registry moved to `deploy/`
- **Action**: Add deprecation warning
- **Remove**: v2.0.0

##### `preprocessing_pipeline.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Superseded by `preprocess/engine.py`
- **Action**: Add deprecation warning
- **Remove**: v2.0.0
- **Migration**: Use `from foodspec.preprocess import PreprocessingEngine`

##### `spectral_io.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Superseded by `io/` module
- **Action**: Add deprecation warning
- **Remove**: v2.0.0
- **Migration**: Use `from foodspec.io import load_folder, read_spectra`

##### `library_search.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Moved to `workflows/library_search.py`
- **Action**: Add deprecation warning
- **Remove**: v2.0.0

##### `validation.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Superseded by `chemometrics/validation.py`
- **Action**: Add deprecation warning
- **Remove**: v2.0.0

##### `harmonization.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Moved to `preprocess/` or `core/`
- **Action**: Add deprecation warning
- **Remove**: v2.0.0

##### `narrative.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Functionality moved to `reporting/`
- **Action**: Add deprecation warning
- **Remove**: v2.0.0

##### `reporting.py` (ROOT LEVEL) - DEPRECATED
- **Status**: Superseded by `reporting/` package
- **Action**: Add deprecation warning
- **Remove**: v2.0.0
- **Migration**: Use `from foodspec.reporting import generate_dossier`

#### 2. **Old CLI Structure**

##### `cli_*.py` files (ROOT LEVEL) - DEPRECATED
- `cli_plugin.py`
- `cli_predict.py`
- `cli_protocol.py`
- `cli_registry.py`

**Status**: Superseded by `cli/` package with proper command structure  
**Action**: Add deprecation warnings  
**Remove**: v2.0.0  
**Migration**: Use `foodspec-cli` command directly

#### 3. **Demo Modules**

##### `demo/` package - REVIEW
- Contains old heating demos
- **Action**: Migrate useful demos to `examples/`, deprecate package
- **Remove**: v2.0.0

#### 4. **Experimental Features**

##### `exp/` package - EVALUATE
- Experimental features that may not be production-ready
- **Action**: Audit, keep useful code, deprecate rest
- **Remove**: v2.0.0

##### `gui/` package - EVALUATE
- GUI components (likely incomplete)
- **Action**: If unused, deprecate
- **Remove**: v2.0.0

##### `hyperspectral/` package - EVALUATE  
- **Status**: Functionality in `core/hyperspectral.py`
- **Action**: Check for duplicates, deprecate if redundant
- **Remove**: v2.0.0

#### 5. **Old Apps Structure**

##### `apps/` package - MODERNIZE
- Contains domain-specific applications
- **Action**: Refactor to use new protocol system
- **Keep**: Modernize and integrate with protocols
- **Timeline**: Phase 2 modernization

Files to modernize:
- `apps/dairy.py`
- `apps/heating.py`
- `apps/meat.py`
- `apps/microbial.py`
- `apps/oils.py`
- `apps/qc.py`
- `apps/protocol_validation.py`

#### 6. **Old Machine Learning**

##### `ml/` package - REVIEW
- Some modules superseded by `chemometrics/`
- **Action**: Consolidate ML functionality
- **Remove duplicates**: v2.0.0

Files to review:
- `ml/calibration.py` - Check vs `trust/calibration.py`
- `ml/fusion.py` - Keep if unique
- `ml/hyperparameter_tuning.py` - Keep if unique
- `ml/lifecycle.py` - Primary, keep
- `ml/nested_cv.py` - Primary, keep

#### 7. **Old Report Structure**

##### `report/` package - DEPRECATED
- **Status**: Superseded by `reporting/`
- **Action**: Deprecate entire package
- **Remove**: v2.0.0
- **Migration**: Use `from foodspec.reporting import ...`

Files:
- `report/captions.py`
- `report/checklist.py`
- `report/methods.py`
- `report/stats_notes.py`
- `report/journals/`

---

## Module Classification Summary

### 🟢 KEEP (Production Ready - No Changes)
- `core/` - New unified API ✅
- `trust/` - Trust subsystem ✅
- `reporting/` - Modern reporting ✅
- `viz/` - Visualization suite ✅
- `protocol/` - Protocol engine ✅
- `qc/` - QC system ✅
- `features/` - Feature engineering ✅
- `preprocess/` - Preprocessing engine ✅
- `io/` - I/O operations ✅
- `data/` - Data loaders ✅
- `deploy/` - Deployment utilities ✅
- `stats/` - Statistical functions ✅
- `synthetic/` - Synthetic data ✅
- `utils/` - Utilities ✅
- `chemometrics/` - Chemometric models ✅
- `plugins/` - Plugin system ✅
- `repro/` - Reproducibility ✅
- `workflows/` - Workflow definitions ✅

### 🟡 MODERNIZE (Phase 2)
- `apps/` - Refactor to protocols (7 files)
- `ml/` - Consolidate with chemometrics (5 files)
- `cli/` - Already modern, minor updates

### 🔴 DEPRECATE (Phase 1) → REMOVE (v2.0.0)
- ROOT LEVEL: 12 deprecated files
  - `spectral_dataset.py`
  - `output_bundle.py`
  - `model_lifecycle.py`
  - `model_registry.py`
  - `preprocessing_pipeline.py`
  - `spectral_io.py`
  - `library_search.py`
  - `validation.py`
  - `harmonization.py`
  - `narrative.py`
  - `reporting.py`
  - `rq.py`
  
- ROOT LEVEL CLI: 4 files
  - `cli_plugin.py`
  - `cli_predict.py`
  - `cli_protocol.py`
  - `cli_registry.py`

- PACKAGES: 4 packages
  - `demo/` package (2 files)
  - `report/` package (6 files)
  - `exp/` package (if unused)
  - `gui/` package (if unused)
  - `hyperspectral/` package (check duplicates)

**Total to Deprecate**: ~30 files + 5 packages

---

## Migration Timeline

### Phase 1: Merge & Mark Deprecations (Months 1-2)

#### Week 1-2: Pre-merge Preparation

1. **Create Migration Documentation**
   ```bash
   docs/migration/v1-to-v2.md
   docs/migration/deprecated-modules.md
   docs/migration/api-changes.md
   ```

2. **Run Full Test Suite**
   ```bash
   pytest tests/ -v --cov=src/foodspec --cov-report=html
   ```

3. **Update CHANGELOG.md**
   - Document all breaking changes
   - List deprecated modules
   - Provide migration examples

4. **Create Release Branch**
   ```bash
   git checkout -b release/v1.1.0
   ```

#### Week 3-4: Add Deprecation Warnings

1. **Create Deprecation Helper**
   ```python
   # src/foodspec/utils/deprecation.py
   def deprecate_module(old_module, new_module, version="2.0.0"):
       """Issue deprecation warning for old module."""
       import warnings
       warnings.warn(
           f"{old_module} is deprecated and will be removed in v{version}. "
           f"Use {new_module} instead.",
           DeprecationWarning,
           stacklevel=2
       )
   ```

2. **Update Root-Level Files**
   - Add deprecation warnings to all identified files
   - Update imports to redirect to new locations
   - Add docstring warnings

3. **Update `__init__.py`**
   - Mark deprecated imports with warnings
   - Keep backward compatibility
   - Document migration path

4. **Tag Version v1.1.0-rc1**
   ```bash
   git tag -a v1.1.0-rc1 -m "Release candidate with deprecation warnings"
   ```

#### Week 5-6: Testing & Documentation

1. **Test Deprecation Warnings**
   ```bash
   pytest tests/ -W error::DeprecationWarning
   ```

2. **Update All Documentation**
   - README.md - Migration notice
   - API docs - Deprecation badges
   - Examples - Update to new APIs

3. **Create Migration Guide**
   - Step-by-step instructions
   - Code examples (old → new)
   - Common pitfalls

#### Week 7-8: Merge to Main

1. **Final Review**
   - Code review
   - Documentation review
   - Test coverage check

2. **Merge Strategy**
   ```bash
   # Option 1: Squash merge (clean history)
   git checkout main
   git merge --squash phase-1/protocol-driven-core
   git commit -m "Merge Phase 1: Protocol-driven core with deprecations"
   
   # Option 2: Preserve history
   git checkout main
   git merge phase-1/protocol-driven-core --no-ff
   ```

3. **Release v1.1.0**
   ```bash
   git tag -a v1.1.0 -m "v1.1.0: New architecture with deprecation warnings"
   git push origin main --tags
   ```

4. **Publish to PyPI**
   ```bash
   python -m build
   twine upload dist/*
   ```

### Phase 2: Migration Support (Months 3-4)

#### Month 3: User Support & Bug Fixes

1. **Monitor Issues**
   - GitHub issues for migration problems
   - Update migration guide based on feedback

2. **Modernize `apps/`**
   - Refactor apps to use protocol system
   - Update tests
   - Release v1.2.0

3. **Create Migration Scripts**
   ```python
   # scripts/migrate_to_v2.py
   # Automated code transformation where possible
   ```

4. **Consolidate ML Modules**
   - Merge duplicate functionality
   - Update imports
   - Release v1.3.0

#### Month 4: Final Warnings

1. **Increase Warning Severity**
   - Change to FutureWarning (more visible)
   - Add countdown messages

2. **Create Migration Checker**
   ```python
   # foodspec-check-migration
   # CLI tool to scan codebases for deprecated usage
   ```

3. **Release v1.4.0**
   - Final minor release before v2.0.0
   - All deprecations marked as URGENT

### Phase 3: Remove Deprecated Code (Months 5-6)

#### Month 5: Code Removal

1. **Create v2.0.0 Branch**
   ```bash
   git checkout -b release/v2.0.0
   ```

2. **Remove Deprecated Code**
   - Delete all marked files
   - Remove deprecated imports from `__init__.py`
   - Clean up tests

3. **Update Version**
   ```python
   __version__ = "2.0.0"
   ```

4. **Comprehensive Testing**
   ```bash
   pytest tests/ -v --cov=src/foodspec --cov-report=html
   # Target: 90% coverage
   ```

#### Month 6: Release v2.0.0

1. **Final Documentation**
   - Update all docs to v2.0.0 API
   - Remove references to deprecated code
   - Update examples

2. **Release Notes**
   - Breaking changes summary
   - Migration guide link
   - New features

3. **Release**
   ```bash
   git tag -a v2.0.0 -m "v2.0.0: Clean architecture, removed deprecated code"
   git push origin main --tags
   ```

4. **Publish**
   ```bash
   python -m build
   twine upload dist/*
   ```

---

## Detailed File-by-File Actions

### Root-Level Deprecated Files

```python
# Status: DEPRECATE → REMOVE in v2.0.0

FILES = {
    "spectral_dataset.py": {
        "replacement": "foodspec.core.SpectralDataset",
        "action": "Add deprecation warning, redirect imports",
        "effort": "1 hour"
    },
    "output_bundle.py": {
        "replacement": "foodspec.core.OutputBundle",
        "action": "Add deprecation warning, redirect imports",
        "effort": "1 hour"
    },
    "model_lifecycle.py": {
        "replacement": "foodspec.ml.ModelLifecycle",
        "action": "Add deprecation warning",
        "effort": "1 hour"
    },
    "model_registry.py": {
        "replacement": "None (functionality removed)",
        "action": "Add deprecation warning only",
        "effort": "30 min"
    },
    "preprocessing_pipeline.py": {
        "replacement": "foodspec.preprocess.PreprocessingEngine",
        "action": "Add deprecation warning, redirect imports",
        "effort": "1 hour"
    },
    "spectral_io.py": {
        "replacement": "foodspec.io",
        "action": "Add deprecation warning, redirect imports",
        "effort": "1 hour"
    },
    "library_search.py": {
        "replacement": "foodspec.workflows.library_search",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
    "validation.py": {
        "replacement": "foodspec.chemometrics.validation",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
    "harmonization.py": {
        "replacement": "foodspec.core.harmonize_datasets",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
    "narrative.py": {
        "replacement": "foodspec.reporting",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
    "reporting.py": {
        "replacement": "foodspec.reporting",
        "action": "Add deprecation warning, redirect imports",
        "effort": "1 hour"
    },
    "rq.py": {
        "replacement": "foodspec.features.rq",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
    "cli_plugin.py": {
        "replacement": "foodspec.cli",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
    "cli_predict.py": {
        "replacement": "foodspec.cli",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
    "cli_protocol.py": {
        "replacement": "foodspec.cli",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
    "cli_registry.py": {
        "replacement": "foodspec.cli",
        "action": "Add deprecation warning",
        "effort": "30 min"
    },
}

# Total effort: ~12 hours
```

### Package Deprecations

```python
PACKAGE_DEPRECATIONS = {
    "demo/": {
        "action": "Deprecate entire package",
        "migration": "Use examples/ directory",
        "effort": "2 hours"
    },
    "report/": {
        "action": "Deprecate entire package",
        "migration": "Use foodspec.reporting",
        "effort": "4 hours"
    },
    "exp/": {
        "action": "Audit and deprecate if unused",
        "migration": "Contact maintainers if needed",
        "effort": "8 hours (audit intensive)"
    },
    "gui/": {
        "action": "Audit and deprecate if unused",
        "migration": "None (may be incomplete)",
        "effort": "4 hours"
    },
    "hyperspectral/": {
        "action": "Check for duplicates vs core/hyperspectral.py",
        "migration": "Use foodspec.core.HyperSpectralCube",
        "effort": "4 hours"
    },
}

# Total effort: ~22 hours
```

---

## Deprecation Template

```python
# Template for deprecated modules
"""
[Module Name] - DEPRECATED

.. deprecated:: 1.1.0
    This module is deprecated and will be removed in v2.0.0.
    Use [new_module] instead.

This module is maintained for backward compatibility only.
All new code should use the modern API.

Migration Guide:
    Old: from foodspec import OldClass
    New: from foodspec.new_module import NewClass

See: docs/migration/v1-to-v2.md
"""

import warnings


def __getattr__(name):
    """Lazy deprecation warning on attribute access."""
    warnings.warn(
        f"foodspec.{__name__}.{name} is deprecated and will be removed in v2.0.0. "
        f"Use foodspec.new_module.{name} instead. "
        f"See docs/migration/v1-to-v2.md for migration guide.",
        DeprecationWarning,
        stacklevel=2
    )
    # Import and return from new location
    from foodspec.new_module import name as new_attr
    return new_attr


# Original code continues below with deprecation warnings...
```

---

## Testing Strategy

### Deprecation Tests

```python
# tests/test_deprecations.py

import pytest
import warnings


class TestDeprecations:
    """Test that deprecated modules issue warnings."""
    
    def test_spectral_dataset_deprecation(self):
        with pytest.warns(DeprecationWarning, match="spectral_dataset"):
            from foodspec import spectral_dataset
    
    def test_output_bundle_deprecation(self):
        with pytest.warns(DeprecationWarning, match="output_bundle"):
            from foodspec import output_bundle
    
    # ... more tests for each deprecated module


class TestMigrationPaths:
    """Test that old imports redirect to new locations."""
    
    def test_spectral_dataset_redirect(self):
        # Old import should work but warn
        with pytest.warns(DeprecationWarning):
            from foodspec.spectral_dataset import SpectralDataset as Old
        
        # New import should not warn
        from foodspec.core import SpectralDataset as New
        
        # Should be same class (or compatible)
        assert Old is New or issubclass(Old, New)
```

### Backward Compatibility Tests

```python
# tests/test_backward_compatibility.py

class TestBackwardCompatibility:
    """Ensure old code still works (with warnings)."""
    
    def test_old_api_import(self):
        """Test that old-style imports still work."""
        with pytest.warns(DeprecationWarning):
            from foodspec.spectral_dataset import SpectralDataset
            ds = SpectralDataset(...)
            assert ds is not None
    
    def test_old_workflow(self):
        """Test that old workflow patterns still work."""
        with pytest.warns(DeprecationWarning):
            # Old-style workflow
            from foodspec.preprocessing_pipeline import Pipeline
            pipeline = Pipeline()
            # Should still work
```

---

## Documentation Updates

### 1. Migration Guide (`docs/migration/v1-to-v2.md`)

```markdown
# Migration Guide: v1.x → v2.0.0

## Overview
FoodSpec v2.0.0 introduces a modern, protocol-driven architecture...

## Breaking Changes
- Root-level modules moved to packages
- Old CLI scripts replaced by unified CLI
- Report → reporting package

## Step-by-Step Migration

### 1. Update Imports
[Detailed import changes with examples]

### 2. Update Workflows
[Workflow migration examples]

### 3. Update Tests
[Test migration examples]

## Automated Migration
```bash
# Use migration checker
foodspec-check-migration /path/to/your/code

# Apply automatic fixes (where possible)
foodspec-migrate --apply /path/to/your/code
```

## Common Issues
[FAQ and troubleshooting]
```

### 2. API Documentation
- Add deprecation badges to old APIs
- Link to new APIs
- Show side-by-side examples

### 3. README Updates
```markdown
# FoodSpec

⚠️ **Important**: v1.1.0+ includes deprecation warnings for old APIs.
Please see [Migration Guide](docs/migration/v1-to-v2.md) to prepare for v2.0.0.

## Quick Start (v2.0.0 API)
[Updated examples using new API]
```

---

## Risk Assessment

### High Risk
1. **Breaking User Code**: Old imports will break in v2.0.0
   - **Mitigation**: 6-month deprecation period, clear warnings
   
2. **Incomplete Migration**: Users may miss deprecation warnings
   - **Mitigation**: Multiple warning levels, migration checker tool

3. **Lost Functionality**: Some features may be removed
   - **Mitigation**: Thorough audit, keep all useful features

### Medium Risk
1. **Test Coverage Gaps**: Some old code may not have tests
   - **Mitigation**: Add tests before deprecation

2. **Documentation Drift**: Docs may not match code
   - **Mitigation**: Comprehensive doc review

### Low Risk
1. **Performance Regressions**: New code may be slower
   - **Mitigation**: Benchmark tests

2. **Dependency Issues**: New deps may conflict
   - **Mitigation**: Careful dependency management

---

## Success Metrics

### Phase 1 (Merge & Deprecate)
- ✅ All tests passing
- ✅ All deprecated modules have warnings
- ✅ Migration guide published
- ✅ v1.1.0 released

### Phase 2 (Migration Support)
- ✅ <10 migration-related issues/month
- ✅ Migration guide used by >50% of users
- ✅ apps/ modernized
- ✅ v1.4.0 released

### Phase 3 (v2.0.0)
- ✅ All deprecated code removed
- ✅ Test coverage >90%
- ✅ Zero breaking change issues
- ✅ v2.0.0 released

---

## Rollback Plan

If critical issues arise:

1. **Minor Issues**: Patch release (v1.1.1, etc.)
2. **Major Issues**: Revert merge, fix in branch, re-merge
3. **Critical Issues**: Roll back to last stable version

```bash
# Emergency rollback
git revert [merge-commit-hash]
git tag -a v1.0.1-hotfix -m "Hotfix: rollback merge"
```

---

## Communication Plan

### Week Before Merge
- Blog post announcing merge
- Email to user mailing list
- Social media announcement

### Merge Day
- GitHub release notes
- PyPI release
- Update documentation

### Ongoing
- Monthly blog posts on migration progress
- Quarterly surveys on migration challenges
- Active issue tracking and support

---

## Action Items

### Immediate (Week 1)
- [ ] Review and approve this plan
- [ ] Create migration branch
- [ ] Begin deprecation implementation
- [ ] Write migration guide outline

### Short-term (Month 1)
- [ ] Complete deprecation warnings
- [ ] Comprehensive testing
- [ ] Documentation updates
- [ ] Release v1.1.0-rc1

### Medium-term (Months 2-4)
- [ ] Merge to main
- [ ] User support and feedback
- [ ] Modernize apps/
- [ ] Consolidate ML modules

### Long-term (Months 5-6)
- [ ] Remove deprecated code
- [ ] Final testing
- [ ] Release v2.0.0

---

## Appendix A: Complete File Listing

### Files to Keep (No Changes)
- core/*.py (19 files)
- trust/*.py (6 files)
- reporting/*.py (5 files)
- viz/*.py (15 files)
- protocol/*.py (8 files)
- qc/*.py (10 files)
- features/*.py (12 files)
- preprocess/*.py (10 files)
- io/*.py (8 files)
- [... full listing]

### Files to Deprecate
[Complete list with replacement paths]

### Files to Remove
[Complete list with removal timeline]

---

## Appendix B: Import Mapping Table

| Old Import | New Import | Status |
|-----------|------------|--------|
| `from foodspec.spectral_dataset import ...` | `from foodspec.core import ...` | Deprecated |
| `from foodspec.output_bundle import ...` | `from foodspec.core import ...` | Deprecated |
| `from foodspec.model_lifecycle import ...` | `from foodspec.ml import ...` | Deprecated |
| ... | ... | ... |

[Complete mapping table]

---

## Sign-Off

**Prepared by**: GitHub Copilot  
**Date**: January 25, 2026  
**Version**: 1.0  

**Approval Required**:
- [ ] Lead Developer
- [ ] Project Maintainer
- [ ] Documentation Lead
- [ ] QA Lead

---

**Next Steps**: Review this plan and proceed with Phase 1 Week 1 actions.
