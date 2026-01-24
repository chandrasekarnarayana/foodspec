# FoodSpec Compatibility Layer Plan

**Document Version**: 1.0  
**Status**: Active / In Implementation  
**Last Updated**: 2026-01-24

---

## Overview

During FoodSpec's refactor into a protocol-driven framework, we must maintain **backward compatibility** so existing user code continues to work. This document defines:

1. **Public API surface** to maintain
2. **Deprecation strategy** with timelines
3. **Re-export modules** for seamless migration
4. **Versioning policy** for breaking changes
5. **User migration guide**

---

## Goals

✓ **No breaking changes** in minor/patch versions  
✓ **Clear deprecation path** with 2+ releases notice  
✓ **Old imports work** via re-exports with `DeprecationWarning`  
✓ **Gradual migration** so users aren't forced to refactor overnight  
✓ **Transparent** — users always know what's deprecated and why  

---

## Public API Surface (Must Maintain)

The following are considered **stable public API** and must remain importable in v1.x:

### Core Classes & Functions

```python
# Core data structures
from foodspec import Spectrum
from foodspec import FoodSpectrumSet
from foodspec import HyperSpectralCube
from foodspec import OutputBundle
from foodspec import RunRecord

# Unified entry point (Phase 1)
from foodspec import FoodSpec

# Data loading (widely used)
from foodspec import (
    load_folder,
    load_library,
    create_library,
    load_csv_spectra,
    read_spectra,
    detect_format,
)

# Metrics & Statistics
from foodspec import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_roc_curve,
    compute_pr_curve,
    run_anova,
    run_ttest,
    run_manova,
    run_tukey_hsd,
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_wilcoxon_signed_rank,
    run_friedman_test,
    bootstrap_metric,
    permutation_test_metric,
)

# Quality Control
from foodspec import (
    estimate_snr,
    summarize_class_balance,
    detect_outliers,
    check_missing_metadata,
)

# Synthetic Data
from foodspec import (
    generate_synthetic_raman_spectrum,
    generate_synthetic_ftir_spectrum,
)

# Advanced Features (Moats)
from foodspec import (
    apply_matrix_correction,
    analyze_heating_trajectory,
    calibration_transfer_workflow,
    direct_standardization,
    piecewise_direct_standardization,
)

# Data Governance
from foodspec import (
    summarize_dataset,
    check_class_balance,
    diagnose_imbalance,
    compute_replicate_consistency,
    assess_variability_sources,
    detect_batch_label_correlation,
    detect_replicate_leakage,
    detect_leakage,
    compute_readiness_score,
)

# RQ Engine
from foodspec import (
    PeakDefinition,
    RatioDefinition,
    RQConfig,
    RatioQualityEngine,
    RatioQualityResult,
)

# HDF5/Dataset
from foodspec import (
    HDF5_SCHEMA_VERSION,
    HyperspectralDataset,
    PreprocessingConfig,
    SpectralDataset,
    baseline_als,
    baseline_polynomial,
    baseline_rubberband,
    harmonize_datasets,
)

# Reproducibility
from foodspec import (
    DatasetSpec,
    ExperimentConfig,
    ExperimentEngine,
    diff_runs,
)

# Artifact Management
from foodspec import (
    Predictor,
    save_artifact,
    load_artifact,
)

# Plugin System
from foodspec import (
    PluginManager,
    install_plugin,
    load_plugins,
)

# Utilities
from foodspec import (
    LOGO_BASE64,
    get_logo_base64,
    get_logo_bytes,
    save_logo,
)
```

All of these **must remain importable** from the top-level `foodspec` package through v1.x.

### Sub-Module Imports (Secondary Surface)

These are also stable but less frequently imported:

```python
# Data I/O
from foodspec.io import load_csv_spectra, read_spectra, detect_format

# Preprocessing
from foodspec.preprocessing import baseline_als, normalize, smooth

# Statistics
from foodspec.stats import run_anova, run_ttest, bootstrap_metric

# Quality Control
from foodspec.qc import detect_outliers, check_missing_metadata

# Features
from foodspec.features.rq import RQConfig, RatioQualityEngine

# Reproducibility
from foodspec.repro import ExperimentEngine, diff_runs
```

---

## Refactoring Plan: Old → New Locations

As we refactor, some modules will move. **Old imports must still work** via re-exports.

### Example: Baseline Subtraction Refactor

**Before (v1.0.0):**
```python
from foodspec.spectral_dataset import baseline_als
```

**After (v1.1.0, new structure):**
```python
from foodspec.core.preprocessing.baseline import baseline_als
```

**Backward Compatibility (v1.1.0+):**
```python
# Old import still works:
from foodspec.spectral_dataset import baseline_als  # ✅ Works, but deprecated

# New import (recommended):
from foodspec.core.preprocessing.baseline import baseline_als  # ✅ Preferred

# Top-level also works:
from foodspec import baseline_als  # ✅ Always works
```

### Example: Re-export Module Structure

**File: `src/foodspec/spectral_dataset.py` (legacy, kept for compat)**

```python
"""
DEPRECATED: Baseline functions moved to foodspec.core.preprocessing.

This module is maintained for backward compatibility only.
Use foodspec.core.preprocessing.baseline instead.

Examples
--------
>>> # OLD (deprecated)
>>> from foodspec.spectral_dataset import baseline_als
>>> # NEW (recommended)
>>> from foodspec.core.preprocessing.baseline import baseline_als
"""

import warnings
from foodspec.core.preprocessing.baseline import (
    baseline_als as _baseline_als,
    baseline_polynomial as _baseline_polynomial,
    baseline_rubberband as _baseline_rubberband,
)

def baseline_als(*args, **kwargs):
    """Baseline subtraction using asymmetric least squares.
    
    .. deprecated:: 1.1.0
        Use :func:`foodspec.core.preprocessing.baseline.baseline_als` instead.
    """
    warnings.warn(
        "baseline_als from foodspec.spectral_dataset is deprecated and will be "
        "removed in v2.0.0. Use foodspec.core.preprocessing.baseline.baseline_als instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _baseline_als(*args, **kwargs)

def baseline_polynomial(*args, **kwargs):
    """Baseline subtraction using polynomial fitting.
    
    .. deprecated:: 1.1.0
        Use :func:`foodspec.core.preprocessing.baseline.baseline_polynomial` instead.
    """
    warnings.warn(
        "baseline_polynomial from foodspec.spectral_dataset is deprecated and will be "
        "removed in v2.0.0. Use foodspec.core.preprocessing.baseline.baseline_polynomial instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _baseline_polynomial(*args, **kwargs)

def baseline_rubberband(*args, **kwargs):
    """Baseline subtraction using rubberband method.
    
    .. deprecated:: 1.1.0
        Use :func:`foodspec.core.preprocessing.baseline.baseline_rubberband` instead.
    """
    warnings.warn(
        "baseline_rubberband from foodspec.spectral_dataset is deprecated and will be "
        "removed in v2.0.0. Use foodspec.core.preprocessing.baseline.baseline_rubberband instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _baseline_rubberband(*args, **kwargs)

__all__ = ['baseline_als', 'baseline_polynomial', 'baseline_rubberband']
```

### Example: Top-level Re-exports in `__init__.py`

**File: `src/foodspec/__init__.py`**

```python
"""foodspec: Protocol-driven spectroscopy framework."""

__all__ = [
    "__version__",
    # ... all public exports ...
]

__version__ = "1.1.0"

# ============================================================================
# SECTION 1: New structure (recommended imports)
# ============================================================================

from foodspec.core.api import FoodSpec
from foodspec.core.spectrum import Spectrum
from foodspec.core.preprocessing.baseline import (
    baseline_als,
    baseline_polynomial,
    baseline_rubberband,
)
# ... more new imports ...

# ============================================================================
# SECTION 2: Backward compatibility re-exports (for old import paths)
# ============================================================================

# These re-exports ensure old code doesn't break, but emit DeprecationWarning

# Option A: Simple re-export (no warning)
# Old code: from foodspec.old_module import func
# New code: from foodspec.new_module import func
# Both work, but old path not recommended
from foodspec.core.preprocessing.baseline import baseline_als  # noqa: F401

# Option B: Re-export with deprecation wrapper (emits warning)
# Use when moving between submodules
def _deprecated_import(old_name, new_module, new_name, version_removed="2.0.0"):
    """Create a deprecated re-export wrapper.
    
    Parameters
    ----------
    old_name : str
        Name users import from (e.g., "baseline_als")
    new_module : str
        New module path (e.g., "foodspec.core.preprocessing.baseline")
    new_name : str
        Name in new module
    version_removed : str
        Version when this import will be removed
    """
    def wrapper(*args, **kwargs):
        import warnings
        from importlib import import_module
        
        warnings.warn(
            f"Importing {old_name} from foodspec is deprecated and will be "
            f"removed in {version_removed}. "
            f"Use: from {new_module} import {new_name}",
            DeprecationWarning,
            stacklevel=2
        )
        mod = import_module(new_module)
        return getattr(mod, new_name)(*args, **kwargs)
    
    return wrapper
```

---

## Deprecation Timeline & Versioning

### Version Strategy

FoodSpec follows **Semantic Versioning (SemVer)**: `MAJOR.MINOR.PATCH`

- **MAJOR** (e.g., v1.0.0 → v2.0.0): Breaking changes allowed
- **MINOR** (e.g., v1.0.0 → v1.1.0): New features, deprecations, but no breaking changes
- **PATCH** (e.g., v1.0.0 → v1.0.1): Bug fixes only

### Deprecation Timeline

```
v1.0.0 (Current, baseline)
  ├─ Original API fully functional
  └─ No deprecation warnings

v1.1.0 (Q1 2026, first refactor phase)
  ├─ NEW: Protocol-driven core API available
  ├─ OLD: Original API still works
  └─ DEPRECATED: Functions moved get DeprecationWarning
      (Warnings added to docs, release notes)

v1.2.0 (Q2 2026, second phase)
  ├─ NEW: More internal restructuring
  ├─ OLD: Original API still works
  └─ SAME DEPRECATIONS: DeprecationWarning still emitted

v2.0.0 (Q4 2026, breaking release)
  ├─ BREAKING: Deprecated functions removed
  ├─ NEW: Clean, modern API structure
  └─ MIGRATION GUIDE: Provided in RELEASE_NOTES_v2.0.0.md
```

### Deprecation Notices

**In docstrings:**
```python
def old_function():
    """Process spectra.
    
    .. deprecated:: 1.1.0
        Use :func:`foodspec.new_module.new_function` instead.
    
    This function is maintained for backward compatibility only and will be
    removed in v2.0.0. See RELEASE_NOTES_v1.1.0.md for migration guide.
    """
```

**In RELEASE_NOTES:**
```markdown
## v1.1.0 Deprecation Warnings

The following APIs are deprecated and will be removed in v2.0.0:

| Deprecated | Use Instead | Migration Path |
|---|---|---|
| `foodspec.spectral_dataset.baseline_als` | `foodspec.core.preprocessing.baseline.baseline_als` | See [Baseline Migration](docs/migration/baseline.md) |
| `foodspec.io.load_csv_spectra` | `foodspec.io.load_csv` (simplified) | [Link](docs/migration/io.md) |
```

---

## Implementation: Re-export Patterns

### Pattern 1: Simple Re-export (No Warning)

Use when moving a function to a new location but signature/behavior unchanged:

```python
# src/foodspec/old_location.py (legacy, kept for compat)

from foodspec.new_location import some_function

__all__ = ['some_function']
```

**Effect**: Old import works, no warning emitted. Users can migrate at their own pace.

### Pattern 2: Re-export with Deprecation Warning

Use when the function should warn users to migrate:

```python
# src/foodspec/old_location.py (legacy, kept for compat)

import warnings

def some_function(*args, **kwargs):
    """[Original docstring]
    
    .. deprecated:: 1.1.0
        Use foodspec.new_location.some_function instead.
    """
    warnings.warn(
        "some_function from foodspec.old_location is deprecated and will be "
        "removed in v2.0.0. Use foodspec.new_location.some_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import and call the real implementation
    from foodspec.new_location import some_function as _impl
    return _impl(*args, **kwargs)

__all__ = ['some_function']
```

**Effect**: Old import works but emits `DeprecationWarning`. Users see clear migration path.

### Pattern 3: Re-export via __init__.py (Top-level)

Keep all top-level exports stable:

```python
# src/foodspec/__init__.py

__all__ = [
    "__version__",
    "some_function",
    "AnotherClass",
    # ... many more ...
]

# Import from wherever it lives now (users don't care about internal structure)
from foodspec.core.preprocessing.baseline import baseline_als as some_function
from foodspec.core.data.spectrum import Spectrum as AnotherClass

# Optional: If it's moved significantly, add deprecation wrapper
# But usually, top-level imports are stable and don't need warnings
```

### Pattern 4: Deprecated Class Moved to New Module

```python
# src/foodspec/old_module.py (legacy)

import warnings
from foodspec.core.new_module import NewClass

class OldClass:
    """Process spectra [original docstring].
    
    .. deprecated:: 1.1.0
        Use :class:`foodspec.core.new_module.NewClass` instead.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OldClass from foodspec.old_module is deprecated. "
            "Use foodspec.core.new_module.NewClass instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._impl = NewClass(*args, **kwargs)
    
    def __getattr__(self, name):
        # Delegate all attributes to the real implementation
        return getattr(self._impl, name)

__all__ = ['OldClass']
```

---

## User Migration Guide

### For End Users

**Step 1: Identify deprecated imports**

```bash
python -W default::DeprecationWarning -c "import foodspec; from foodspec.spectral_dataset import baseline_als"
```

You'll see:
```
DeprecationWarning: baseline_als from foodspec.spectral_dataset is deprecated and will be
removed in v2.0.0. Use foodspec.core.preprocessing.baseline.baseline_als instead.
```

**Step 2: Update imports**

```python
# OLD (deprecated)
from foodspec.spectral_dataset import baseline_als

# NEW (recommended)
from foodspec.core.preprocessing.baseline import baseline_als
# OR (also works)
from foodspec import baseline_als
```

**Step 3: Run tests**

```bash
pytest tests/ -v -W error::DeprecationWarning  # Fail on any deprecation warnings
```

**Step 4: Update code**

Replace old imports with new ones. See [MIGRATION_GUIDE_v1.1_to_v2.0.md](./MIGRATION_GUIDE_v1.1_to_v2.0.md) for detailed path mapping.

### For Package Maintainers

If you depend on FoodSpec and want to update your code:

```bash
# Audit current imports
python -W default::DeprecationWarning -c "import your_package" 2>&1 | grep DeprecationWarning

# Create a compatibility shim (if needed)
# see below...
```

**Example: Creating a Compatibility Shim**

If you can't immediately update your code:

```python
# your_package/_foodspec_compat.py

"""Compatibility shim for FoodSpec migration."""

import warnings

# Suppress FoodSpec deprecation warnings while you migrate
warnings.filterwarnings('ignore', category=DeprecationWarning, module='foodspec')

# Now old imports work silently (use temporarily, not permanently!)
from foodspec.spectral_dataset import baseline_als
from foodspec.spectral_dataset import baseline_polynomial

__all__ = ['baseline_als', 'baseline_polynomial']
```

Then in your code:
```python
# Temporary: Use shim during migration
from your_package._foodspec_compat import baseline_als

# Once ready: Switch to new import
from foodspec.core.preprocessing.baseline import baseline_als
```

---

## Testing Backward Compatibility

### Test Structure

```python
# tests/test_compat.py

import pytest
import warnings

class TestBackwardCompatibility:
    """Ensure deprecated APIs still work."""
    
    def test_old_import_path_works(self):
        """Old import path should work (but emit warning)."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            from foodspec.spectral_dataset import baseline_als
            assert callable(baseline_als)
    
    def test_new_import_path_works(self):
        """New import path should work."""
        from foodspec.core.preprocessing.baseline import baseline_als
        assert callable(baseline_als)
    
    def test_top_level_import_works(self):
        """Top-level import should always work."""
        from foodspec import baseline_als
        assert callable(baseline_als)
    
    def test_deprecated_function_behavior(self):
        """Deprecated function should behave identically to new one."""
        import numpy as np
        
        spectrum = np.array([0.1, 0.2, 1.5, 1.2, 0.3, 0.15, 0.1])
        
        # Get result from old path (with warning suppression)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from foodspec.spectral_dataset import baseline_als as old_baseline
            result_old = old_baseline(spectrum, lam=1e4)
        
        # Get result from new path
        from foodspec.core.preprocessing.baseline import baseline_als as new_baseline
        result_new = new_baseline(spectrum, lam=1e4)
        
        # Should be identical
        np.testing.assert_array_equal(result_old, result_new)
```

### CI/CD Integration

```yaml
# .github/workflows/compatibility.yml

name: Backward Compatibility Tests

on: [push, pull_request]

jobs:
  test-compat:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/test_compat.py -v
      
  test-no-warnings:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -W error::DeprecationWarning -v
        # Fail if any unexpected deprecation warnings
```

---

## Breaking Changes (For v2.0.0)

When releasing v2.0.0, these deprecated items will be removed:

1. All functions in `foodspec.spectral_dataset` (moved to `foodspec.core.*`)
2. All functions in `foodspec.io.legacy` (replaced by new `foodspec.io`)
3. Old `PreprocessingConfig` (replaced with new protocol-based config)
4. ... others documented in RELEASE_NOTES_v2.0.0.md

**Migration guide will be provided** in RELEASE_NOTES_v2.0.0.md with clear before/after examples.

---

## Re-export Module Template

Use this template for creating compat modules:

```python
"""[Module name] - Legacy interface for backward compatibility.

This module is maintained for compatibility with FoodSpec < 1.1.0.
New code should use:

from foodspec.core.new_location import [function]

This module will be removed in v2.0.0.
"""

import warnings
from foodspec.core.new_location import (
    function1,
    function2,
    ClassA,
)

__all__ = [
    'function1',
    'function2',
    'ClassA',
]

# Optional: Emit deprecation warning for entire module
def __getattr__(name):
    if name in __all__:
        warnings.warn(
            f"Module foodspec.old_location is deprecated. "
            f"Import {name} from foodspec.core.new_location instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return globals()[name]
    raise AttributeError(f"module has no attribute {name}")
```

---

## Summary Checklist

- [ ] All public APIs listed in [Public API Surface](#public-api-surface)
- [ ] Re-export modules created for moved functions
- [ ] All re-exports emit `DeprecationWarning` with migration path
- [ ] Top-level `__init__.py` re-exports are up-to-date
- [ ] Backward compatibility tests written (test_compat.py)
- [ ] CI/CD checks for deprecation warnings
- [ ] Release notes document all deprecations
- [ ] Migration guide provided (docs/migration/)
- [ ] Example scripts updated to new imports
- [ ] Documentation updated with both old and new import paths (with deprecation notice)

---

## Related Documents

- [CONTRIBUTING.md](../../CONTRIBUTING.md) — Contributing guidelines
- [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) — Engineering principles
- [RELEASE_NOTES_v1.1.0.md](../../RELEASE_NOTES_v1.1.0.md) — v1.1.0 release info
- [MIGRATION_GUIDE_v1.1_to_v2.0.md](./MIGRATION_GUIDE_v1.1_to_v2.0.md) — Detailed migration guide

---

**Maintained by**: FoodSpec Core Team  
**Last Updated**: 2026-01-24  
**Questions?** Open an issue or email chandrasekarnarayana@gmail.com
