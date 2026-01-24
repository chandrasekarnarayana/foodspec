# Backward Compatibility Examples

This file contains ready-to-use examples for implementing backward compatibility in FoodSpec.

---

## Example 1: Simple Function Re-export (No Warning)

**Use case**: Function moved to new module, but interface is stable.

**File: `src/foodspec/old_module.py`**

```python
"""Legacy import path for backward compatibility.

DEPRECATED: Use foodspec.core.new_module instead.
This module is maintained for compatibility only and will be removed in v2.0.0.
"""

# Re-export from new location (no warning)
from foodspec.core.new_module import baseline_als, baseline_polynomial

__all__ = ['baseline_als', 'baseline_polynomial']
```

**Effect**: 
```python
# Both import paths work
from foodspec.old_module import baseline_als  # ✅ Works (no warning)
from foodspec.core.new_module import baseline_als  # ✅ Works (recommended)
```

---

## Example 2: Function Re-export with Deprecation Warning

**Use case**: Function moved, and you want users to be notified.

**File: `src/foodspec/old_module.py`**

```python
"""Legacy import path for backward compatibility.

DEPRECATED: Use foodspec.core.new_module instead.
This module is maintained for compatibility only and will be removed in v2.0.0.

Migration example:
    # OLD
    from foodspec.old_module import baseline_als
    
    # NEW
    from foodspec.core.new_module import baseline_als
"""

import warnings

def baseline_als(spectrum, lam=1e4, p=0.01, niter=10):
    """Apply asymmetric least squares baseline subtraction.
    
    .. deprecated:: 1.1.0
        Use :func:`foodspec.core.new_module.baseline_als` instead.
        This function is maintained for backward compatibility only
        and will be removed in v2.0.0.
    """
    warnings.warn(
        "baseline_als from foodspec.old_module is deprecated and will be "
        "removed in v2.0.0. Use foodspec.core.new_module.baseline_als instead. "
        "See https://foodspec.readthedocs.io/migration/ for details.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import and call the real implementation
    from foodspec.core.new_module import baseline_als as _baseline_als
    return _baseline_als(spectrum, lam=lam, p=p, niter=niter)

__all__ = ['baseline_als']
```

**Effect**:
```python
from foodspec.old_module import baseline_als
result = baseline_als(spectrum)
# Output:
# DeprecationWarning: baseline_als from foodspec.old_module is deprecated...
```

---

## Example 3: Class Re-export with Delegation

**Use case**: Class moved to new module but needs seamless interface.

**File: `src/foodspec/old_module.py`**

```python
"""Legacy interface for BaselineCorrector.

DEPRECATED: Use foodspec.core.preprocessing.BaselineCorrector instead.
"""

import warnings

class BaselineCorrector:
    """Apply baseline correction to spectra.
    
    .. deprecated:: 1.1.0
        Use :class:`foodspec.core.preprocessing.BaselineCorrector` instead.
    """
    
    def __init__(self, method='als', **kwargs):
        warnings.warn(
            "BaselineCorrector from foodspec.old_module is deprecated. "
            "Use foodspec.core.preprocessing.BaselineCorrector instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Import and instantiate the real class
        from foodspec.core.preprocessing import BaselineCorrector as _Impl
        self._impl = _Impl(method=method, **kwargs)
    
    def fit(self, spectra):
        """Fit baseline parameters to spectra."""
        return self._impl.fit(spectra)
    
    def transform(self, spectra):
        """Apply baseline correction."""
        return self._impl.transform(spectra)
    
    def fit_transform(self, spectra):
        """Fit and transform in one call."""
        return self._impl.fit_transform(spectra)
    
    # Delegate unknown attributes to implementation
    def __getattr__(self, name):
        return getattr(self._impl, name)

__all__ = ['BaselineCorrector']
```

---

## Example 4: Module-Level Deprecation (Using __getattr__)

**Use case**: Entire module is deprecated; fine-grained control over each export.

**File: `src/foodspec/old_legacy_module.py`**

```python
"""Legacy module for backward compatibility.

DEPRECATED: All functions in this module are deprecated.
See docs/migration/ for the new structure.

This module will be removed in v2.0.0.
"""

import warnings
from importlib import import_module

# Map old names to new locations
_MIGRATION_MAP = {
    'baseline_als': ('foodspec.core.preprocessing.baseline', 'baseline_als'),
    'baseline_polynomial': ('foodspec.core.preprocessing.baseline', 'baseline_polynomial'),
    'normalize_spectrum': ('foodspec.core.preprocessing.normalization', 'normalize'),
    'smooth_spectrum': ('foodspec.core.preprocessing.smoothing', 'smooth'),
}

def __getattr__(name):
    """Dynamically import from new locations with deprecation warnings."""
    
    if name not in _MIGRATION_MAP:
        raise AttributeError(f"module has no attribute '{name}'")
    
    new_module, new_name = _MIGRATION_MAP[name]
    
    warnings.warn(
        f"Importing {name} from foodspec.old_legacy_module is deprecated. "
        f"Use: from {new_module} import {new_name}. "
        f"This import will be removed in v2.0.0.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import and return the actual function
    mod = import_module(new_module)
    return getattr(mod, new_name)

def __dir__():
    """List available (deprecated) names."""
    return list(_MIGRATION_MAP.keys())

__all__ = list(_MIGRATION_MAP.keys())
```

**Effect**:
```python
from foodspec.old_legacy_module import baseline_als, normalize_spectrum
# Each import emits DeprecationWarning with clear migration path

# List available functions
dir(foodspec.old_legacy_module)
# ['baseline_als', 'baseline_polynomial', 'normalize_spectrum', 'smooth_spectrum']
```

---

## Example 5: Top-Level __init__.py Re-exports

**File: `src/foodspec/__init__.py`**

```python
"""FoodSpec: Protocol-driven spectroscopy framework."""

__version__ = "1.1.0"

__all__ = [
    "__version__",
    # Core API (stable)
    "FoodSpec",
    "Spectrum",
    "RunRecord",
    "OutputBundle",
    # Preprocessing (stable, may come from new location)
    "baseline_als",
    "baseline_polynomial",
    "baseline_rubberband",
    "normalize",
    "smooth",
    # ... many more ...
]

# ============================================================================
# Section 1: Import from NEW locations (primary sources)
# ============================================================================

# Phase 1 API
from foodspec.core.api import FoodSpec
from foodspec.core.spectrum import Spectrum
from foodspec.core.run_record import RunRecord
from foodspec.core.output_bundle import OutputBundle

# Preprocessing (after refactor, these come from core)
from foodspec.core.preprocessing.baseline import (
    baseline_als as _baseline_als,
    baseline_polynomial as _baseline_polynomial,
    baseline_rubberband as _baseline_rubberband,
)
from foodspec.core.preprocessing.normalization import normalize as _normalize
from foodspec.core.preprocessing.smoothing import smooth as _smooth

# ============================================================================
# Section 2: Create stable top-level interface
# ============================================================================
# These are always available and don't change (unless major version bump)

# Option A: Direct re-export (most functions)
baseline_als = _baseline_als
baseline_polynomial = _baseline_polynomial
baseline_rubberband = _baseline_rubberband
normalize = _normalize
smooth = _smooth

# Option B: Wrapper with additional logic (if needed)
def run_experiment(config, seed=None):
    """Run a complete spectroscopy experiment.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    OutputBundle
        Results and artifacts.
    
    Examples
    --------
    >>> config = ExperimentConfig(...)
    >>> result = run_experiment(config, seed=42)
    """
    from foodspec.core.experiment import run_experiment as _run
    return _run(config, seed=seed)

# ============================================================================
# Section 3: Backward compatibility re-exports
# ============================================================================
# These are deprecated but still work via old import paths

# The old_module.py files handle these, but we can also re-export here
# if needed for convenience

# Example: If old code imported from foodspec.io
from foodspec.io import (
    load_folder,
    load_library,
    create_library,
    load_csv_spectra,
    read_spectra,
    detect_format,
)
```

---

## Example 6: Test for Backward Compatibility

**File: `tests/test_backward_compat.py`**

```python
"""Tests for backward compatibility."""

import pytest
import warnings
import numpy as np


class TestBackwardCompatibility:
    """Verify deprecated imports still work."""
    
    @pytest.fixture
    def sample_spectrum(self):
        """Create a test spectrum."""
        np.random.seed(42)
        x = np.arange(100)
        signal = 10 * np.exp(-0.05 * (x - 50)**2)
        baseline = 0.1 * x
        noise = np.random.normal(0, 0.05, 100)
        return signal + baseline + noise
    
    def test_old_import_path_emits_deprecation(self, sample_spectrum):
        """Deprecated import should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            from foodspec.old_module import baseline_als
            result = baseline_als(sample_spectrum)
            assert len(result) == len(sample_spectrum)
    
    def test_new_import_path_no_warning(self, sample_spectrum):
        """New import should not emit warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # This should not raise DeprecationWarning
            from foodspec.core.preprocessing.baseline import baseline_als
            result = baseline_als(sample_spectrum)
            assert len(result) == len(sample_spectrum)
    
    def test_top_level_import_works(self, sample_spectrum):
        """Top-level import should work."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Top-level import should not emit warnings
            from foodspec import baseline_als
            result = baseline_als(sample_spectrum)
            assert len(result) == len(sample_spectrum)
    
    def test_deprecated_function_produces_same_result(self, sample_spectrum):
        """Old and new imports should produce identical results."""
        
        # Get result from old path (suppress warning for this test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from foodspec.old_module import baseline_als as old_func
            result_old = old_func(sample_spectrum, lam=1e4)
        
        # Get result from new path
        from foodspec.core.preprocessing.baseline import baseline_als as new_func
        result_new = new_func(sample_spectrum, lam=1e4)
        
        # Results should be identical
        np.testing.assert_array_equal(result_old, result_new)
    
    def test_module_getattr_migration_map(self):
        """Test __getattr__ based module migration."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            from foodspec.old_legacy_module import baseline_als
            assert callable(baseline_als)
    
    def test_dir_on_deprecated_module(self):
        """Test that __dir__ works on deprecated module."""
        import foodspec.old_legacy_module as mod
        names = dir(mod)
        assert 'baseline_als' in names
        assert 'smooth_spectrum' in names
```

---

## Example 7: Deprecation Test in CI/CD

**File: `.github/workflows/deprecation-check.yml`**

```yaml
name: Deprecation Warnings Check

on: [push, pull_request]

jobs:
  check-deprecations:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Test backward compatibility
        run: pytest tests/test_backward_compat.py -v
      
      - name: Check for unexpected deprecation warnings
        run: |
          # Run tests and fail if ANY deprecation warnings are emitted
          # (except for intentional deprecations in test_backward_compat.py)
          pytest tests/ -W error::DeprecationWarning --ignore=tests/test_backward_compat.py -v
```

---

## Example 8: Migration Guide Template

**File: `docs/migration/baseline_functions.md`**

```markdown
# Migrating Baseline Functions (v1.1.0 → v1.2.0)

In FoodSpec v1.1.0, baseline subtraction functions moved from `foodspec.spectral_dataset` to `foodspec.core.preprocessing.baseline`.

## Quick Start

### Before (v1.0.x)

```python
from foodspec.spectral_dataset import baseline_als, baseline_polynomial

spectrum = load_spectrum("my_data.csv")
corrected = baseline_als(spectrum, lam=1e4)
```

### After (v1.1.0+)

```python
from foodspec.core.preprocessing.baseline import baseline_als, baseline_polynomial
# OR (recommended for end users)
from foodspec import baseline_als, baseline_polynomial

spectrum = load_spectrum("my_data.csv")
corrected = baseline_als(spectrum, lam=1e4)
```

## Function Mapping

| Old Location | New Location |
|---|---|
| `foodspec.spectral_dataset.baseline_als` | `foodspec.core.preprocessing.baseline.baseline_als` or `foodspec.baseline_als` |
| `foodspec.spectral_dataset.baseline_polynomial` | `foodspec.core.preprocessing.baseline.baseline_polynomial` or `foodspec.baseline_polynomial` |
| `foodspec.spectral_dataset.baseline_rubberband` | `foodspec.core.preprocessing.baseline.baseline_rubberband` or `foodspec.baseline_rubberband` |

## Why This Change?

- **Organization**: Grouping related functions in `core.preprocessing` for clarity
- **Scalability**: New preprocessing modules (smoothing, normalization) also live here
- **Future-proof**: Aligns with protocol-driven architecture

## Timeline

- **v1.1.0**: New location available, old imports emit `DeprecationWarning`
- **v1.2.0+**: Both locations continue to work
- **v2.0.0**: Old imports removed

## Common Issues

**Q: I'm getting `DeprecationWarning` — should I update now?**  
A: It's recommended but not urgent. You have until v2.0.0. See [When to Update](#when-to-update).

**Q: How do I suppress the warning temporarily?**  
A: Use this compatibility shim:
```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='foodspec')

# Now old imports work without warnings (temporary)
from foodspec.spectral_dataset import baseline_als
```

**When to Update**

1. **Soon** (recommended): Update before v2.0.0 release
2. **Now** (if you distribute packages that depend on FoodSpec): Update to avoid warnings for your users
3. **Later** (if casual user): Update when you upgrade to v2.0.0
```

---

## Summary Checklist for Implementation

When refactoring a module, follow this checklist:

- [ ] Create new implementation in `src/foodspec/core/*`
- [ ] Update `src/foodspec/__init__.py` to import from new location
- [ ] Create legacy re-export module (e.g., `src/foodspec/old_module.py`)
- [ ] All re-exports emit `DeprecationWarning` with clear migration path
- [ ] Add test in `tests/test_backward_compat.py`
- [ ] Document in RELEASE_NOTES_v*.md
- [ ] Add migration guide in `docs/migration/`
- [ ] Update docstrings with deprecation notices
- [ ] Run CI/CD to verify no unexpected warnings

---

**See also**: [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md), [ENGINEERING_RULES.md](./ENGINEERING_RULES.md)
