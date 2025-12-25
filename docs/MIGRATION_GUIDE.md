# FoodSpec Module Migration Guide

**Last Updated:** December 25, 2025

## Overview

FoodSpec has undergone a codebase reorganization to improve structure and maintainability. Several modules have been relocated to more appropriate subpackages. **All old import paths remain functional** with deprecation warnings to ensure backward compatibility.

---

## Relocated Modules

### 1. Spectral Dataset → Core

**Old Location:** `foodspec.spectral_dataset`  
**New Location:** `foodspec.core.spectral_dataset`

**Migration:**

```python
# ❌ Old (deprecated, but still works)
from foodspec.spectral_dataset import SpectralDataset, HyperspectralDataset

# ✅ New (recommended)
from foodspec.core.spectral_dataset import SpectralDataset, HyperspectralDataset
```

**Affected Classes/Functions:**
- `SpectralDataset`
- `HyperspectralDataset`
- `PreprocessingConfig`
- `harmonize_datasets()`
- `baseline_als()`, `baseline_rubberband()`, `baseline_polynomial()`

---

### 2. RQ Engine → Features

**Old Location:** `foodspec.rq`  
**New Location:** `foodspec.features.rq`

**Migration:**

```python
# ❌ Old (deprecated, but still works)
from foodspec.rq import PeakDefinition, RatioDefinition, RQConfig, RatioQualityEngine

# ✅ New (recommended)
from foodspec.features.rq import PeakDefinition, RatioDefinition, RQConfig, RatioQualityEngine
```

**Affected Classes:**
- `PeakDefinition`
- `RatioDefinition`
- `RQConfig`
- `RatioQualityEngine`
- `RatioQualityResult`

---

### 3. Matrix Correction → Preprocess

**Old Location:** `foodspec.matrix_correction`  
**New Location:** `foodspec.preprocess.matrix_correction`

**Migration:**

```python
# ❌ Old (deprecated, but still works)
from foodspec.matrix_correction import apply_matrix_correction

# ✅ New (recommended)
from foodspec.preprocess.matrix_correction import apply_matrix_correction
```

**Affected Functions:**
- `apply_matrix_correction()`
- Matrix effect correction utilities

---

### 4. Calibration Transfer → Preprocess

**Old Location:** `foodspec.calibration_transfer`  
**New Location:** `foodspec.preprocess.calibration_transfer`

**Migration:**

```python
# ❌ Old (deprecated, but still works)
from foodspec.calibration_transfer import direct_standardization, piecewise_direct_standardization

# ✅ New (recommended)
from foodspec.preprocess.calibration_transfer import direct_standardization, piecewise_direct_standardization
```

**Affected Functions:**
- `direct_standardization()`
- `piecewise_direct_standardization()`
- `detect_drift()`
- `adapt_calibration_incremental()`
- `calibration_transfer_workflow()`

---

### 5. Heating Trajectory → Workflows

**Old Location:** `foodspec.heating_trajectory`  
**New Location:** `foodspec.workflows.heating_trajectory`

**Migration:**

```python
# ❌ Old (deprecated, but still works)
from foodspec.heating_trajectory import analyze_heating_trajectory

# ✅ New (recommended)
from foodspec.workflows.heating_trajectory import analyze_heating_trajectory
```

**Affected Functions:**
- `analyze_heating_trajectory()`
- Oxidation index extraction and trajectory modeling

---

## Migration Strategy

### For Existing Code

**Option 1: Gradual Migration (Recommended)**
1. Update imports one module at a time
2. Test after each change
3. Commit incrementally

**Option 2: Keep Old Imports**
- Old imports will continue to work indefinitely
- Deprecation warnings can be suppressed if needed:
  ```python
  import warnings
  warnings.filterwarnings("ignore", category=DeprecationWarning, module="foodspec")
  ```

### For New Code

**Always use new import paths** to avoid technical debt:

```python
# ✅ Correct for new code
from foodspec.core.spectral_dataset import SpectralDataset
from foodspec.features.rq import RatioQualityEngine
from foodspec.preprocess.calibration_transfer import direct_standardization
from foodspec.workflows.heating_trajectory import analyze_heating_trajectory
```

---

## Timeline

- **December 25, 2025:** Module relocations completed with backward-compatible shims
- **Future Releases:** Deprecation warnings will remain for at least 2 major versions
- **Removal Timeline:** Old import paths will be removed no earlier than v2.0.0 (TBD)

---

## Automated Migration Script

For codebases with many files, use this script to update imports automatically:

```bash
#!/bin/bash
# update_imports.sh

# Update spectral_dataset imports
find . -type f -name "*.py" -exec sed -i 's/from foodspec\.spectral_dataset/from foodspec.core.spectral_dataset/g' {} +

# Update rq imports
find . -type f -name "*.py" -exec sed -i 's/from foodspec\.rq import/from foodspec.features.rq import/g' {} +

# Update matrix_correction imports
find . -type f -name "*.py" -exec sed -i 's/from foodspec\.matrix_correction/from foodspec.preprocess.matrix_correction/g' {} +

# Update calibration_transfer imports
find . -type f -name "*.py" -exec sed -i 's/from foodspec\.calibration_transfer/from foodspec.preprocess.calibration_transfer/g' {} +

# Update heating_trajectory imports
find . -type f -name "*.py" -exec sed -i 's/from foodspec\.heating_trajectory/from foodspec.workflows.heating_trajectory/g' {} +

echo "✅ Import paths updated. Please review changes and run tests."
```

---

## Need Help?

- **Documentation:** See updated API docs in [docs/api/](api/index.md)
- **Issues:** Report migration problems on [GitHub Issues](https://github.com/chandrasekarnarayana/foodspec/issues)
- **Discussions:** Ask questions in [GitHub Discussions](https://github.com/chandrasekarnarayana/foodspec/discussions)

---

## API Stability Guarantee

- **High-Level API (`foodspec.FoodSpec`)**: Stable, no changes required
- **CLI Commands**: Stable, no changes required
- **Protocol YAML**: Stable, no changes required
- **Internal Imports**: Changed as documented above, shims provided

Your production pipelines and published analyses remain functional without modification.
