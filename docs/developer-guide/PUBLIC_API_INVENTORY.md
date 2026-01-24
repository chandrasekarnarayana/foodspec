# FoodSpec Public API Inventory

**Status**: Active  
**Version**: 1.0.0  
**Last Updated**: 2026-01-24

This document is the **definitive list** of all public APIs that must remain stable and importable during the FoodSpec refactor. Any changes to these require a **major version bump** (e.g., v1.0 → v2.0) and a **deprecation cycle** of at least one minor version.

---

## Table of Contents

1. [Top-Level Exports](#top-level-exports-from-foodspec)
2. [Sub-Module Exports](#sub-module-exports)
3. [Class/Function Index](#classifunction-index-with-locations)
4. [Stability Guarantees](#stability-guarantees)
5. [Adding to Public API](#adding-to-public-api)
6. [Deprecation Process](#deprecation-process)

---

## Top-Level Exports (from `foodspec`)

All of these **must** remain importable from `foodspec` package root:

```python
from foodspec import (
    # Version info
    __version__,
    
    # ========== CORE CLASSES ==========
    FoodSpec,                 # Unified entry point (v1.1+)
    Spectrum,                 # Single spectrum representation
    FoodSpectrumSet,          # Set of spectra (legacy)
    HyperSpectralCube,        # 3D hyperspectral data
    OutputBundle,             # Result container
    RunRecord,                # Provenance/metadata
    
    # ========== I/O FUNCTIONS ==========
    load_folder,
    load_library,
    create_library,
    load_csv_spectra,
    read_spectra,
    detect_format,
    
    # ========== PREPROCESSING ==========
    baseline_als,
    baseline_polynomial,
    baseline_rubberband,
    harmonize_datasets,
    
    # ========== QUALITY CONTROL ==========
    estimate_snr,
    summarize_class_balance,
    detect_outliers,
    check_missing_metadata,
    
    # ========== STATISTICS & METRICS ==========
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
    
    # ========== SYNTHETIC DATA ==========
    generate_synthetic_raman_spectrum,
    generate_synthetic_ftir_spectrum,
    
    # ========== ADVANCED FEATURES (MOATS) ==========
    apply_matrix_correction,
    analyze_heating_trajectory,
    calibration_transfer_workflow,
    direct_standardization,
    piecewise_direct_standardization,
    
    # ========== DATASET INTELLIGENCE ==========
    summarize_dataset,
    check_class_balance,
    diagnose_imbalance,
    compute_replicate_consistency,
    assess_variability_sources,
    detect_batch_label_correlation,
    detect_replicate_leakage,
    detect_leakage,
    compute_readiness_score,
    
    # ========== RATIO QUALITY (RQ) ENGINE ==========
    PeakDefinition,
    RatioDefinition,
    RQConfig,
    RatioQualityEngine,
    RatioQualityResult,
    
    # ========== HYPERSPECTRAL DATASETS ==========
    HDF5_SCHEMA_VERSION,
    HyperspectralDataset,
    PreprocessingConfig,
    SpectralDataset,
    
    # ========== REPRODUCIBILITY ==========
    DatasetSpec,
    ExperimentConfig,
    ExperimentEngine,
    diff_runs,
    
    # ========== ARTIFACT MANAGEMENT ==========
    Predictor,
    save_artifact,
    load_artifact,
    
    # ========== PLUGIN SYSTEM ==========
    PluginManager,
    install_plugin,
    load_plugins,
    
    # ========== UTILITIES ==========
    LOGO_BASE64,
    get_logo_base64,
    get_logo_bytes,
    save_logo,
)
```

---

## Sub-Module Exports

These are also stable and may be imported from sub-modules:

### `foodspec.io`
```python
from foodspec.io import (
    load_folder,
    load_library,
    create_library,
    load_csv_spectra,
    read_spectra,
    detect_format,
)
```

### `foodspec.preprocessing` or `foodspec.core.preprocessing.*`
```python
from foodspec.preprocessing import (  # if this path exists
    baseline_als,
    baseline_polynomial,
    baseline_rubberband,
    normalize,
    smooth,
    # ... others
)

# OR (after refactor)
from foodspec.core.preprocessing.baseline import baseline_als
```

### `foodspec.stats`
```python
from foodspec.stats import (
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
```

### `foodspec.qc` (Quality Control)
```python
from foodspec.qc import (
    estimate_snr,
    summarize_class_balance,
    detect_outliers,
    check_missing_metadata,
    check_class_balance,
    diagnose_imbalance,
    compute_replicate_consistency,
    assess_variability_sources,
    detect_batch_label_correlation,
    detect_replicate_leakage,
    detect_leakage,
    compute_readiness_score,
)
```

### `foodspec.repro` (Reproducibility)
```python
from foodspec.repro import (
    DatasetSpec,
    ExperimentConfig,
    ExperimentEngine,
    diff_runs,
)
```

### `foodspec.features.rq` (Ratio Quality)
```python
from foodspec.features.rq import (
    PeakDefinition,
    RatioDefinition,
    RQConfig,
    RatioQualityEngine,
    RatioQualityResult,
)
```

### `foodspec.core`
```python
from foodspec.core import (
    FoodSpec,
    Spectrum,
    RunRecord,
    OutputBundle,
)

from foodspec.core.dataset import FoodSpectrumSet, SpectralDataset
from foodspec.core.hyperspectral import HyperSpectralCube, HyperspectralDataset
```

---

## Class/Function Index with Locations

This table shows **current (v1.0.0) locations**. During refactor, new locations will be added and old ones deprecated.

| Symbol | Type | Current Location | Status | Notes |
|--------|------|------------------|--------|-------|
| `FoodSpec` | Class | `foodspec.core.api` | STABLE | Phase 1 entry point |
| `Spectrum` | Class | `foodspec.core.spectrum` | STABLE | Single spectrum |
| `FoodSpectrumSet` | Class | `foodspec.core.dataset` | STABLE | Legacy, for compat |
| `HyperSpectralCube` | Class | `foodspec.core.hyperspectral` | STABLE | 3D data |
| `OutputBundle` | Class | `foodspec.core.output_bundle` | STABLE | Result container |
| `RunRecord` | Class | `foodspec.core.run_record` | STABLE | Provenance |
| `load_folder` | Function | `foodspec.io` | STABLE | Load from directory |
| `load_library` | Function | `foodspec.io` | STABLE | Load library |
| `create_library` | Function | `foodspec.io` | STABLE | Create library |
| `load_csv_spectra` | Function | `foodspec.io` | STABLE | CSV import |
| `read_spectra` | Function | `foodspec.io` | STABLE | Generic import |
| `detect_format` | Function | `foodspec.io` | STABLE | Format detection |
| `baseline_als` | Function | `foodspec.spectral_dataset` → `foodspec.core.preprocessing.baseline` | MOVED | ALS baseline |
| `baseline_polynomial` | Function | `foodspec.spectral_dataset` → `foodspec.core.preprocessing.baseline` | MOVED | Poly baseline |
| `baseline_rubberband` | Function | `foodspec.spectral_dataset` → `foodspec.core.preprocessing.baseline` | MOVED | Rubberband |
| `harmonize_datasets` | Function | `foodspec.core.spectral_dataset` | STABLE | Dataset alignment |
| `HyperspectralDataset` | Class | `foodspec.core.spectral_dataset` | STABLE | HSI container |
| `PreprocessingConfig` | Class | `foodspec.core.spectral_dataset` | STABLE | Preprocessing config |
| `SpectralDataset` | Class | `foodspec.core.spectral_dataset` | STABLE | Spectral data |
| `HDF5_SCHEMA_VERSION` | Constant | `foodspec.core.spectral_dataset` | STABLE | Schema version |
| `compute_classification_metrics` | Function | `foodspec.metrics` | STABLE | Metrics |
| `compute_regression_metrics` | Function | `foodspec.metrics` | STABLE | Metrics |
| `compute_roc_curve` | Function | `foodspec.metrics` | STABLE | ROC curve |
| `compute_pr_curve` | Function | `foodspec.metrics` | STABLE | PR curve |
| `run_anova` | Function | `foodspec.stats` | STABLE | ANOVA |
| `run_ttest` | Function | `foodspec.stats` | STABLE | T-test |
| `run_manova` | Function | `foodspec.stats` | STABLE | MANOVA |
| `run_tukey_hsd` | Function | `foodspec.stats` | STABLE | Tukey |
| `run_kruskal_wallis` | Function | `foodspec.stats` | STABLE | Kruskal-Wallis |
| `run_mannwhitney_u` | Function | `foodspec.stats` | STABLE | Mann-Whitney |
| `run_wilcoxon_signed_rank` | Function | `foodspec.stats` | STABLE | Wilcoxon |
| `run_friedman_test` | Function | `foodspec.stats` | STABLE | Friedman |
| `bootstrap_metric` | Function | `foodspec.stats` | STABLE | Bootstrap |
| `permutation_test_metric` | Function | `foodspec.stats` | STABLE | Permutation test |
| `estimate_snr` | Function | `foodspec.utils.troubleshooting` | STABLE | SNR estimation |
| `summarize_class_balance` | Function | `foodspec.utils.troubleshooting` | STABLE | Balance summary |
| `detect_outliers` | Function | `foodspec.utils.troubleshooting` | STABLE | Outlier detection |
| `check_missing_metadata` | Function | `foodspec.utils.troubleshooting` | STABLE | Metadata check |
| `generate_synthetic_raman_spectrum` | Function | `foodspec.synthetic` | STABLE | Synthetic data |
| `generate_synthetic_ftir_spectrum` | Function | `foodspec.synthetic` | STABLE | Synthetic data |
| `PeakDefinition` | Class | `foodspec.features.rq` | STABLE | Peak config |
| `RatioDefinition` | Class | `foodspec.features.rq` | STABLE | Ratio config |
| `RQConfig` | Class | `foodspec.features.rq` | STABLE | RQ config |
| `RatioQualityEngine` | Class | `foodspec.features.rq` | STABLE | RQ engine |
| `RatioQualityResult` | Class | `foodspec.features.rq` | STABLE | RQ result |
| `apply_matrix_correction` | Function | `foodspec.matrix_correction` | STABLE | Matrix correction |
| `analyze_heating_trajectory` | Function | `foodspec.heating_trajectory` | STABLE | Heating analysis |
| `calibration_transfer_workflow` | Function | `foodspec.calibration_transfer` | STABLE | Calibration transfer |
| `direct_standardization` | Function | `foodspec.calibration_transfer` | STABLE | Standardization |
| `piecewise_direct_standardization` | Function | `foodspec.calibration_transfer` | STABLE | Piecewise |
| `summarize_dataset` | Function | `foodspec.core.summary` | STABLE | Dataset summary |
| `check_class_balance` | Function | `foodspec.qc.dataset_qc` | STABLE | Class balance |
| `diagnose_imbalance` | Function | `foodspec.qc.dataset_qc` | STABLE | Imbalance diagnosis |
| `compute_replicate_consistency` | Function | `foodspec.qc.replicates` | STABLE | Replicate QC |
| `assess_variability_sources` | Function | `foodspec.qc.replicates` | STABLE | Variability |
| `detect_batch_label_correlation` | Function | `foodspec.qc.leakage` | STABLE | Leakage detection |
| `detect_replicate_leakage` | Function | `foodspec.qc.leakage` | STABLE | Leakage detection |
| `detect_leakage` | Function | `foodspec.qc.leakage` | STABLE | Leakage detection |
| `compute_readiness_score` | Function | `foodspec.qc.readiness` | STABLE | Readiness check |
| `DatasetSpec` | Class | `foodspec.repro` | STABLE | Dataset spec |
| `ExperimentConfig` | Class | `foodspec.repro` | STABLE | Experiment config |
| `ExperimentEngine` | Class | `foodspec.repro` | STABLE | Experiment engine |
| `diff_runs` | Function | `foodspec.repro` | STABLE | Run comparison |
| `Predictor` | Class | `foodspec.artifact` | STABLE | Model predictor |
| `save_artifact` | Function | `foodspec.artifact` | STABLE | Save model |
| `load_artifact` | Function | `foodspec.artifact` | STABLE | Load model |
| `PluginManager` | Class | `foodspec.plugin` | STABLE | Plugin system |
| `install_plugin` | Function | `foodspec.plugin` | STABLE | Install plugin |
| `load_plugins` | Function | `foodspec.plugins` | STABLE | Load plugins |
| `LOGO_BASE64` | Constant | `foodspec.logo` | STABLE | Logo data |
| `get_logo_base64` | Function | `foodspec.logo` | STABLE | Get logo |
| `get_logo_bytes` | Function | `foodspec.logo` | STABLE | Get logo bytes |
| `save_logo` | Function | `foodspec.logo` | STABLE | Save logo |

---

## Stability Guarantees

### STABLE Status (v1.0+)
- **API Contract**: Signature, behavior, and return type are guaranteed.
- **Location**: Can be moved internally, but must remain importable from original and/or top-level.
- **Deprecation**: If removed, requires ≥1 minor version with `DeprecationWarning`.
- **Testing**: Must have comprehensive tests that never change semantically.

### MOVED Status
- **Old Location**: Still works, emits `DeprecationWarning`.
- **New Location**: Recommended import path.
- **Timeline**: Both locations available for ≥1 minor version.
- **Example**: `baseline_als` moved from `foodspec.spectral_dataset` to `foodspec.core.preprocessing.baseline`.

### EXPERIMENTAL Status
- **Not in this list**: Experimental APIs are not part of the public surface.
- **Prefix**: Usually prefixed with `_` (private) or in `_experimental` module.
- **Policy**: Can change without warning, no deprecation needed.

---

## Adding to Public API

To add a new function/class to the public API:

1. **Design & Review**: Open issue/discussion proposing the API
2. **Implement**: Add function/class with full docstring + examples
3. **Test**: ≥80% coverage, including edge cases
4. **Document**: Add to `docs/api/` and relevant user guide
5. **Update This File**: Add to relevant section above
6. **Update RELEASE_NOTES.md**: Note the new API
7. **Update CONTRIBUTING.md**: Mention the new feature
8. **Merge**: Once approved

### Template

```python
def my_new_public_function(param1: int, param2: str = "default") -> dict:
    """One-line summary.
    
    Longer description explaining purpose and use case.
    
    Parameters
    ----------
    param1 : int
        First parameter description.
    param2 : str, default "default"
        Second parameter description.
    
    Returns
    -------
    dict
        Result with keys ...
    
    Raises
    ------
    ValueError
        If parameters are invalid.
    
    Examples
    --------
    >>> result = my_new_public_function(42, param2="custom")
    >>> result['key']
    'value'
    
    Notes
    -----
    This is a new public API added in v1.1.0.
    """
```

---

## Deprecation Process

When deprecating a public API:

### Step 1: Add Deprecation Warning (Minor Version)

```python
import warnings

def old_function():
    """Deprecated function.
    
    .. deprecated:: 1.1.0
        Use :func:`new_function` instead.
        This function will be removed in v2.0.0.
    """
    warnings.warn(
        "old_function is deprecated and will be removed in v2.0.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

### Step 2: Document in RELEASE_NOTES.md

```markdown
## v1.1.0 Deprecations

The following APIs are deprecated and will be removed in v2.0.0:

| API | Replacement | Notes |
|-----|-------------|-------|
| `old_function` | `new_function` | Signature identical, just move internally |
```

### Step 3: Update This Inventory

Mark as `DEPRECATED` in the status column.

### Step 4: Provide Migration Guide

Create `docs/migration/old_function.md` with examples.

### Step 5: Remove (Major Version)

In v2.0.0 (or later major), remove the deprecated code entirely:
- Delete the wrapper function
- Update this inventory
- Document removal in RELEASE_NOTES_v2.0.0.md

---

## Checking Your Code

Use this checklist to verify API stability:

- [ ] Is this function/class in the "Public API" lists above?
- [ ] If adding new: Have I followed "Adding to Public API" steps?
- [ ] If modifying: Is this a breaking change? If yes, deprecate first.
- [ ] If moving: Did I add re-export wrapper in old location?
- [ ] If re-exporting: Does it emit `DeprecationWarning`?
- [ ] Have I updated this document?
- [ ] Have I updated RELEASE_NOTES.md?
- [ ] Have I updated CONTRIBUTING.md if needed?

---

## Links

- 📖 [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) — Full rules
- 🤝 [CONTRIBUTING.md](../../CONTRIBUTING.md) — Contributing guide
- 🔄 [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md) — Compatibility strategy
- 💡 [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md) — Implementation examples

---

**Maintained by**: FoodSpec Core Team  
**Last Updated**: 2026-01-24  
**Questions?** Open an issue or contact chandrasekarnarayana@gmail.com
