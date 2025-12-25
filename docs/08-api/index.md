# API Reference

!!! info "Context Block"
    **Purpose:** Complete API documentation for all FoodSpec modules with auto-generated function signatures, parameters, examples, and theory links.
    
    **Audience:** Developers, power users, contributors
    
    **Prerequisites:** Basic Python knowledge, understanding of spectroscopy concepts
    
    **Related:** [Architecture](../05-advanced-topics/architecture.md) | [Developer Guide](../06-developer-guide/contributing.md)

---

## Overview

The FoodSpec API is organized by module, mirroring the codebase structure. Each module page provides:

- **Complete function/class signatures** with type hints
- **Parameter descriptions** with default values and valid ranges
- **Return value documentation** with types
- **Working examples** demonstrating typical usage
- **Theory links** to relevant documentation pages
- **Metric interpretation** for functions returning performance metrics

---

## Module Organization

### Core Modules

**Registry & Provenance**
- `FeatureModelRegistry`: Model and run tracking
- `RegistryEntry`: Immutable provenance records
- `_hash_dataset()`: Dataset fingerprinting for reproducibility

**Output & Reporting**
- `create_run_folder()`: Timestamped run directory creation
- `save_tables()`, `save_figures()`, `save_metadata()`: Result persistence
- `append_log()`: Audit trail logging

---

### Preprocessing

**Baseline Correction**
- `ALSBaseline`: Asymmetric Least Squares (Eilers 2005)
- `RubberbandBaseline`: Convex hull method
- `PolynomialBaseline`: Polynomial fitting
- Theory: [Baseline Correction](../preprocessing/baseline_correction/)

**Normalization & Scaling**
- `SNVNormalization`: Standard Normal Variate
- `MSCCorrection`: Multiplicative Scatter Correction
- `VectorNormalization`: L1/L2 normalization
- Theory: [Normalization & Smoothing](../preprocessing/normalization_smoothing/)

**Derivatives & Smoothing**
- `SavitzkyGolay`: Savitzky-Golay smoothing/derivatives
- Derivative order selection guide
- Theory: [Derivatives & Feature Enhancement](../preprocessing/derivatives_and_feature_enhancement/)

---

### Feature Extraction

**RQ Engine**
- `RatioQualityEngine`: Ratiometric feature analysis
- `PeakDefinition`, `RatioDefinition`: Feature configuration
- Quality metrics: stability, discrimination, heating trends
- Theory: [RQ Engine Theory](../07-theory-and-background/rq_engine_theory.md)

**Peak Detection**
- Peak finding algorithms
- Integration methods
- Band assignment utilities

---

### Machine Learning

**Models**
- PLS-DA, PLS-R: Partial Least Squares
- Random Forest, SVM: Ensemble and kernel methods
- Model selection criteria
- Theory: [Chemometrics & ML Basics](../07-theory-and-background/chemometrics_and_ml_basics.md)

**Evaluation Metrics**
- Classification: AUC, F1, balanced accuracy
- Regression: R², RMSE, Q²
- Interpretation: [Metric Significance Tables](../09-reference/metric_significance_tables.md)

**Cross-Validation**
- Stratified K-fold
- Nested CV for hyperparameter tuning
- Leave-one-group-out for batch effects
- Theory: [Model Evaluation & Validation](../ml/model_evaluation_and_validation.md)

---

### Statistics

**Hypothesis Tests**
- t-tests: paired, independent, Welch's
- ANOVA: one-way, two-way, MANOVA
- Non-parametric: Mann-Whitney, Kruskal-Wallis
- Theory: [Hypothesis Testing](../stats/hypothesis_testing_in_food_spectroscopy.md)

**Effect Sizes**
- Cohen's d: standardized mean difference
- η² (eta-squared): ANOVA effect size
- Interpretation: [Metric Significance Tables](../09-reference/metric_significance_tables.md)
- Theory: [T-tests & Effect Sizes](../stats/t_tests_effect_sizes_and_power.md)

---

### Data I/O

**File Readers**
- OPUS (Bruker FTIR), SPC (Thermo/Galactic)
- CSV, Excel: tabular data
- HDF5: FoodSpec native format

**Export Utilities**
- CSV export with metadata
- JCAMP-DX format
- Interoperability considerations

---

## Auto-Generated Documentation

This API reference is generated using [mkdocstrings](https://mkdocstrings.github.io/) which extracts docstrings directly from source code. If you find missing or outdated documentation, please:

1. Check the source code for the actual function signature
2. Open an issue on [GitHub](https://github.com/chandrasekarnarayana/foodspec/issues)
3. Contribute improved docstrings via pull request

---

## Example: Using the API

### Basic Workflow

```python
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.features.rq import RatioQualityEngine, PeakDefinition, RatioDefinition
from foodspec.registry import FeatureModelRegistry
from foodspec.output_bundle import create_run_folder, save_metadata
from pathlib import Path
import pandas as pd

# 1. Load data (assume df has spectral columns + metadata)
df = pd.read_csv("data/oils.csv")

# 2. Preprocess: ALS baseline correction
als = ALSBaseline(lambda_=1e5, p=0.001)
X = df.filter(like="wn_").values  # Spectral columns
X_corrected = als.fit_transform(X)
df_corrected = df.copy()
df_corrected[df.filter(like="wn_").columns] = X_corrected

# 3. Feature extraction: RQ ratios
peaks = [
    PeakDefinition(name="carbonyl", wavenumber=1743, tolerance=10),
    PeakDefinition(name="cis_bend", wavenumber=1654, tolerance=10),
]
ratios = [
    RatioDefinition(name="rq_1743_1654", numerator="carbonyl", denominator="cis_bend")
]
rq_engine = RatioQualityEngine(peaks=peaks, ratios=ratios)
result = rq_engine.run_all(df_corrected)

# 4. Save results with provenance
run_dir = create_run_folder(Path("outputs"))
save_metadata(run_dir, {
    "protocol": "oil_authentication",
    "protocol_version": "2.0.0",
    "preprocessing": {"baseline": "ALS", "lambda": 1e5, "p": 0.001},
    "features": [r.name for r in ratios],
})

# 5. Register in model registry
registry = FeatureModelRegistry(Path("registry.json"))
registry.register_run(run_id=run_dir.name, metadata={
    "dataset_hash": _hash_dataset(df),
    "protocol": "oil_authentication",
    "protocol_version": "2.0.0",
    "preprocessing": {"baseline": "ALS"},
    "features": [{"name": r.name, "type": "ratio"} for r in ratios],
    "timestamp": "2025-12-25T14:30:00Z",
})

print(f"Results saved to {run_dir}")
print(f"Registry updated: {len(registry.entries)} total runs")
```

---

## See Also

- **[Architecture](../05-advanced-topics/architecture.md)** — System design overview
- **[Extending Protocols](../06-developer-guide/extending_protocols_and_steps.md)** — Adding custom preprocessing/features
- **[Writing Plugins](../06-developer-guide/writing_plugins.md)** — Plugin system API
- **[Metric Significance Tables](../09-reference/metric_significance_tables.md)** — Interpreting API-returned metrics
