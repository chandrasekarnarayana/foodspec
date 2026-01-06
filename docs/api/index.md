# API Reference

Complete API documentation for FoodSpec Python modules with auto-generated signatures and examples.

## Module Organization

### Data Management

| Module | Purpose | Key Components |
|--------|---------|----------------|
| [**Core**](core.md) | Data structures and workflows | `FoodSpectrumSet`, `Spectrum`, `OutputBundle` |
| [**IO**](io.md) | Loading and saving data | `load_folder()`, `read_spectra()`, `detect_format()` |
| [**Datasets**](datasets.md) | Bundled example data | `load_olive_oils()`, `load_meat_samples()` |

### Data Processing

| Module | Purpose | Key Components |
|--------|---------|----------------|
| [**Preprocessing**](preprocessing.md) | Spectral preprocessing | `baseline_als()`, `normalize_snv()` |
| [**Features**](features.md) | Feature extraction | `detect_peaks()`, `RatioQualityEngine` |

### Analysis & Modeling

| Module | Purpose | Key Components |
|--------|---------|----------------|
| [**Chemometrics**](chemometrics.md) | PCA, PLS, MCR-ALS | `run_pca()`, `make_pls_da()` |
| [**ML**](ml.md) | Model training & validation | `nested_cross_validate()` |
| [**Stats**](stats.md) | Hypothesis testing | `run_ttest()`, `run_anova()` |
| [**Metrics**](metrics.md) | Evaluation metrics | `compute_classification_metrics()` |

### Applications

| Module | Purpose | Key Components |
|--------|---------|----------------|
| [**Workflows**](workflows.md) | High-level workflows | Domain-specific analysis |

## See Also

- **[Methods Guide](../methods/index.md)** - Detailed methodology documentation
- **[Examples Gallery](../examples_gallery.md)** - Runnable code examples
- **[User Guide](../user-guide/index.md)** - Usage patterns and best practices
