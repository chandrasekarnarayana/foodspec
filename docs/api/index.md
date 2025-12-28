# API Reference: FoodSpec Python Modules

Auto‑generated documentation for IO, Core, Preprocessing, Features, ML, Stats and Workflows modules.

!!! info "Module Purpose"
    Complete API documentation for all FoodSpec modules with auto-generated signatures, examples, and cross-references.

---

## Quick Start

**New to FoodSpec?** Start with these essential modules:

1. **[IO](io.md)** - Load data: `load_folder()`, `load_from_metadata_table()`
2. **[Core](core.md)** - Understand: `FoodSpectrumSet`, `FoodSpec`
3. **[Preprocessing](preprocessing.md)** - Clean data: `PreprocessPipeline`, `BaselineStep`
4. **[ML](ml.md)** - Build models: `nested_cross_validate()`, `train_classifier()`

---

## Module Organization

### Data Management

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| [**Core**](core.md) | Data structures and workflows | `FoodSpectrumSet`, `OutputBundle` |
| [**IO**](io.md) | Loading and saving data | `load_folder()`, `to_hdf5()` |
| [**Datasets**](datasets.md) | Bundled example data | `load_olive_oils()` |

### Data Processing

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| [**Preprocessing**](preprocessing.md) | Spectral preprocessing | `PreprocessPipeline`, `BaselineStep` |
| [**Features**](features.md) | Feature extraction | `detect_peaks()`, `RatioQualityEngine` |

### Analysis & Modeling

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| [**Chemometrics**](chemometrics.md) | PCA, PLS, MCR-ALS | `run_pca()`, `make_pls_da()` |
| [**ML**](ml.md) | Model training & validation | `nested_cross_validate()` |
| [**Stats**](stats.md) | Hypothesis testing | `run_ttest()`, `run_anova()` |
| [**Metrics**](metrics.md) | Evaluation metrics | `accuracy_score()`, `r2_score()` |

### Applications

| Module | Purpose | Key Workflows |
|--------|---------|--------------|
| [**Workflows**](workflows.md) | High-level workflows | Oil auth, heating quality, QC |

---

## Common Workflows

### Complete Analysis Pipeline

```python
from foodspec.io import load_folder
from foodspec.preprocess import PreprocessPipeline, BaselineStep, NormalizationStep
from foodspec.chemometrics import make_pls_da
from foodspec.ml import nested_cross_validate

# 1. Load data
fs = load_folder('data/oils/')

# 2. Preprocess
pipeline = PreprocessPipeline(steps=[
    BaselineStep(method='als', lam=1e4),
    NormalizationStep(method='snv')
])
fs_clean = pipeline.transform(fs)

# 3. Build and validate model
clf = make_pls_da(n_components=5)
results = nested_cross_validate(fs_clean, clf, target_col='variety')

print(f"Accuracy: {results['test_accuracy'].mean():.3f}")
```

### Feature-Based Analysis

```python
from foodspec.features import detect_peaks, compute_ratios, RatioQualityEngine

# Extract features
peaks = detect_peaks(fs[0].intensity, fs.wavenumbers, height=0.1)
ratios = compute_ratios(fs, peak_positions={'A': 1650, 'B': 1450})

# Or use RQ engine for automated workflow
from foodspec.features import RQConfig, PeakDefinition
config = RQConfig(peaks=[
    PeakDefinition(name='lipid', center=2850, window=20),
    PeakDefinition(name='protein', center=1650, window=15)
])
engine = RatioQualityEngine(config)
result = engine.analyze(fs)
```

### Regression Analysis

```python
from foodspec.chemometrics import make_pls_regression, calculate_vip
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Train PLS regression
pls = make_pls_regression(n_components=10)
pls.fit(X_train, y_train)

# Predict and evaluate
y_pred = pls.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Feature importance
vip = calculate_vip(pls, X_train, y_train)
important_features = vip > 1.0
```

---

## API Design Principles

### Consistent Interfaces

All FoodSpec functions follow scikit-learn conventions:

- **fit()**: Train on data
- **transform()**: Apply transformation
- **fit_transform()**: Train and transform
- **predict()**: Make predictions
- **score()**: Evaluate performance

### Type Hints

All public APIs have type hints for IDE support:

```python
def load_folder(
    path: PathLike,
    modality: str = 'auto',
    metadata: Optional[pd.DataFrame] = None
) -> FoodSpectrumSet:
    ...
```

### Provenance Tracking

All operations are logged for reproducibility:

```python
fs_clean = pipeline.transform(fs)
print(fs_clean.provenance)  # Complete processing history
```

---

## Module Cross-Reference Map

```plaintext
Core (FoodSpectrumSet)
  ├─→ IO (load/save)
  ├─→ Preprocessing (clean)
  ├─→ Features (extract)
  ├─→ Chemometrics (model)
  └─→ ML (train/validate)

Preprocessing
  ├─→ BaselineStep
  ├─→ SmoothingStep
  └─→ NormalizationStep

Features
  ├─→ Peaks (detect_peaks)
  ├─→ Bands (integrate_bands)
  ├─→ Ratios (compute_ratios)
  └─→ RQ Engine (RatioQualityEngine)

Chemometrics
  ├─→ PCA (run_pca)
  ├─→ Classification (make_pls_da, make_simca)
  ├─→ Regression (make_pls_regression)
  └─→ Mixture (mcr_als)
```

---

## API Conventions

### Function Naming

- `load_*()` - Load data from file/database
- `save_*()` - Save data to file
- `make_*()` - Create model/object
- `run_*()` - Execute complete workflow
- `compute_*()` - Calculate metrics/features
- `detect_*()` - Find patterns/peaks

### Return Types

- **Single object**: Direct return
- **Multiple outputs**: Named tuple or dict
- **Provenance**: Automatic via `FoodSpectrumSet`

---

## Getting Help

### In-Code Documentation

```python
# View function signature and docstring
help(load_folder)

# Or in Jupyter/IPython
?load_folder
??load_folder  # View source
```

### External Resources

- **Tutorials**: [Getting Started Guide](../getting-started/quickstart_15min.md)
- **Workflows**: [Workflow Documentation](../workflows/index.md)
- **Theory**: [Background Theory](../theory/index.md)
- **Examples**: `examples/` directory in repository

---

## Module Details

Click a module below for detailed API documentation:

- [**Core**](core.md) - Data structures (`FoodSpectrumSet`, `OutputBundle`)
- [**IO**](io.md) - Loading and saving (`load_folder()`, `to_hdf5()`)
- [**Preprocessing**](preprocessing.md) - Spectral preprocessing (`PreprocessPipeline`)
- [**Features**](features.md) - Feature extraction (`detect_peaks()`, `RatioQualityEngine`)
- [**Chemometrics**](chemometrics.md) - Chemometric models (`PCA`, `PLS-DA`, `MCR-ALS`)
- [**ML**](ml.md) - Machine learning (`nested_cross_validate()`)
- [**Stats**](stats.md) - Statistical analysis (`run_ttest()`, `run_anova()`)
- [**Metrics**](metrics.md) - Evaluation metrics (`accuracy`, `R²`, `RMSE`)
 
---

## Keywords

- IO
- preprocessing
- features
- machine learning
- statistics
- workflows
- [**Workflows**](workflows.md) - High-level workflows (oil auth, QC)
- [**Datasets**](datasets.md) - Example datasets (`load_olive_oils()`)
