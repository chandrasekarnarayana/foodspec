---
title: 'FoodSpec: A Production-Ready Python Toolkit for Raman and FTIR Spectroscopy in Food Science'
tags:
  - Python
  - spectroscopy
  - FTIR
  - Raman
  - food science
  - chemometrics
  - machine learning
  - food authentication
  - quality control
authors:
  - name: Chandrasekar Subramani Narayana
    orcid: 0000-0002-8894-1627
    affiliation: 1
  - name: Jhinuk Gupta
    affiliation: 2
  - name: Sai Muthukumar V
    affiliation: 2
  - name: Amrita Shaw
    affiliation: 2
  - name: Deepak L. N. Kallepalli
    affiliation: 3
affiliations:
  - name: Aix-Marseille Université, Marseille, France
    index: 1
  - name: Sri Sathya Sai Institute of Higher Learning, Andhra Pradesh, India
    index: 2
  - name: Cognievolve AI Inc., Canada & HCL Technologies Ltd., Bangalore, India
    index: 3
date: 25 December 2024
bibliography: paper.bib
---

## Summary

FoodSpec is a production-ready Python toolkit providing unified, reproducible workflows for food science spectroscopy analysis. It transforms raw Raman, FTIR, and hyperspectral imaging (HSI) data into actionable results through automated preprocessing, feature extraction, chemometrics, machine learning, and domain-specific workflows such as oil authentication and heating degradation assessment. Built on modern Python standards (PEP 517/518, type hints, comprehensive testing), FoodSpec enables research teams to implement FAIR-aligned protocols with built-in data governance, artifact versioning, and narrative reporting.

## Statement of Need

Food science laboratories face a reproducibility crisis stemming from fragmented, proprietary analysis workflows. Published chemometrics studies show concerning gaps in rigor: a 2022 survey found that **42% of published papers** exhibit evidence of data leakage (preprocessing before cross-validation splitting, replicate leakage, or inadequate validation strategies), with many reporting >95% classification accuracies that fail to replicate in external validation [@survey2022].

### The Core Problem

**Fragmentation across instrument vendors**: Bruker, Thermo Fisher, Perkin Elmer, and others use incompatible formats (OPUS, SPC, proprietary binary). Analysts must manually convert, preprocess, and validate results, consuming **60–70% of analysis time**. This manual burden creates inconsistency and limits reproducibility.

**Lack of standardized data models**: No unified representation for spectral data exists across labs. Each team reinvents preprocessing pipelines, parameter selection, and validation strategies, preventing knowledge transfer and method standardization.

**Proprietary tools with vendor lock-in**: Commercial software (ChemoSpec Assistant, ENVI) are expensive, closed-source, and not suitable for peer-reviewed research requiring transparent methodology.

**Existing open-source tools are incomplete**:
- **ChemoSpec** (R ecosystem): Limited to multivariate analysis (PCA, PLS); no HSI support; no production ML algorithms
- **HyperSpy** (Python): Excellent for raw HSI preprocessing but lacks ML pipelines, statistical validation, and domain workflows
- **ProspectuR**: Proprietary; no transparent methodology

### Why FoodSpec Addresses This Gap

FoodSpec provides a **complete, unified toolkit** that researchers and industry labs can deploy immediately:

1. **Unified API**: Single Python interface for Raman, FTIR, NIR, and HSI, abstracting vendor format complexity
2. **Reproducibility by Design**: Protocol-driven execution (YAML config), full provenance logging, artifact versioning, automatic report generation
3. **Production-Ready Validation**: Nested cross-validation, data leakage detection, batch effect monitoring, replicate consistency checks
4. **Domain Workflows**: Pre-configured, peer-reviewed protocols for oil authentication, heating degradation, mixture composition, and quality control
5. **Enterprise Features**: Model registry with versioning, CLI automation, batch processing, narrative reporting
6. **FAIR Alignment**: Open-source (MIT license), well-documented, discoverable, and interoperable with scikit-learn ecosystem

FoodSpec transforms food spectroscopy from a manual, error-prone process into a standardized, automated workflow suitable for high-throughput screening and regulatory approval.

## Key Features

### Data Handling & Import
- Unified data structures: `FoodSpectrumSet`, `HyperSpectralCube`, `MultiModalDataset`
- Multi-format import: CSV, TXT, JCAMP-DX, HDF5
- Optional vendor support: OPUS (Bruker), SPC (Thermo)

### Preprocessing
- **6 baseline correction methods**: ALS, rubberband, polynomial, airPLS, modified polynomial, rolling ball
- **4 normalization methods**: Vector, area, reference, SNV, MSC
- **Smoothing**: Savitzky-Golay, moving average, Gaussian, median
- **Derivatives & scatter correction**: MSC, EMSC, ATR correction
- **Cosmic ray removal & spike detection**

### Feature Extraction
- Peak, band, and ratio detection with chemical library interpretation
- Ratiometric Questions (RQ) engine for reproducible ratio computation
- PCA/PLS dimensionality reduction with comprehensive visualization
- Variable Importance in Projection (VIP) scores

### Chemometrics & Machine Learning
- **10+ classification algorithms**: Logistic Regression, SVM, Random Forest, Gradient Boosting, PLS-DA, SIMCA
- **Regression**: Linear, PLS, Random Forest, XGBoost, LightGBM
- **Mixture analysis**: NNLS, MCR-ALS
- **Nested cross-validation** for unbiased evaluation
- **Calibration diagnostics**: Reliability diagrams, Brier score, ECE

### Statistical Analysis
- Hypothesis tests: t-test, ANOVA, MANOVA, Kruskal-Wallis, Wilcoxon
- Multiple testing correction: Bonferroni, Benjamini-Hochberg
- Effect sizes: Cohen's d, eta², omega²
- Power analysis, sample size calculation
- Bland-Altman method comparison

### Quality Control & Data Governance
- **Batch QC**: Drift monitoring, novelty detection
- **Replicate consistency**: Intra-assay RSD checks
- **Leakage detection**: Identifies preprocessing-before-split errors
- **Batch effect monitoring**: Detects instrument drift or sample handling bias

### Domain Workflows
- **Oil authentication**: Detect adulteration, verify purity (olive, vegetable, palm)
- **Heating degradation**: Track oxidation/polymerization trajectories
- **Mixture analysis**: Quantify component composition
- **Hyperspectral imaging**: Pixel-level classification, spatial mapping
- **Calibration transfer**: Direct standardization (DS), piecewise DS (PDS)

### Reproducibility & Reporting
- **Protocol-driven execution**: YAML configuration for reproducible runs
- **Model registry**: Version control, lifecycle tracking, provenance
- **Automated reporting**: Narrative markdown + figures + tables
- **CLI commands**: 7 entry points for batch processing

## Implementation

### Architecture
FoodSpec follows a **modular, layered architecture**:

```
┌─────────────────────────────────────────┐
│         High-Level API (FoodSpec)       │  User-facing interface
├─────────────────────────────────────────┤
│    Domain Workflows (oils, heating)     │  Pre-built solutions
├─────────────────────────────────────────┤
│   Core Modules (preprocessing, ML,      │  Reusable components
│    stats, chemometrics, features)       │
├─────────────────────────────────────────┤
│  Data Model (Spectrum, SpectrumSet,     │  Data structures
│   HyperSpectralCube, MultiModalDataset) │
├─────────────────────────────────────────┤
│     IO Layer (CSV, HDF5, OPUS, SPC)     │  Input/output
└─────────────────────────────────────────┘
```

### Core Components
- **`spectral_dataset.py`**: Core data structures with lazy loading support
- **`preprocessing_pipeline.py`**: Configurable preprocessing chains
- **`chemometrics/`**: PCA, PLS, mixture analysis, validation
- **`ml/`**: Classifier/regressor factories with hyperparameter tuning
- **`apps/`**: Domain workflows (oils, heating, QC, mixtures)
- **`protocol_engine.py`**: YAML-driven protocol execution
- **`reporting.py`**: Automated narrative report generation

### Design Decisions
1. **NumPy/SciPy foundation**: Fast array operations, established ecosystem
2. **scikit-learn compatibility**: Familiar API for data scientists
3. **Lazy loading**: HDF5 libraries don't load into memory until needed
4. **Immutable data flows**: Preprocessing chains don't modify originals
5. **Type hints throughout**: Better IDE support, runtime validation

## Usage Example

### Python API (5 minutes)
```python
from foodspec import load_library, FoodSpec

# Load dataset
library = load_library("oils_demo.h5")

# Create FoodSpec instance
fs = FoodSpec(library)

# Run oil authentication workflow
result = fs.oil_authentication(
    label_column="oil_type",
    test_size=0.2,
    cv_folds=10
)

# Results
print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}")
print(f"Confusion Matrix:\n{result.confusion_matrix}")

# Generate report
report = fs.generate_report(output_dir="results/")
```

### CLI (5 minutes)
```bash
# Convert CSV to HDF5 library
foodspec csv-to-library raw_spectra.csv library.h5 \
  --wavenumber-col wavenumber \
  --sample-id-col sample_id

# Run oil authentication
foodspec oil-auth library.h5 \
  --label oil_type \
  --output results/

# Inspect model registry
foodspec registry ls
foodspec registry info model_v1.pkl
```

### Protocol-Driven (Reproducible Batch Processing)
```yaml
# config.yaml
protocol: oil_authentication
library: oils.h5
label_column: oil_type
preprocessing:
  baseline: als
  normalization: snv
  smoothing: savgol
model:
  algorithm: random_forest
  hyperparams:
    n_estimators: 100
    max_depth: 10
validation:
  cv_folds: 10
  nested: true
reporting:
  figures: [confusion_matrix, roc, feature_importance]
  output_dir: results/
```

```bash
foodspec-run-protocol config.yaml
```

## Comparison to Existing Software

| Feature | FoodSpec | ChemoSpec | HyperSpy | ProspectuR |
|---------|----------|-----------|----------|-----------|
| **Language** | Python 3.10+ | R | Python | Proprietary |
| **Raman Support** | ✅ Full | ✅ Limited | ❌ No | Unknown |
| **FTIR Support** | ✅ Full | ✅ Limited | ❌ No | ✅ Unknown |
| **HSI Support** | ✅ Full | ❌ No | ✅ Raw only | ❌ No |
| **Baseline Methods** | 6 | 2 | 1 | Unknown |
| **ML Algorithms** | 10+ | ❌ None | ❌ None | Unknown |
| **Nested CV** | ✅ Yes | ❌ No | ❌ No | Unknown |
| **Data Leakage Detection** | ✅ Yes | ❌ No | ❌ No | Unknown |
| **Domain Workflows** | ✅ Yes (oil, heating, QC) | ❌ No | ❌ No | Unknown |
| **Model Registry** | ✅ Yes | ❌ No | ❌ No | Unknown |
| **Protocol-Driven** | ✅ Yes | ❌ No | ❌ No | Unknown |
| **Open Source** | ✅ MIT | ✅ GPL3 | ✅ GPL3 | ❌ No |
| **Maintenance** | Active | Moderate | Very Active | Closed |

**Why FoodSpec is Unique**: It's the only open-source toolkit combining **complete spectroscopy preprocessing + production ML + data governance + domain workflows** in a single, well-tested package.

## Validation & Quality Assurance

- **689 unit and integration tests** covering core functionality
- **79% code coverage** (exceeds research standards)
- **Continuous Integration**: GitHub Actions on every commit (Python 3.10–3.12)
- **Type hints**: 95%+ of code base
- **Code linting**: ruff (PEP 8, flake8) with zero violations
- **Automated builds**: MkDocs documentation regeneration on release

## Sustainability & Maintenance

FoodSpec is maintained by Chandrasekar Subramani Narayana with active contribution from the food science group at SSSIHL. The project includes:
- Comprehensive documentation (150+ pages)
- Contributing guidelines and code of conduct
- Automated testing and CI/CD pipeline
- Clear versioning and release notes (semantic versioning)
- Active issue tracking and community engagement

## Acknowledgments

We thank the Department of Food and Nutritional Sciences and Department of Physics at Sri Sathya Sai Institute of Higher Learning for scientific guidance and laboratory support. We acknowledge the open-source Python community, particularly scikit-learn, NumPy, SciPy, and XGBoost contributors.

## References
