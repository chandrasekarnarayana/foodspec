# FoodSpec Documentation

**Version:** 1.0.0 | **License:** MIT | **Python:** 3.10+

---

## What is FoodSpec?

FoodSpec is a Python toolkit for reproducible vibrational spectroscopy analysis in food science. It integrates import (vendor formats), preprocessing (6 baseline methods, 4 normalizations), feature extraction (peaks, ratios, fingerprinting), chemometrics (PCA, PLS-DA, MCR-ALS), machine learning (classifier/regressor factory, nested CV), and domain-specific workflows (oil authentication, heating degradation, batch QC) in a single library with full provenance logging.

---

## Who This Is For

- **Food scientists** conducting authenticity or quality assurance studies
- **Chemometricians** implementing and validating multivariate methods
- **Spectroscopists** standardizing preprocessing and baseline correction workflows
- **Data scientists and ML engineers** building reproducible pipelines for regulatory submissions
- **Researchers and auditors** verifying methodological soundness and absence of data leakage

---

## Problems FoodSpec Solves

1. **Fragmented toolchains**: Stop exporting OPUS → CSV → baseline tool → normalization script → modeling notebook. FoodSpec keeps import, preprocessing, features, modeling, and validation in one library with consistent interfaces.

2. **Reproducibility failures and leakage**: FoodSpec enforces best practices by default—preprocessing occurs inside cross-validation folds, replicate/batch groups remain intact, and full run artifacts (parameters, versions, logs, figures) are saved together so analyses can be rerun identically.

3. **Missing domain defaults**: Oil authentication relies on specific band ratios; heating studies need time-aware validation; hyperspectral cubes need per-pixel pipelines. FoodSpec packages these patterns as ready workflows with sensible food-science defaults.

---

## When NOT to Use FoodSpec

- **Certified regulatory methods** that mandate specific instrument vendor software (e.g., ISO 12966 for milk fat analysis using proprietary apps).
- **Exploratory, one-off analysis** where custom scripts or notebooks are faster than learning a new library.
- **Real-time embedded systems** or edge computing (FoodSpec targets desktop/server-grade Python environments).
- **Non-vibrational modalities** (e.g., chromatography, immunoassays); use domain-specific tools instead.

---

## Three Learning Paths

### Path 1: 15-Minute Quickstart (New Users)

Start here if you want to run your first example immediately without deep dives.

→ [15-Minute Quickstart](getting-started/quickstart_15min.md)

- Install FoodSpec in 3 commands
- Load a sample oil spectra dataset
- Run oil authentication workflow in 5 lines of code
- View results (confusion matrix, ROC curve, feature importance)

Estimated time: 15 minutes. No prior FoodSpec knowledge required.

### Path 2: Applied Workflow Tutorial (Practitioners)

Hands-on guide to building a real-world analysis from data import to validated results.

→ [Oil Authentication Workflow](tutorials/intermediate/01-oil-authentication.md)

- Load and visualize spectral data (OPUS, CSV, HDF5)
- Choose and apply preprocessing (baseline, normalization, smoothing)
- Extract features (peak detection, chemical ratios)
- Train and validate a classifier with nested CV and leakage checks
- Generate publication-ready figures and metrics reports

Estimated time: 1–2 hours. Assumes basic Python and chemistry knowledge.

### Path 3: API Reference (Developers)

Complete function and class documentation for extending FoodSpec or integrating into custom pipelines.

→ [API Reference](api/index.md)

- [Core data structures](api/core.md) (SpectralDataset, HyperSpectralCube, MultiModalDataset)
- [Preprocessing methods](api/preprocessing.md) (baselines, normalization, smoothing)
- [Chemometrics and ML](api/chemometrics.md) (PCA, PLS-DA, classifiers, regressors)
- [Feature extraction](api/features.md) (peak detection, ratios, fingerprinting)
- [Validation and metrics](api/metrics.md) (nested CV, balanced accuracy, calibration)
- [Workflows and protocols](api/workflows.md) (oil auth, heating, QC)

Estimated time: Reference as needed. Requires Python proficiency.

---

## Core Capabilities Summary

| Capability | Examples |
|------------|----------|
| **Import** | OPUS (Bruker), SPA (Thermo), DPT (PerkinElmer), CSV, HDF5 |
| **Preprocessing** | ALS, rubberband, polynomial, airPLS baselines; SNV, MSC, vector normalization; Savitzky-Golay smoothing; ATR and cosmic-ray corrections |
| **Features** | Automated peak detection; band ratios with chemical interpretation; PCA/PLS projections; Variable Importance in Projection (VIP) scores |
| **Validation** | Stratified and batch-aware cross-validation; nested CV for unbiased performance; leakage detection; calibration diagnostics; uncertainty estimates |
| **Workflows** | Oil authentication (Raman/FTIR), heating degradation tracking, mixture analysis (NNLS, MCR-ALS), batch quality control, hyperspectral mapping |
| **Reproducibility** | YAML protocol definitions, run artifacts (figures, models, methods text), provenance logging (versions, timestamps, parameters), automated report generation |

---

## Next Steps

- **Installation issues?** → [Installation Guide](getting-started/installation.md)
- **Choose preprocessing methods?** → [Preprocessing Methods](methods/preprocessing/baseline_correction.md)
- **Understand validation?** → [Validation & Reproducibility](methods/validation/cross_validation_and_leakage.md)
- **Build custom workflows?** → [Protocol Design](user-guide/protocols_and_yaml.md)
- **Trouble running your data?** → [Troubleshooting](help/troubleshooting.md)
- **Questions on scope?** → [Scope & Limitations](non_goals_and_limitations.md)

---

## Citation

If you use FoodSpec in published research, please cite:

> Subramani Narayana, C. (2024). FoodSpec: A Python toolkit for Raman and FTIR spectroscopy in food science (v1.0.0). https://github.com/chandrasekarnarayana/foodspec

Full BibTeX and citation guidance: [Citing FoodSpec](reference/citing.md)

---

## Problems Addressed

FoodSpec solves four critical challenges in food spectroscopy research:

### 1. Workflow Fragmentation
**Problem:** Researchers rely on vendor-specific software for data import, ad hoc scripts for preprocessing, and manual figure generation for publication.  
**Solution:** Unified Python API with consistent data structures (SpectralDataset, HyperspectralCube), automated preprocessing pipelines, and publication-ready reporting.

### 2. Reproducibility Barriers
**Problem:** Preprocessing parameters, model hyperparameters, and validation strategies are often undocumented or scattered across notebooks and scripts.  
**Solution:** YAML protocol execution with full provenance tracking, timestamped run artifacts, and methods text generation for transparent documentation.

### 3. Domain-Specific Gaps
**Problem:** General spectroscopy tools lack food-matrix optimizations (oil-specific baseline methods, ATR correction), chemical interpretability (peak assignments, ratio significance), and domain workflows.  
**Solution:** Pre-validated workflows for oil authentication, heating degradation, and mixture analysis; ratiometric quality engine with chemical annotations; food-relevant preprocessing defaults.

### 4. Validation Complexity
**Problem:** Cross-validation leakage, batch effects, and small sample sizes lead to overoptimistic performance estimates.  
**Solution:** Batch-aware splitting, nested cross-validation, permutation tests, and confidence-bounded metrics with uncertainty quantification.

---

## Differences from Alternative Tools

| Feature | **FoodSpec** | ChemoSpec (R) | HyperSpy (Python) | Vendor Software |
|---------|--------------|---------------|-------------------|-----------------|
| **Food-matrix workflows** | ✅ Native (oils, mixtures, heating) | ❌ Generic chemometrics | ❌ Materials/microscopy focus | ⚠️ Limited to instrument ecosystem |
| **End-to-end pipelines** | ✅ Import → preprocess → analyze → report | ⚠️ Requires manual assembly | ⚠️ Preprocessing-focused | ⚠️ Limited export options |
| **Interpretability** | ✅ Peak ratios, VIP scores, chemical labels | ⚠️ Basic PCA/HCA scores | ❌ Limited feature naming | ❌ Black-box results |
| **Reproducibility** | ✅ YAML protocols, provenance tracking | ⚠️ Script-based | ⚠️ Script-based | ❌ Manual parameter recording |
| **CLI availability** | ✅ Full CLI + batch processing | ❌ R console only | ❌ Python API only | ⚠️ GUI-only or proprietary formats |
| **Open source** | ✅ MIT license | ✅ GPL-2 | ✅ GPL-3 | ❌ Proprietary |
| **Target domain** | Food authentication & QC | General chemistry | Electron microscopy, XRD | Instrument-specific |

**Key differentiators:** FoodSpec uniquely combines domain-specific workflows, chemical interpretability, and reproducibility infrastructure in a single, production-ready toolkit designed for food science research.

---

## Quick Start

### Installation

```bash
pip install foodspec

# Optional: Machine learning extensions
pip install 'foodspec[ml]'  # XGBoost, LightGBM

# Optional: Deep learning
pip install 'foodspec[deep]'  # Conv1D, MLP models
```

**Requirements:** Python 3.10+, NumPy, pandas, scikit-learn, SciPy  
**Documentation:** [Installation Guide](getting-started/installation.md)

### Minimal Example (Python API)

```python
from foodspec import SpectralDataset
from foodspec.workflows.oils import run_oil_authentication_workflow

# Load spectra from CSV
ds = SpectralDataset.from_csv(
    "oils.csv", 
    wavenumber_col="wavenumber",
    label_col="oil_type"
)

# Run complete authentication workflow
result = run_oil_authentication_workflow(ds, label_column="oil_type")

# Access results
print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}")
print(result.confusion_matrix)
print(result.feature_importance.head())
```

**Next steps:** [15-Minute Quickstart](getting-started/quickstart_15min.md)

### Minimal Example (CLI)

```bash
# Convert raw CSV to HDF5 library
foodspec csv-to-library oils.csv library.h5 \
  --wavenumber-col wavenumber \
  --sample-id-col sample_id

# Run oil authentication workflow
foodspec oil-auth library.h5 \
  --label oil_type \
  --output results/
```

**Next steps:** [CLI Quickstart](getting-started/quickstart_cli.md)

---

## Core Capabilities

### Data Import and Management
- **Formats:** CSV, HDF5, JCAMP-DX; vendor support for Bruker OPUS, Thermo SPA, Agilent DPT
- **Structures:** SpectralDataset (1D spectra), HyperspectralCube (spatial imaging)
- **Metadata:** Sample IDs, labels, batch information, instrument parameters

### Preprocessing
- **Baseline correction:** ALS, rubberband, polynomial, airPLS, rolling ball, modified polynomial
- **Normalization:** Vector, SNV, MSC, min-max, reference peak
- **Smoothing:** Savitzky-Golay, moving average, Gaussian
- **Corrections:** ATR correction (FTIR), cosmic ray removal (Raman), atmospheric compensation

### Feature Extraction
- **Peak analysis:** Automatic peak detection, integration, width/position tracking
- **Ratiometric features:** Peak ratios with chemical interpretation (e.g., C=O/C-H for oxidation)
- **Dimensionality reduction:** PCA, PLS, t-SNE for exploratory analysis
- **Chemical libraries:** Pre-defined peak assignments for oils, carbohydrates, proteins

### Chemometrics and Machine Learning
- **Unsupervised:** PCA, hierarchical clustering, k-means with silhouette validation
- **Supervised classification:** Logistic regression, SVM, Random Forest, Gradient Boosting, PLS-DA
- **Supervised regression:** Linear regression, PLS, Ridge, LASSO
- **Mixture analysis:** NNLS, MCR-ALS with constraints (non-negativity, unimodality)
- **Model management:** Versioning, hyperparameter tracking, artifact storage

### Validation and Metrics
- **Cross-validation:** Stratified k-fold, batch-aware splitting, nested CV for hyperparameter tuning
- **Metrics:** Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, Cohen's kappa
- **Uncertainty:** Bootstrap confidence intervals, permutation tests, calibration curves
- **Leakage detection:** Automatic checks for preprocessing-before-split, batch confounding

### Domain Workflows
- **Oil authentication:** Classify edible oils, detect adulteration, assess purity
- **Heating degradation:** Track oxidation markers, estimate shelf life, monitor frying stability
- **Batch quality control:** Screen incoming ingredients, detect production anomalies
- **Mixture quantification:** Estimate blend composition, decompose overlapping spectra
- **Hyperspectral imaging:** Spatial mapping, region segmentation, compositional heterogeneity

### Reproducibility and Reporting
- **Protocol execution:** YAML configuration files specify full analysis pipeline
- **Run artifacts:** Timestamped bundles with preprocessed data, model weights, metrics, figures
- **Automated reporting:** Methods text generation, figure exports (PNG/PDF), metrics tables
- **Provenance tracking:** Git commit hashes, package versions, execution timestamps

---

## Scope and Limitations

### Appropriate Use Cases
- Rapid screening and decision support in food authentication
- Comparative analysis of batches, treatments, or production runs
- Method development and validation studies
- Educational applications in food science and chemometrics

### Known Limitations
- **Not a certified method:** Regulatory compliance (ISO, FDA, AOAC) requires additional validation
- **Detection limits:** Trace contaminants (<1% w/w) may be below spectroscopic resolution
- **Matrix effects:** Performance depends on sample preparation and instrument calibration
- **Inference scope:** Models trained on specific oils/conditions may not generalize to novel matrices

**Full documentation:** [Non-Goals and Limitations](non_goals_and_limitations.md)

---

## Documentation Navigation

This documentation is organized into 12 sections, designed to support users at different expertise levels and research stages.

### For New Users

| Section | Purpose | Start Here |
|---------|---------|------------|
| **[Getting Started](getting-started/index.md)** | Installation, quickstarts (15-min, Python, CLI), basic FAQ | [Installation Guide](getting-started/installation.md) |
| **[Tutorials](tutorials/index.md)** | Step-by-step guided examples (beginner → intermediate → advanced) | [Beginner: Load and Plot](tutorials/beginner/01-load-and-plot.md) |
| **[Theory](theory/spectroscopy_basics.md)** | Spectroscopy fundamentals, food applications, chemometrics basics | [Spectroscopy Basics](theory/spectroscopy_basics.md) |

### For Practitioners

| Section | Purpose | Navigate By |
|---------|---------|-------------|
| **[Workflows](workflows/index.md)** | Domain-specific pipelines (oil authentication, heating, QC, mixtures) | Use case (authentication, quality monitoring, quantification) |
| **[Methods](methods/preprocessing/baseline_correction.md)** | Preprocessing, chemometrics, statistics, validation strategies | Method type (baseline, PCA, ANOVA, cross-validation) |
| **[User Guide](user-guide/index.md)** | CLI reference, data formats, protocols, automation, visualization | Feature (CLI, HDF5, YAML, plotting) |

### For Developers and Researchers

| Section | Purpose | Navigate By |
|---------|---------|-------------|
| **[API Reference](api/index.md)** | Python module documentation with function signatures and examples | Module (core, preprocessing, chemometrics, features) |
| **[User Guide - Advanced](user-guide/model_registry.md)** | Model registry, lifecycle, and deployment | Feature (models, registry, lifecycle) |
| **[Developer Guide](developer-guide/index.md)** | Contributing, testing, documentation style, release process | Activity (contributing, testing, documenting) |

### For Reference

| Section | Purpose | Navigate By |
|---------|---------|-------------|
| **[Reference](reference/index.md)** | Metrics tables, glossary, method comparison, changelog, citations | Lookup (metric definitions, terminology, version history) |
| **[Help](help/index.md)** | Troubleshooting, FAQ, issue reporting guidelines | Problem (installation errors, preprocessing issues, model failures) |

---

## Documentation Conventions

### Code Examples
All Python examples assume the following imports unless otherwise specified:

```python
from foodspec import SpectralDataset, HyperspectralCube
from foodspec.workflows import oils, heating, qc
import numpy as np
import pandas as pd
```

### File Paths
- **Relative paths** (e.g., `data/oils.csv`) assume execution from project root
- **Absolute paths** are shown when context matters
- **HDF5 libraries** use `.h5` extension by convention

### Terminology
- **Spectrum (pl. spectra):** Single intensity vs. wavenumber/wavelength measurement
- **Dataset:** Collection of spectra with metadata (labels, batches, timestamps)
- **Workflow:** End-to-end pipeline from raw data to validated results
- **Protocol:** YAML configuration specifying preprocessing, modeling, and validation steps

Full definitions: [Glossary](reference/glossary.md)

---

## Getting Help

### Documentation Resources
- **Installation issues:** [Troubleshooting Guide](help/troubleshooting.md)
- **Common questions:** [FAQ](help/faq.md)
- **Method selection:** [Method Comparison Table](reference/method_comparison.md)
- **Metric interpretation:** [Metrics Reference](reference/metrics_reference.md)

### Community Support
- **Discussion forum:** [GitHub Discussions](https://github.com/chandrasekarnarayana/foodspec/discussions) for questions, use cases, and feedback
- **Bug reports:** [GitHub Issues](https://github.com/chandrasekarnarayana/foodspec/issues) for reproducible errors or unexpected behavior
- **Feature requests:** Open an issue with `[Feature Request]` prefix

### Reporting Issues
When reporting bugs, include:
1. FoodSpec version: `python -c "import foodspec; print(foodspec.__version__)"`
2. Python version and operating system
3. Minimal reproducible example (MRE)
4. Expected vs. actual behavior

Guidelines: [Issue Reporting](help/troubleshooting.md)

---

## Citation

If you use FoodSpec in published research, please cite:

```bibtex
@software{foodspec2024,
  author = {Subramani Narayana, Chandrasekar},
  title = {{FoodSpec}: A Python toolkit for Raman and FTIR spectroscopy in food science},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/chandrasekarnarayana/foodspec},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

**Plain text:**  
Subramani Narayana, C. (2024). FoodSpec: A Python toolkit for Raman and FTIR spectroscopy in food science (v1.0.0). https://github.com/chandrasekarnarayana/foodspec

**Additional citations:** [Full Citation Guide](reference/citing.md)

---

## Acknowledgments

FoodSpec is developed collaboratively by:

- **Chandrasekar Subramani Narayana** (Aix-Marseille Université, France) — Lead developer
- **Jhinuk Gupta** (Sri Sathya Sai Institute of Higher Learning, India) — Chemometrics validation
- **Sai Muthukumar V** (Sri Sathya Sai Institute of Higher Learning, India) — Domain workflows
- **Amrita Shaw** (Sri Sathya Sai Institute of Higher Learning, India) — Statistical methods
- **Deepak L. N. Kallepalli** (Cognievolve AI Inc., HCL Technologies) — Software architecture

**Funding and institutional support:** This work was conducted as part of doctoral research at Aix-Marseille Université with collaboration from Sri Sathya Sai Institute of Higher Learning.

---

## License and Redistribution

FoodSpec is released under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute the software for academic, commercial, or personal purposes, provided that:

1. The original copyright notice is retained
2. The software is provided "as-is" without warranty
3. Modifications are clearly indicated if redistributed

**Full license text:** [LICENSE](https://github.com/chandrasekarnarayana/foodspec/blob/main/LICENSE)

---

**Last updated:** January 2026 | **Documentation version:** 1.0.0
