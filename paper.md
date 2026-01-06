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

FoodSpec is an open-source Python toolkit for vibrational spectroscopy analysis in food science. It addresses reproducibility and standardization gaps by providing unified data structures, validated preprocessing methods, and production-ready machine learning pipelines for Raman, FTIR, NIR, and hyperspectral imaging data. The software integrates 30+ preprocessing methods, 10+ machine learning algorithms, automated data governance checks, and domain workflows for oil authentication, thermal degradation, and mixture analysis. FoodSpec is built on modern Python packaging with 79% test coverage across 689 tests, ensuring reliability for research and industrial applications.

## Statement of Need

Vibrational spectroscopy (Raman, FTIR, NIR) enables rapid, non-destructive assessment of food composition, authenticity, and quality. However, spectroscopic workflows face critical barriers to reproducibility and standardization.

### Fragmentation and Vendor Lock-in

Commercial instruments (Bruker, Thermo Fisher, Perkin Elmer) generate data in proprietary formats (OPUS, SPC, binary files), requiring manual conversion and vendor-specific software [@larkin2011]. This prevents data sharing between laboratories and limits method transferability. Existing open-source tools address narrow aspects: ChemoSpec [@chemospec] provides multivariate analysis (PCA, PLS) in R but lacks machine learning and hyperspectral support; HyperSpy [@hyperspy] excels at HSI preprocessing but lacks chemometric workflows; commercial alternatives (ProspectuR, ENVI) remain closed-source and expensive.

### Reproducibility Crisis in Chemometrics

Published food spectroscopy studies exhibit methodological issues compromising reproducibility. Common errors include preprocessing before train-test splitting (data leakage), splitting replicates across training and validation sets (inflating accuracy), and lack of standardized validation [@leite2013; @varoquaux2017]. Classification models reporting >95% accuracy on olive oil authentication often fail to generalize to external datasets or different instruments [@danezis2016]. No tools automatically detect and prevent such errors.

### The Need for Integrated, Domain-Aware Tooling

Food science requires domain-specific workflows encoding expert knowledge. Authenticating extra virgin olive oil requires specific Raman band ratios (1440/1660 cm⁻¹, 1655/1440 cm⁻¹) [@cepeda2019], thermal degradation needs trajectory analysis over heating time [@galtier2007], and hyperspectral imaging demands per-pixel classification with spatial context [@gowen2007]. Existing tools do not provide these application-ready workflows.

### FoodSpec's Solution

FoodSpec provides the first comprehensive, open-source toolkit combining vendor-agnostic import, validated preprocessing, production machine learning, and domain workflows. It implements six baseline correction methods (ALS, rubberband, polynomial, airPLS, modified polynomial, rolling ball) [@eilers2005], four normalization approaches (vector, SNV, MSC, area), and automated data governance detecting preprocessing-before-split leakage, replicate contamination, and batch effects. Machine learning includes nested cross-validation, calibration diagnostics (Brier score, reliability diagrams), and hyperparameter optimization for 10+ algorithms. Domain workflows for oil authentication, heating degradation, and mixture analysis encapsulate peer-reviewed protocols.

FoodSpec prioritizes reproducibility: protocol-driven execution via YAML ensures all parameters are version-controlled; automated artifact versioning tracks transformations, models, and figures; narrative report generation produces publication-ready markdown. The software is validated through 689 tests covering preprocessing correctness, statistical accuracy, and end-to-end workflows, with continuous integration on Python 3.10–3.12.

## Key Features and Implementation

FoodSpec's architecture separates data I/O, preprocessing, feature extraction, statistical analysis, and domain applications. The core data model provides `FoodSpectrumSet` for 1D spectra (Raman, FTIR, NIR), `HyperSpectralCube` for 3D spatial-spectral data, and `MultiModalDataset` for fused measurements. All structures support lazy HDF5 loading for large datasets.

The preprocessing module implements validated methods: asymmetric least squares baseline correction [@eilers2005], rubberband (convex hull), polynomial fitting, airPLS, and rolling ball algorithms; normalization (vector, SNV, MSC, area); Savitzky-Golay smoothing; and derivatives. All operations are immutable, creating new spectral objects for reproducibility. Feature extraction includes automated peak detection with chemical lookups, band ratio calculations, spectral fingerprinting, PCA/PLS projections, and Variable Importance in Projection (VIP) scores.

Machine learning integrates scikit-learn-compatible classifiers and regressors with nested cross-validation for unbiased performance estimation. The validation framework prevents errors: preprocessing occurs within cross-validation folds, replicate groups remain together, and batch effects are monitored. Calibration diagnostics (Brier score, expected calibration error, reliability diagrams) assess prediction uncertainty. Statistical testing includes parametric and non-parametric methods with multiple testing correction and effect size reporting.

Pre-configured workflows implement peer-reviewed protocols. Oil authentication combines Raman band ratios with PLS-DA classification [@muik2004; @cepeda2019]. Heating degradation tracks oxidation markers over time [@galtier2007]. Mixture analysis via NNLS and MCR-ALS quantifies component concentrations [@tauler1995]. Each workflow generates standardized reports with confusion matrices, ROC curves, and feature importance plots.

FoodSpec employs modern Python packaging (PEP 517/518, pyproject.toml, type hints throughout). The test suite comprises 689 tests achieving 79% coverage. Continuous integration via GitHub Actions runs tests on Python 3.10–3.12 for every commit. Documentation uses MkDocs with 150+ pages of guides, API references, and examples.

## Usage Example

Oil authentication from Raman spectra:

```python
from foodspec import load_library, FoodSpec

library = load_library("oils_raman.h5")
fs = FoodSpec(library)

result = fs.oil_authentication(
    label_column="oil_type",
    test_size=0.2,
    cv_folds=10,
    preprocessing=["baseline_als", "normalize_snv", "smooth_savgol"]
)

print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}")
report_path = fs.generate_report(output_dir="results/")
```

Command-line batch processing:

```bash
foodspec csv-to-library raw_spectra.csv library.h5 \
  --wavenumber-col wavenumber --sample-id-col sample_id

foodspec oil-auth library.h5 --label oil_type \
  --preprocessing als,snv,savgol --output results/
```

Protocol-driven execution via YAML:

```yaml
protocol: oil_authentication
data: {library: oils_raman.h5, label_column: oil_type}
preprocessing:
  - {method: baseline_als, lambda: 1e5}
  - {method: normalize_snv}
  - {method: smooth_savgol, window_length: 11}
model: {algorithm: random_forest, hyperparameters: {n_estimators: 100}}
validation: {cv_strategy: stratified_kfold, n_folds: 10, nested: true}
output: {figures: [confusion_matrix, roc_curve, feature_importance]}
```

## Comparison to Related Software

FoodSpec differs from existing tools in scope and integration. ChemoSpec [@chemospec] provides PCA and clustering but lacks machine learning and data governance. HyperSpy [@hyperspy] excels at HSI preprocessing but omits classification pipelines and validation frameworks. Orange Data Mining offers visual workflows but is not spectroscopy-specialized. Commercial software (ProspectuR, ENVI) remains closed-source, limiting reproducibility. FoodSpec is the only open-source toolkit combining validated preprocessing, production machine learning, automated data governance, and food-specific workflows suitable for regulatory submissions.

## Acknowledgments

The authors thank Dr. Jhinuk Gupta, Dr. Sai Muthukumar V, Ms. Amrita Shaw (Department of Food and Nutritional Sciences and Department of Physics, Sri Sathya Sai Institute of Higher Learning, India), and Deepak L. N. Kallepalli (Cognievolve AI Inc., Canada) for scientific guidance, laboratory support, and collaborative development. This work benefited from infrastructure support at Aix-Marseille Université. We acknowledge the open-source Python scientific computing community, particularly the scikit-learn, NumPy, SciPy, XGBoost, and LightGBM development teams, whose libraries form the foundation of FoodSpec's implementation.

## References
