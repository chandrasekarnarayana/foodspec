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
  - name: Deepak L. N. Kallepalli
    affiliation: 3
  - name: Jhinuk Gupta
    affiliation: 2
  - name: Sai Muthukumar V
    affiliation: 2
  - name: Amrita Shaw
    affiliation: 2
  
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

FoodSpec is a Python toolkit I built to stop juggling vendor software, ad hoc scripts, and half-documented notebooks for Raman, FTIR, and NIR data in food science. It keeps import, preprocessing, feature extraction, model validation, and reporting in one place, with provenance logged by default. The library includes validated preprocessing (baseline, normalization, smoothing), leakage-aware validation, and domain workflows for oil authentication, heating degradation, mixture analysis, and hyperspectral use cases. Modern packaging, type hints, and 689 tests (79% coverage) back the code so it can be reused in labs and QC settings without guesswork.

## Statement of Need

In my lab work on food spectroscopy, three recurring problems slowed every project: fragmented toolchains, quiet reproducibility failures, and missing domain defaults.

**1. Fragmented toolchains.** Raman and FTIR instruments write OPUS/SPC/binary files. Baseline correction happened in one vendor app, normalization in another script, modeling in a notebook, and figures somewhere else. Sharing or rerunning the same analysis was painful because steps lived in different places. Existing open-source tools cover pieces (ChemoSpec for multivariate analysis, HyperSpy for hyperspectral preprocessing) but do not offer an end-to-end, food-focused path.

**2. Reproducibility and leakage.** Many published studies preprocess before splitting, split replicates across folds, or report single-split accuracy without uncertainty [@leite2013; @varoquaux2017]. I saw models that claimed >95% accuracy on olive oil but failed on external data [@danezis2016]. We needed guardrails that make the right thing the default: preprocessing inside CV folds, batch-aware splits, and clear artifacts and logs.

**3. Domain-aware workflows.** Food matrices need specific choices: oil authentication relies on band ratios around 1650/1440 cm⁻¹ [@cepeda2019]; heating degradation needs trajectories over time [@galtier2007]; hyperspectral cubes need per-pixel pipelines [@gowen2007]. Generic toolkits do not ship these patterns, so every lab rebuilds them.

**FoodSpec is the response.** It keeps import → preprocessing → features → modeling → validation → reporting in one toolkit, with defaults tuned for food science. Baseline methods (ALS, rubberband, polynomial, airPLS, rolling ball) [@eilers2005], normalization (vector, SNV, MSC, area), leakage detection, nested CV, and domain workflows (oil authentication, heating degradation, mixture analysis) are packaged with provenance logging. YAML protocols, run artifacts, and methods text make reruns and review straightforward.

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

The authors thank Dr. Jhinuk Gupta, Dr. Sai Muthukumar V, Ms. Amrita Shaw (Department of Food and Nutritional Sciences and Department of Physics, Sri Sathya Sai Institute of Higher Learning, India), and Deepak L. N. Kallepalli (Cognievolve AI Inc., Canada) for scientific guidance, laboratory support, and collaborative development. We acknowledge the open-source Python scientific computing community, particularly the scikit-learn, NumPy, SciPy, XGBoost, and LightGBM development teams, whose libraries form the foundation of FoodSpec's implementation.

## References
