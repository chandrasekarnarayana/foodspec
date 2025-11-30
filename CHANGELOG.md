# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased
### Added
- Core data model: `FoodSpectrumSet` for aligned spectra, wavenumbers, and metadata.
- Preprocessing transformers: ALS, rubberband, polynomial baselines; Savitzky-Golay and moving average smoothing; vector/area/internal-peak normalization; cropping utilities.
- FTIR/Raman helpers: atmospheric/ATR corrections with wavenumber-aware mixin; Raman cosmic-ray remover.
- Feature extractors: peak detection and expected-peak features, band integration, ratio generation, fingerprint similarities.
- Chemometrics utilities: PCA with result container, PLS/PLS-DA pipelines, classifier factory (logreg/SVM/RF/XGB/LGBM/KNN), validation metrics, cross-validation helpers, permutation test wrapper.
- Food applications: edible oil authentication workflow with preprocessing + peak/ratio features, CV metrics, confusion matrix, and optional feature importances.
- Heating analysis: ratios vs time with regression trends and optional ANOVA; plotting helper for ratio vs time.
- QC/novelty detection: OneClassSVM/IsolationForest training and scoring, QC tutorial.
- IO utilities: folder loaders with metadata merge, tidy CSV/HDF5 exporters, library create/load/search, synthetic example loader.
- CLI: preprocess spectra to HDF5 and run oil-auth with HTML report output.
- Visualization: spectra plots, PCA scores/loadings, confusion matrices, ROC, and HTML report generator.
- Docs: validation pages, QC tutorial, FTIR/Raman preprocessing, citing page, API reference expanded.
- Docs scaffold: MkDocs with Material + mkdocstrings; getting started, tutorial, API placeholder; design overview.
- Benchmark and CI: Jupytext-friendly benchmark script, GitHub Actions for lint/format/test/docs, and end-to-end CLI test.
