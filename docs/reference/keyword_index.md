# Keyword index & glossary

You are here: Reference & index → Keyword index & glossary

Questions this page answers
- Where do I find a concept (preprocessing method, test, model, metric, workflow, CLI command)?
- Which docs and API pages explain it?

## Spectral preprocessing
- **ALSBaseline (ALS baseline correction)** — removes fluorescence/sloping background. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.baseline.ALSBaseline`.
- **RubberbandBaseline** — convex-hull baseline for concave backgrounds. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.baseline.RubberbandBaseline`.
- **PolynomialBaseline** — low-degree baseline fit. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.baseline.PolynomialBaseline`.
- **SavitzkyGolaySmoother (SavGol)** — noise reduction preserving peaks. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.smoothing.SavitzkyGolaySmoother`.
- **MovingAverageSmoother** — simple denoising; may broaden peaks. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.smoothing.MovingAverageSmoother`.
- **Vector/Area/Max normalization** — scales spectra to unit norm/area. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.normalization.VectorNormalizer`.
- **SNVNormalizer (Standard Normal Variate)** — mean/std per spectrum to reduce scatter. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.normalization.SNVNormalizer`.
- **MSCNormalizer (Multiplicative Scatter Correction)** — corrects additive/multiplicative scatter via reference. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.normalization.MSCNormalizer`.
- **InternalPeakNormalizer** — normalize to a stable internal band/window. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.normalization.InternalPeakNormalizer`.
- **DerivativeTransformer (derivatives)** — Savitzky–Golay derivatives (1st/2nd). See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.derivatives.DerivativeTransformer`.
- **AtmosphericCorrector (FTIR)** — remove water/CO₂ contributions. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.ftir.AtmosphericCorrector`.
- **SimpleATRCorrector (FTIR)** — heuristic ATR depth correction. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.ftir.SimpleATRCorrector`.
- **CosmicRayRemover (Raman)** — remove spike artifacts. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.raman.CosmicRayRemover`.
- **RangeCropper** — crop to target wavenumber window. See: `ftir_raman_preprocessing.md`, `api/preprocessing.md#foodspec.preprocess.cropping.RangeCropper`.

## Features and ratios
- **PeakFeatureExtractor / detect_peaks** — peak heights/areas near expected bands. See: `workflows/oil_authentication.md`, `api/features.md#foodspec.features.peaks.PeakFeatureExtractor`.
- **integrate_bands** — integrate intensity over defined bands. See: `workflows/mixture_analysis.md`, `api/features.md#foodspec.features.bands.integrate_bands`.
- **RatioFeatureGenerator / compute_ratios** — compute band/peak ratios (e.g., 1655/1742). See: `workflows/oil_authentication.md`, `api/features.md#foodspec.features.ratios.RatioFeatureGenerator`.
- **Fingerprint similarity (cosine/correlation)** — spectral similarity matrices. See: `hyperspectral_tutorial.md`, `api/features.md#foodspec.features.fingerprint`.

## Statistical tests
- **t-tests (independent/paired/one-sample)** — compare means. See: `stats_tests.md`.
- **ANOVA / MANOVA** — multi-group mean differences. See: `stats_tests.md`.
- **Mann–Whitney U / Kruskal–Wallis / Wilcoxon / Friedman** — non-parametric comparisons. See: `stats_tests.md`.
- **Correlation (Pearson/Spearman) / simple regression** — associations and trends. See: `stats_tests.md`.

## Machine learning models
- **Logistic regression / Linear SVM / PLS-DA** — linear classifiers. See: `ml/models_and_best_practices.md`, `api/chemometrics.md#foodspec.chemometrics.models`.
- **RBF SVM / k-NN / Random Forest (RF) / Gradient Boosting** — nonlinear classifiers. See: `ml/models_and_best_practices.md`, `api/chemometrics.md#foodspec.chemometrics.models`.
- **PCA / clustering** — unsupervised exploration/visualization. See: `chemometrics_guide.md`, `api/chemometrics.md#foodspec.chemometrics.pca`.
- **Conv1DSpectrumClassifier (1D CNN)** — optional deep model. See: `advanced_deep_learning.md`, `api/chemometrics.md#foodspec.chemometrics.deep.Conv1DSpectrumClassifier`.
- **Mixture models (NNLS, MCR-ALS)** — estimate component fractions. See: `workflows/mixture_analysis.md`, `api/chemometrics.md#foodspec.chemometrics.mixture`.

## Metrics and validation
- **Accuracy, Precision, Recall, F1 (macro/micro), ROC-AUC, Confusion matrix** — classification metrics. See: `metrics/metrics_and_evaluation/`, `api/metrics.md`.
- **R², RMSE, MAE, Residuals** — regression/mixture metrics. See: `metrics/metrics_and_evaluation/`, `api/metrics.md`.
- **Cross-validation (CV)** — k-fold, stratified CV for models. See: `metrics/metrics_and_evaluation/`, `api/chemometrics.md#foodspec.chemometrics.validation`.

## Workflows
- **Oil authentication** — classify oils/adulteration. See: `workflows/oil_authentication.md`, `api/workflows.md#foodspec.apps.oils`.
- **Heating degradation** — ratios vs time/temperature. See: `workflows/heating_quality_monitoring.md`, `api/workflows.md#foodspec.apps.heating`.
- **Mixture analysis (NNLS/MCR-ALS)** — estimate fractions. See: `workflows/mixture_analysis.md`, `api/chemometrics.md#foodspec.chemometrics.mixture`.
- **QC / Novelty detection** — one-class scoring. See: `qc_tutorial.md`, `api/workflows.md#foodspec.apps.qc`.
- **Hyperspectral analysis** — ratio/cluster maps. See: `hyperspectral_tutorial.md`, `api/datasets.md#foodspec.core.hyperspectral.HyperSpectralCube`.
- **Protocol benchmarks** — standardized evaluation. See: `protocol_benchmarks.md`, `api/workflows.md#foodspec.apps.protocol_validation`.
- **Domain templates (meat/microbial)** — adapt oil workflow to other domains. See: `domains_overview.md`, `meat_tutorial.md`, `microbial_tutorial.md`, `api/workflows.md#foodspec.apps.meat`, `api/workflows.md#foodspec.apps.microbial`.

## CLI commands
- **about** — version/info. See: `cli.md`.
- **csv-to-library / preprocess** — build libraries, preprocess raw data. See: `cli.md`.
- **oil-auth / heating / qc / domains** — workflow commands. See: `cli.md`.
- **mixture / hyperspectral** — mixture and hyperspectral utilities. See: `cli.md`.
- **protocol-benchmarks** — protocol runs. See: `cli.md`.
- **model-info** — inspect saved model metadata. See: `cli.md`.

See also
- [Metrics & evaluation](../reference/metrics_reference.md)
- [workflows/oil_authentication.md](../workflows/authentication/oil_authentication.md)

- [API index](../api/index.md)
