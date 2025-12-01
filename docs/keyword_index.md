# Keyword index & glossary

You are here: Reference & index → Keyword index & glossary

Questions this page answers
- Where do I find a concept (preprocessing method, test, model, metric, workflow, CLI command)?
- Which docs and API pages explain it?

## Spectral preprocessing
- **ALSBaseline (ALS baseline correction)** — removes fluorescence/sloping background. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.baseline.ALSBaseline`.
- **RubberbandBaseline** — convex-hull baseline for concave backgrounds. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.baseline.RubberbandBaseline`.
- **PolynomialBaseline** — low-degree baseline fit. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.baseline.PolynomialBaseline`.
- **SavitzkyGolaySmoother (SavGol)** — noise reduction preserving peaks. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.smoothing.SavitzkyGolaySmoother`.
- **MovingAverageSmoother** — simple denoising; may broaden peaks. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.smoothing.MovingAverageSmoother`.
- **Vector/Area/Max normalization** — scales spectra to unit norm/area. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.normalization.VectorNormalizer`.
- **SNVNormalizer (Standard Normal Variate)** — mean/std per spectrum to reduce scatter. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.normalization.SNVNormalizer`.
- **MSCNormalizer (Multiplicative Scatter Correction)** — corrects additive/multiplicative scatter via reference. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.normalization.MSCNormalizer`.
- **InternalPeakNormalizer** — normalize to a stable internal band/window. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.normalization.InternalPeakNormalizer`.
- **DerivativeTransformer (derivatives)** — Savitzky–Golay derivatives (1st/2nd). See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.derivatives.DerivativeTransformer`.
- **AtmosphericCorrector (FTIR)** — remove water/CO₂ contributions. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.ftir.AtmosphericCorrector`.
- **SimpleATRCorrector (FTIR)** — heuristic ATR depth correction. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.ftir.SimpleATRCorrector`.
- **CosmicRayRemover (Raman)** — remove spike artifacts. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.raman.CosmicRayRemover`.
- **RangeCropper** — crop to target wavenumber window. See: `ftir_raman_preprocessing.md`, `api_reference.md#foodspec.preprocess.cropping.RangeCropper`.

## Features and ratios
- **PeakFeatureExtractor / detect_peaks** — peak heights/areas near expected bands. See: `oil_auth_tutorial.md`, `api_reference.md#foodspec.features.peaks.PeakFeatureExtractor`.
- **integrate_bands** — integrate intensity over defined bands. See: `mixture_tutorial.md`, `api_reference.md#foodspec.features.bands.integrate_bands`.
- **RatioFeatureGenerator / compute_ratios** — compute band/peak ratios (e.g., 1655/1742). See: `oil_auth_tutorial.md`, `api_reference.md#foodspec.features.ratios.RatioFeatureGenerator`.
- **Fingerprint similarity (cosine/correlation)** — spectral similarity matrices. See: `hyperspectral_tutorial.md`, `api_reference.md#foodspec.features.fingerprint`.

## Statistical tests
- **t-tests (independent/paired/one-sample)** — compare means. See: `stats_tests.md`.
- **ANOVA / MANOVA** — multi-group mean differences. See: `stats_tests.md`.
- **Mann–Whitney U / Kruskal–Wallis / Wilcoxon / Friedman** — non-parametric comparisons. See: `stats_tests.md`.
- **Correlation (Pearson/Spearman) / simple regression** — associations and trends. See: `stats_tests.md`.

## Machine learning models
- **Logistic regression / Linear SVM / PLS-DA** — linear classifiers. See: `ml_models.md`, `api_reference.md#foodspec.chemometrics.models`.
- **RBF SVM / k-NN / Random Forest (RF) / Gradient Boosting** — nonlinear classifiers. See: `ml_models.md`, `api_reference.md#foodspec.chemometrics.models`.
- **PCA / clustering** — unsupervised exploration/visualization. See: `chemometrics_guide.md`, `api_reference.md#foodspec.chemometrics.pca`.
- **Conv1DSpectrumClassifier (1D CNN)** — optional deep model. See: `ml_models.md`, `api_reference.md#foodspec.chemometrics.deep.Conv1DSpectrumClassifier`.
- **Mixture models (NNLS, MCR-ALS)** — estimate component fractions. See: `mixture_tutorial.md`, `api_reference.md#foodspec.chemometrics.mixture`.

## Metrics and validation
- **Accuracy, Precision, Recall, F1 (macro/micro), ROC-AUC, Confusion matrix** — classification metrics. See: `metrics_interpretation.md`, `api_reference.md#foodspec.chemometrics.validation`.
- **R², RMSE, MAE, Residuals** — regression/mixture metrics. See: `metrics_interpretation.md`, `api_reference.md#foodspec.chemometrics.validation`.
- **Cross-validation (CV)** — k-fold, stratified CV for models. See: `metrics_interpretation.md`, `api_reference.md#foodspec.chemometrics.validation`.

## Workflows
- **Oil authentication** — classify oils/adulteration. See: `oil_auth_tutorial.md`, `methodsx_protocol.md`, `api_reference.md#foodspec.apps.oils`.
- **Heating degradation** — ratios vs time/temperature. See: `heating_tutorial.md`, `api_reference.md#foodspec.apps.heating`.
- **Mixture analysis (NNLS/MCR-ALS)** — estimate fractions. See: `mixture_tutorial.md`, `api_reference.md#foodspec.chemometrics.mixture` and `api_reference.md#foodspec.apps.methodsx_reproduction`.
- **QC / Novelty detection** — one-class scoring. See: `qc_tutorial.md`, `api_reference.md#foodspec.apps.qc`.
- **Hyperspectral analysis** — ratio/cluster maps. See: `hyperspectral_tutorial.md`, `api_reference.md#foodspec.core.hyperspectral.HyperSpectralCube`.
- **Protocol benchmarks / MethodsX reproduction** — standardized evaluation. See: `protocol_benchmarks.md`, `methodsx_protocol.md`, `api_reference.md#foodspec.apps.protocol_validation`, `api_reference.md#foodspec.apps.methodsx_reproduction`.
- **Domain templates (meat/microbial)** — adapt oil workflow to other domains. See: `domains_overview.md`, `meat_tutorial.md`, `microbial_tutorial.md`, `api_reference.md#foodspec.apps.meat`, `api_reference.md#foodspec.apps.microbial`.

## CLI commands
- **about** — version/info. See: `cli.md`.
- **csv-to-library / preprocess** — build libraries, preprocess raw data. See: `cli.md`.
- **oil-auth / heating / qc / domains** — workflow commands. See: `cli.md`.
- **mixture / hyperspectral** — mixture and hyperspectral utilities. See: `cli.md`.
- **protocol-benchmarks / reproduce-methodsx** — protocol runs. See: `cli.md`.
- **model-info** — inspect saved model metadata. See: `cli.md`.

See also
- `metrics_interpretation.md`
- `oil_auth_tutorial.md`
- `methodsx_protocol.md`
- `api_reference.md`
