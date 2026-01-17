# FoodSpec Methods Inventory

**Generated:** January 6, 2026  
**Purpose:** Complete inventory of methods supported by FoodSpec with pointers to documentation and API  
**Scope:** All methods across preprocessing, chemometrics, validation, statistics, and quality control

---

## Preprocessing Methods (37 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **Asymmetric Least Squares (ALS)** | Preprocessing/Baseline | [baseline_correction.md](../methods/preprocessing/baseline_correction.md) | `ALSBaseline`, `baseline_als()` | Remove fluorescence backgrounds, moderate-strong curvature |
| **Rubberband (Convex Hull)** | Preprocessing/Baseline | [baseline_correction.md](../methods/preprocessing/baseline_correction.md) | `RubberbandBaseline`, `baseline_rubberband()` | Quick baseline for well-separated peaks |
| **Polynomial Baseline** | Preprocessing/Baseline | [baseline_correction.md](../methods/preprocessing/baseline_correction.md) | `PolynomialBaseline`, `baseline_polynomial()` | Mild baseline curvature, simple backgrounds |
| **SNIP Baseline** | Preprocessing/Baseline | Internal | `_baseline_snip()` (engine) | Sensitive nonlinear iterative peeling |
| **Vector Normalization** | Preprocessing/Normalization | [normalization_smoothing.md](../methods/preprocessing/normalization_smoothing.md) | `VectorNormalizer` | Remove intensity scaling, inter-sample comparison |
| **Area Normalization** | Preprocessing/Normalization | [normalization_smoothing.md](../methods/preprocessing/normalization_smoothing.md) | `AreaNormalizer` | Normalize to total spectral area |
| **Peak Normalization (Internal)** | Preprocessing/Normalization | [normalization_smoothing.md](../methods/preprocessing/normalization_smoothing.md) | `InternalPeakNormalizer` | Normalize to internal standard peak |
| **Standard Normal Variate (SNV)** | Preprocessing/Normalization | [normalization_smoothing.md](../methods/preprocessing/normalization_smoothing.md) | `SNVNormalizer` | Remove scatter effects, particle size variations |
| **Multiplicative Scatter Correction (MSC)** | Preprocessing/Normalization | [scatter_correction_cosmic_ray_removal.md](../methods/preprocessing/scatter_correction_cosmic_ray_removal.md) | `MSCNormalizer` | Correct ATR contact variations, path length differences |
| **Savitzky-Golay Smoothing** | Preprocessing/Smoothing | [normalization_smoothing.md](../methods/preprocessing/normalization_smoothing.md) | `SavitzkyGolaySmoother` | Reduce noise while preserving peak shapes |
| **Moving Average Smoothing** | Preprocessing/Smoothing | [normalization_smoothing.md](../methods/preprocessing/normalization_smoothing.md) | `MovingAverageSmoother` | Simple noise reduction |
| **Gaussian Smoothing** | Preprocessing/Smoothing | Internal | `SmoothingStep(method="gaussian")` | Gentle noise reduction |
| **Median Smoothing** | Preprocessing/Smoothing | Internal | `SmoothingStep(method="median")` | Spike-resistant smoothing |
| **1st Derivative** | Preprocessing/Feature Enhancement | [derivatives_and_feature_enhancement.md](../methods/preprocessing/derivatives_and_feature_enhancement.md) | `DerivativeStep(order=1)` | Resolve overlapping peaks, baseline suppression |
| **2nd Derivative** | Preprocessing/Feature Enhancement | [derivatives_and_feature_enhancement.md](../methods/preprocessing/derivatives_and_feature_enhancement.md) | `DerivativeStep(order=2)` | Maximum peak resolution, fine structure |
| **Cosmic Ray Removal** | Preprocessing/Artifact Removal | [scatter_correction_cosmic_ray_removal.md](../methods/preprocessing/scatter_correction_cosmic_ray_removal.md) | `CosmicRayRemover`, `correct_cosmic_rays()` | Remove CCD spike artifacts in Raman spectra |
| **Atmospheric Correction (FTIR)** | Preprocessing/Artifact Removal | Internal | `AtmosphericCorrector` | Remove H₂O/CO₂ atmospheric interference |
| **ATR Correction** | Preprocessing/Artifact Removal | Internal | `SimpleATRCorrector` | Correct wavelength-dependent penetration depth |
| **Spike Detection & Removal** | Preprocessing/Artifact Removal | Internal | `correct_cosmic_rays()` | Identify and interpolate isolated spikes |
| **Spectral Cropping** | Preprocessing/Preprocessing | Internal | `RangeCropper` | Select wavenumber ranges of interest |
| **Spectral Alignment** | Preprocessing/Preprocessing | Internal | `AlignmentStep` | Correct peak position shifts via cross-correlation |
| **Spectral Resampling** | Preprocessing/Preprocessing | Internal | `ResampleStep` | Interpolate to common wavenumber grid |
| **Direct Standardization (DS)** | Preprocessing/Harmonization | [harmonization_automated_calibration.md](../workflows/harmonization_automated_calibration.md) | `direct_standardization()` | Transfer calibration between instruments |
| **Piecewise Direct Standardization (PDS)** | Preprocessing/Harmonization | [harmonization_automated_calibration.md](../workflows/harmonization_automated_calibration.md) | `piecewise_direct_standardization()` | Local calibration transfer with sliding windows |
| **Preprocessing Pipeline** | Preprocessing/Workflow | [methods/preprocessing/](../methods/preprocessing/) | `PreprocessPipeline` | Chain multiple preprocessing steps |
| **Auto-Preprocessing** | Preprocessing/Workflow | Internal | `AutoPreprocess` | Automatic preprocessing optimization |
| **Peak Detection** | Preprocessing/Feature Extraction | [feature_extraction.md](../methods/preprocessing/feature_extraction.md) | `detect_peaks()` | Find peaks in spectra (scipy.signal) |
| **Band Integration** | Preprocessing/Feature Extraction | [feature_extraction.md](../methods/preprocessing/feature_extraction.md) | `integrate_bands()` | Compute peak areas |
| **Peak Statistics** | Preprocessing/Feature Extraction | [feature_extraction.md](../methods/preprocessing/feature_extraction.md) | `compute_band_features()` | Height, width, area, position |
| **Ratio Quality (RQ) Engine** | Preprocessing/Feature Extraction | [feature_extraction.md](../methods/preprocessing/feature_extraction.md), [rq_engine_detailed.md](../theory/rq_engine_detailed.md) | `RatioQualityEngine` | Compute peak ratios with quality scores |
| **Cosine Similarity** | Preprocessing/Similarity | Internal | `cosine_similarity_matrix()` | Spectral fingerprinting |
| **Correlation Similarity** | Preprocessing/Similarity | Internal | `correlation_similarity_matrix()` | Spectral similarity via correlation |
| **Library Search** | Preprocessing/Similarity | [library_search.md](../user-guide/library_search.md) | `similarity_search()` | Match spectrum to reference library |
| **Drift Detection** | Preprocessing/QC | [harmonization_automated_calibration.md](../workflows/harmonization_automated_calibration.md) | `detect_drift()` | Monitor instrument drift over time |
| **Incremental Calibration Adaptation** | Preprocessing/Harmonization | Internal | `adapt_calibration_incremental()` | Update calibration with new batches |
| **Baseline Metrics** | Preprocessing/Diagnostics | Internal | `baseline_metrics()` | Assess baseline correction quality |
| **Smoothing Metrics** | Preprocessing/Diagnostics | Internal | `smoothing_metrics()` | Quantify noise reduction |

---

## Chemometrics Methods (25 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **Principal Component Analysis (PCA)** | Chemometrics/Dimensionality Reduction | [pca_and_dimensionality_reduction.md](../methods/chemometrics/pca_and_dimensionality_reduction.md) | `run_pca()` | Exploratory analysis, dimensionality reduction |
| **Partial Least Squares Discriminant Analysis (PLS-DA)** | Chemometrics/Classification | [classification_regression.md](../methods/chemometrics/classification_regression.md) | `make_pls_da()` | Supervised classification with correlated features |
| **Partial Least Squares Regression (PLS-R)** | Chemometrics/Regression | [classification_regression.md](../methods/chemometrics/classification_regression.md) | `make_pls_regression()` | Calibration, property prediction |
| **Support Vector Machine (SVM)** | Chemometrics/Classification | [classification_regression.md](../methods/chemometrics/classification_regression.md) | `make_classifier("svm_rbf")` | Nonlinear classification |
| **Random Forest** | Chemometrics/Classification | [classification_regression.md](../methods/chemometrics/classification_regression.md) | `make_classifier("random_forest")` | Robust classification with feature importance |
| **Gradient Boosting (XGBoost)** | Chemometrics/Classification | [classification_regression.md](../methods/chemometrics/classification_regression.md) | `make_classifier("xgboost")` | Handle imbalanced data, strong performance |
| **k-Nearest Neighbors (kNN)** | Chemometrics/Classification | [classification_regression.md](../methods/chemometrics/classification_regression.md) | `make_classifier("knn")` | Simple baseline classifier |
| **Logistic Regression** | Chemometrics/Classification | [classification_regression.md](../methods/chemometrics/classification_regression.md) | `make_classifier("logistic")` | Interpretable linear classification |
| **Multi-Layer Perceptron (MLP)** | Chemometrics/Deep Learning | [advanced_deep_learning.md](../methods/chemometrics/advanced_deep_learning.md) | `make_mlp_regressor()` | Nonlinear regression with neural networks |
| **Conv1D Classifier** | Chemometrics/Deep Learning | [advanced_deep_learning.md](../methods/chemometrics/advanced_deep_learning.md) | `Conv1DSpectrumClassifier` | 1D convolutional neural network for spectra |
| **Support Vector Regression (SVR)** | Chemometrics/Regression | [classification_regression.md](../methods/chemometrics/classification_regression.md) | `make_regressor("svr")` | Nonlinear regression |
| **Non-Negative Least Squares (NNLS)** | Chemometrics/Mixture Analysis | [mixture_models.md](../methods/chemometrics/mixture_models.md) | `nnls_mixture()` | Quantify mixture components |
| **Multivariate Curve Resolution - ALS (MCR-ALS)** | Chemometrics/Mixture Analysis | [mixture_models.md](../methods/chemometrics/mixture_models.md) | `mcr_als()` | Resolve pure component spectra from mixtures |
| **One-Class SVM (OCSVM)** | Chemometrics/Novelty Detection | [models_and_best_practices.md](../methods/chemometrics/models_and_best_practices.md) | `make_one_class_scanner("ocsvm")` | Detect outliers/novelty |
| **Isolation Forest** | Chemometrics/Novelty Detection | [models_and_best_practices.md](../methods/chemometrics/models_and_best_practices.md) | `make_one_class_scanner("isolation_forest")` | Anomaly detection |
| **Local Outlier Factor (LOF)** | Chemometrics/Novelty Detection | [models_and_best_practices.md](../methods/chemometrics/models_and_best_practices.md) | `make_one_class_scanner("lof")` | Local density-based outlier detection |
| **SIMCA (Soft Independent Modeling of Class Analogy)** | Chemometrics/Classification | [models_and_best_practices.md](../methods/chemometrics/models_and_best_practices.md) | `make_simca()` | Class modeling via PCA + Hotelling T² |
| **Variable Importance in Projection (VIP)** | Chemometrics/Interpretability | [model_interpretability.md](../methods/chemometrics/model_interpretability.md) | `calculate_vip()` | Feature importance for PLS models |
| **SHAP Values** | Chemometrics/Interpretability | [model_interpretability.md](../methods/chemometrics/model_interpretability.md) | `compute_shap_values()` | Model-agnostic feature attribution |
| **Permutation Importance** | Chemometrics/Interpretability | [model_interpretability.md](../methods/chemometrics/model_interpretability.md) | `permutation_importance_wrapper()` | Model-agnostic feature importance |
| **Grid Search Hyperparameter Tuning** | Chemometrics/Model Selection | Internal | `grid_search_classifier()` | Exhaustive hyperparameter search |
| **Quick Hyperparameter Tuning** | Chemometrics/Model Selection | Internal | `quick_tune_classifier()` | Fast hyperparameter optimization |
| **Late Fusion** | Chemometrics/Multi-Modal | [multimodal_workflows.md](../workflows/multimodal_workflows.md) | `late_fusion_concat()` | Concatenate features from multiple modalities |
| **Decision Fusion (Vote)** | Chemometrics/Multi-Modal | [multimodal_workflows.md](../workflows/multimodal_workflows.md) | `decision_fusion_vote()` | Combine predictions via majority vote |
| **Mixture Analysis Workflow** | Chemometrics/Workflow | [mixture_analysis.md](../workflows/mixture_analysis.md) | `run_mixture_analysis_workflow()` | End-to-end mixture quantification |

---

## Validation Methods (18 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **k-Fold Cross-Validation** | Validation/Cross-Validation | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | `cross_validate_pipeline()` | Standard model validation |
| **Stratified k-Fold CV** | Validation/Cross-Validation | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | sklearn `StratifiedKFold` | Maintain class distribution in folds |
| **Leave-One-Out CV (LOO)** | Validation/Cross-Validation | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | sklearn `LeaveOneOut` | Small datasets |
| **Nested Cross-Validation** | Validation/Cross-Validation | [advanced_validation_strategies.md](../methods/validation/advanced_validation_strategies.md) | `nested_cross_validate()` | Unbiased model selection + evaluation |
| **Permutation Test** | Validation/Statistical Testing | [robustness_checks.md](../methods/validation/robustness_checks.md) | `permutation_test_score_wrapper()` | Test if model beats chance |
| **Bootstrap Confidence Intervals** | Validation/Uncertainty | [robustness_checks.md](../methods/validation/robustness_checks.md) | `bootstrap_prediction_intervals()` | Estimate prediction uncertainty |
| **Conformal Prediction** | Validation/Uncertainty | [advanced_validation_strategies.md](../methods/validation/advanced_validation_strategies.md) | `split_conformal_regression()` | Calibrated prediction intervals |
| **Confusion Matrix** | Validation/Classification Metrics | [metrics_and_uncertainty.md](../methods/validation/metrics_and_uncertainty.md) | `confusion_matrix_table()` | Classification performance breakdown |
| **ROC Curve / AUC** | Validation/Classification Metrics | [metrics_and_uncertainty.md](../methods/validation/metrics_and_uncertainty.md) | `compute_roc_curve()` | Binary classification performance |
| **Precision-Recall Curve** | Validation/Classification Metrics | [metrics_and_uncertainty.md](../methods/validation/metrics_and_uncertainty.md) | `compute_pr_curve()` | Performance for imbalanced data |
| **Classification Metrics** | Validation/Classification Metrics | [metrics_and_uncertainty.md](../methods/validation/metrics_and_uncertainty.md) | `compute_classification_metrics()` | F1, precision, recall, accuracy |
| **Regression Metrics** | Validation/Regression Metrics | [metrics_and_uncertainty.md](../methods/validation/metrics_and_uncertainty.md) | `compute_regression_metrics()` | RMSE, MAE, R², adjusted R² |
| **Calibration Curve** | Validation/Calibration | [metrics_and_uncertainty.md](../methods/validation/metrics_and_uncertainty.md) | `calibration_summary()` | Assess probability calibration |
| **Reliability Diagram** | Validation/Calibration | [metrics_and_uncertainty.md](../methods/validation/metrics_and_uncertainty.md) | `reliability_diagram()` | Visualize calibration quality |
| **Bland-Altman Analysis** | Validation/Method Comparison | [metrics_and_uncertainty.md](../methods/validation/metrics_and_uncertainty.md) | `bland_altman()` | Compare two measurement methods |
| **Explained Variance** | Validation/Dimensionality Reduction | Internal | `compute_explained_variance()` | Assess PCA component importance |
| **Leakage Detection** | Validation/Data Quality | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | `detect_leakage()` | Detect train/test contamination |
| **VIP Table with Meanings** | Validation/Interpretability | [model_interpretability.md](../methods/chemometrics/model_interpretability.md) | `vip_table_with_meanings()` | Annotate VIP scores with chemical meanings |

---

## Statistical Methods (23 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **Independent t-test** | Statistics/Hypothesis Testing | [t_tests_effect_sizes_and_power.md](../methods/statistics/t_tests_effect_sizes_and_power.md) | `run_ttest(paired=False)` | Compare means of two independent groups |
| **Paired t-test** | Statistics/Hypothesis Testing | [t_tests_effect_sizes_and_power.md](../methods/statistics/t_tests_effect_sizes_and_power.md) | `run_ttest(paired=True)` | Compare paired observations |
| **One-Way ANOVA** | Statistics/Hypothesis Testing | [anova_and_manova.md](../methods/statistics/anova_and_manova.md) | `run_anova()` | Compare means of 3+ groups |
| **MANOVA** | Statistics/Hypothesis Testing | [anova_and_manova.md](../methods/statistics/anova_and_manova.md) | `run_manova()` | Multivariate ANOVA (multiple response variables) |
| **Tukey HSD Post-hoc** | Statistics/Hypothesis Testing | [anova_and_manova.md](../methods/statistics/anova_and_manova.md) | `run_tukey_hsd()` | Pairwise comparisons after ANOVA (equal variances) |
| **Games-Howell Post-hoc** | Statistics/Hypothesis Testing | [anova_and_manova.md](../methods/statistics/anova_and_manova.md) | `games_howell()` | Pairwise comparisons (unequal variances) |
| **Mann-Whitney U Test** | Statistics/Nonparametric | [nonparametric_methods_and_robustness.md](../methods/statistics/nonparametric_methods_and_robustness.md) | `run_mannwhitney_u()` | Nonparametric alternative to t-test |
| **Wilcoxon Signed-Rank Test** | Statistics/Nonparametric | [nonparametric_methods_and_robustness.md](../methods/statistics/nonparametric_methods_and_robustness.md) | `run_wilcoxon_signed_rank()` | Nonparametric paired test |
| **Kruskal-Wallis Test** | Statistics/Nonparametric | [nonparametric_methods_and_robustness.md](../methods/statistics/nonparametric_methods_and_robustness.md) | `run_kruskal_wallis()` | Nonparametric alternative to ANOVA |
| **Friedman Test** | Statistics/Nonparametric | [nonparametric_methods_and_robustness.md](../methods/statistics/nonparametric_methods_and_robustness.md) | `run_friedman_test()` | Nonparametric repeated measures |
| **Shapiro-Wilk Normality Test** | Statistics/Assumption Testing | [introduction_to_statistical_analysis.md](../methods/statistics/introduction_to_statistical_analysis.md) | `run_shapiro()` | Test for normality |
| **Cohen's d** | Statistics/Effect Sizes | [t_tests_effect_sizes_and_power.md](../methods/statistics/t_tests_effect_sizes_and_power.md) | `compute_cohens_d()` | Standardized mean difference |
| **ANOVA Effect Sizes (η², ω²)** | Statistics/Effect Sizes | [anova_and_manova.md](../methods/statistics/anova_and_manova.md) | `compute_anova_effect_sizes()` | Proportion of variance explained |
| **Benjamini-Hochberg Correction** | Statistics/Multiple Testing | [hypothesis_testing_in_food_spectroscopy.md](../methods/statistics/hypothesis_testing_in_food_spectroscopy.md) | `benjamini_hochberg()` | False discovery rate control |
| **Bonferroni Correction** | Statistics/Multiple Testing | Internal | Manual application | Family-wise error rate control |
| **Pearson Correlation** | Statistics/Correlation | [correlation_and_mapping.md](../methods/statistics/correlation_and_mapping.md) | `compute_correlations(method="pearson")` | Linear correlation |
| **Spearman Correlation** | Statistics/Correlation | [correlation_and_mapping.md](../methods/statistics/correlation_and_mapping.md) | `compute_correlations(method="spearman")` | Rank-based correlation |
| **Kendall Correlation** | Statistics/Correlation | [correlation_and_mapping.md](../methods/statistics/correlation_and_mapping.md) | `compute_correlations(method="kendall")` | Tau rank correlation |
| **Cross-Correlation** | Statistics/Time Series | Internal | `compute_cross_correlation()` | Lag-based correlation |
| **Bootstrap Metric** | Statistics/Robustness | [robustness_checks.md](../methods/validation/robustness_checks.md) | `bootstrap_metric()` | Bootstrap confidence intervals |
| **Permutation Test Metric** | Statistics/Robustness | [robustness_checks.md](../methods/validation/robustness_checks.md) | `permutation_test_metric()` | Permutation-based p-values |
| **Bland-Altman Method Comparison** | Statistics/Method Comparison | Internal | `bland_altman()`, `bland_altman_plot()` | Agreement between two methods |
| **Embedding Quality Metrics** | Statistics/Dimensionality Reduction | Internal | `evaluate_embedding()` | Silhouette, between/within cluster F-stat |

---

## Quality Control Methods (20 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **Spectrum Health Score** | QC/Spectral Quality | Internal (scaffolded) | `SpectrumHealthScore` | Assess SNR, spikes, saturation |
| **Compute Health Scores** | QC/Spectral Quality | [data_governance.md](../user-guide/data_governance.md) | `compute_health_scores()` | Batch spectral quality assessment |
| **Novelty Detection (Distance-based)** | QC/Outlier Detection | [data_governance.md](../user-guide/data_governance.md) | `novelty_scores()` | Detect novel/outlier spectra |
| **Outlier Detection (Engine)** | QC/Outlier Detection | [data_governance.md](../user-guide/data_governance.md) | `detect_outliers()` | Statistical outlier identification |
| **Drift Detection (PSI)** | QC/Monitoring | [data_governance.md](../user-guide/data_governance.md) | `population_stability_index()` | Population Stability Index for drift |
| **Drift Detection (KL Divergence)** | QC/Monitoring | Internal | `kl_divergence()` | Kullback-Leibler divergence for distribution shift |
| **Drift Detection (Engine)** | QC/Monitoring | [data_governance.md](../user-guide/data_governance.md) | `detect_drift()` | Multi-metric drift detection |
| **Feature Drift Detection** | QC/Monitoring | Internal | `detect_feature_drift()` | Per-feature drift monitoring |
| **Recalibration Decision** | QC/Monitoring | Internal | `should_recalibrate()` | Automated recalibration triggers |
| **Class Balance Check** | QC/Dataset Quality | [data_governance.md](../user-guide/data_governance.md) | `check_class_balance()` | Diagnose class imbalance |
| **Imbalance Diagnosis** | QC/Dataset Quality | Internal | `diagnose_imbalance()` | Recommend sampling strategies |
| **Replicate Consistency** | QC/Dataset Quality | [data_governance.md](../user-guide/data_governance.md) | `compute_replicate_consistency()` | Assess technical replicates |
| **Variability Source Assessment** | QC/Dataset Quality | Internal | `assess_variability_sources()` | Decompose variance sources |
| **Leakage Detection (Batch-Label)** | QC/Dataset Quality | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | `detect_batch_label_correlation()` | Detect batch confounding |
| **Leakage Detection (Replicate)** | QC/Dataset Quality | Internal | `detect_replicate_leakage()` | Detect replicate splitting issues |
| **Leakage Detection (Unified)** | QC/Dataset Quality | [cross_validation_and_leakage.md](../methods/validation/cross_validation_and_leakage.md) | `detect_leakage()` | Comprehensive leakage checks |
| **Readiness Score** | QC/Dataset Quality | [data_governance.md](../user-guide/data_governance.md) | `compute_readiness_score()` | Overall dataset readiness for modeling |
| **Prediction QC** | QC/Inference | Internal | `evaluate_prediction_qc()` | Assess prediction quality flags |
| **Threshold Optimization (Quantile)** | QC/Thresholding | Internal | `estimate_threshold_quantile()` | Set thresholds via quantiles |
| **Threshold Optimization (Youden)** | QC/Thresholding | Internal | `estimate_threshold_youden()` | Optimize threshold via Youden index |

---

## Distance & Similarity Metrics (6 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **Euclidean Distance** | Metrics/Distance | Internal | `euclidean_distance()` | L2 distance between spectra |
| **Cosine Distance** | Metrics/Distance | Internal | `cosine_distance()` | Angular distance (1 - cosine similarity) |
| **Pearson Distance** | Metrics/Distance | Internal | `pearson_distance()` | 1 - Pearson correlation |
| **Spectral Information Divergence (SID)** | Metrics/Distance | Internal | `sid_distance()` | Information-theoretic spectral distance |
| **Spectral Angle Mapper (SAM)** | Metrics/Distance | Internal | `sam_angle()` | Angle between spectral vectors |
| **Distance Matrix Computation** | Metrics/Distance | Internal | `compute_distances()` | Pairwise distance matrix |

---

## Time-Series & Trajectory Methods (4 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **Linear Slope Estimation** | Time-Series/Trajectory | [heating_quality_monitoring.md](../workflows/heating_quality_monitoring.md) | `linear_slope()` | Fit linear trend to time series |
| **Quadratic Acceleration** | Time-Series/Trajectory | [heating_quality_monitoring.md](../workflows/heating_quality_monitoring.md) | `quadratic_acceleration()` | Estimate curvature in degradation |
| **Rolling Slope** | Time-Series/Trajectory | Internal | `rolling_slope()` | Moving window slope estimation |
| **Heating Trajectory Analysis** | Time-Series/Workflow | [heating_quality_monitoring.md](../workflows/heating_quality_monitoring.md) | `analyze_heating_trajectory()` | Analyze thermal degradation over time |

---

## Multi-Modal & Fusion Methods (3 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **Modality Agreement (Kappa)** | Fusion/Metrics | [multimodal_workflows.md](../workflows/multimodal_workflows.md) | `modality_agreement_kappa()` | Cohen's kappa between modalities |
| **Modality Consistency Rate** | Fusion/Metrics | Internal | `modality_consistency_rate()` | Fraction of consistent predictions |
| **Cross-Modality Correlation** | Fusion/Metrics | Internal | `cross_modality_correlation()` | Feature correlation across modalities |

---

## Workflow & Reporting Methods (6 methods)

| Method Name | Family | Primary Doc Page | Primary API Object | Typical Use Case |
|-------------|--------|------------------|-------------------|------------------|
| **Dataset Summary** | Workflow/Reporting | [data_governance.md](../user-guide/data_governance.md) | `summarize_dataset()` | Generate dataset overview |
| **QC Report Generation** | Workflow/Reporting | [data_governance.md](../user-guide/data_governance.md) | `generate_qc_report()` | Unified QC report |
| **Statistics Report (Feature)** | Workflow/Reporting | Internal | `stats_report_for_feature()` | Per-feature statistical summary |
| **Statistics Report (Table)** | Workflow/Reporting | Internal | `stats_report_for_features_table()` | Multi-feature stats table |
| **Group Size Summary** | Workflow/Study Design | [study_design_and_data_requirements.md](../methods/statistics/study_design_and_data_requirements.md) | `summarize_group_sizes()` | Check sample sizes per group |
| **Minimum Sample Check** | Workflow/Study Design | Internal | `check_minimum_samples()` | Validate sufficient samples |

---

## Summary Statistics

| Category | Method Count |
|----------|-------------|
| **Preprocessing** | 37 methods |
| **Chemometrics** | 25 methods |
| **Validation** | 18 methods |
| **Statistics** | 23 methods |
| **Quality Control** | 20 methods |
| **Distance Metrics** | 6 methods |
| **Time-Series** | 4 methods |
| **Multi-Modal** | 3 methods |
| **Workflow/Reporting** | 6 methods |
| **TOTAL** | **142 methods** |

---

## Method Coverage by Workflow

| Workflow | Key Methods Used | Doc Reference |
|----------|------------------|---------------|
| **Oil Authentication** | PLS-DA, PCA, VIP, cross-validation, confusion matrix | [01_oil_authentication.md](../examples/01_oil_authentication.md) |
| **Heating Quality Monitoring** | RQ Engine, linear slope, trajectory analysis, peak ratios | [heating_quality_monitoring.md](../workflows/heating_quality_monitoring.md) |
| **Mixture Analysis** | NNLS, MCR-ALS, band integration, regression metrics | [mixture_analysis.md](../workflows/mixture_analysis.md) |
| **Hyperspectral Mapping** | PCA, k-means clustering, spatial segmentation | [04_hyperspectral_mapping.md](../examples/04_hyperspectral_mapping.md) |
| **Multi-Modal Fusion** | Late fusion, decision fusion, cross-modality correlation | [multimodal_workflows.md](../workflows/multimodal_workflows.md) |
| **Batch QC** | Drift detection, outlier detection, readiness score, health scores | [batch_quality_control.md](../workflows/batch_quality_control.md) |

---

## API Organization

| Module | Methods Exposed | Primary Use |
|--------|----------------|-------------|
| `foodspec.preprocess` | Baseline, normalization, smoothing, derivatives | Spectral preprocessing |
| `foodspec.chemometrics` | PCA, PLS, classifiers, mixture models | Multivariate analysis |
| `foodspec.stats` | Hypothesis tests, correlations, effect sizes | Statistical analysis |
| `foodspec.features` | Peaks, bands, ratios, RQ engine, similarity | Feature extraction |
| `foodspec.qc` | Drift, outliers, leakage, readiness, health | Quality control |
| `foodspec.ml` | Nested CV, fusion, hyperparameter tuning | Machine learning |
| `foodspec.metrics` | Classification/regression metrics, ROC/PR | Model evaluation |
| `foodspec.workflows` | Heating trajectory, library search, shelf life | Domain applications |

---

## Notes

1. **Scaffolded Methods**: `SpectrumHealthScore` is defined but not fully implemented (planned for v1.1)
2. **Internal Methods**: Methods marked "Internal" are implemented but not yet documented in user-facing pages
3. **Deprecated Methods**: Not included in this inventory (see CHANGELOG.md for deprecated APIs)
4. **Vendor I/O**: 10+ vendor format readers not listed (see [api/io.md](../api/io.md))
5. **Protocol System**: YAML-driven workflow orchestration spans multiple methods (see [protocols_and_yaml.md](../user-guide/protocols_and_yaml.md))

---

**Inventory Maintained By:** FoodSpec Documentation Team  
**Last Updated:** January 6, 2026  
**Next Review:** Post v1.1 release
