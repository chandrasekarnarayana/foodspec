## COPY THIS YAML BLOCK TO mkdocs.yml

Replace the entire `nav:` section (starting at `nav:` and ending at the line before `markdown_extensions:`)
with the following:

```yaml
nav:
  - Home: index.md
  - Examples: examples_gallery.md
  
  - Getting Started:
    - Overview: getting-started/index.md
    - Installation: getting-started/installation.md
    - 15-Minute Quickstart: getting-started/quickstart_15min.md
    - First Steps (CLI): getting-started/first-steps_cli.md
    - Understanding Results: getting-started/getting_started.md
    - FAQ: getting-started/faq_basic.md
  
  - Workflows:
    - Overview: workflows/index.md
    - Oil Authentication:
      - Complete Example: workflows/authentication/oil_authentication.md
      - Domain Templates: workflows/authentication/domain_templates.md
    - Quality & Monitoring:
      - Heating Quality: workflows/quality-monitoring/heating_quality_monitoring.md
      - Aging Analysis: workflows/quality-monitoring/aging_workflows.md
      - Batch QC: workflows/quality-monitoring/batch_quality_control.md
    - Quantification:
      - Mixture Analysis: workflows/quantification/mixture_analysis.md
      - Calibration: workflows/quantification/calibration_regression_example.md
    - Harmonization:
      - Multi-Instrument: workflows/harmonization/harmonization_automated_calibration.md
      - Calibration Transfer: workflows/harmonization/standard_templates.md
    - Spatial Analysis:
      - Hyperspectral Mapping: workflows/spatial/hyperspectral_mapping.md
    - End-to-End Pipeline: workflows/end_to_end_pipeline.md
    - Design & Reporting: workflows/workflow_design_and_reporting.md
  
  - Methods:
    - Preprocessing:
      - Baseline Correction: methods/preprocessing/baseline_correction.md
      - Normalization: methods/preprocessing/normalization_smoothing.md
      - Derivatives: methods/preprocessing/derivatives_and_feature_enhancement.md
      - Scatter Correction: methods/preprocessing/scatter_correction_cosmic_ray_removal.md
      - Feature Extraction: methods/preprocessing/feature_extraction.md
    - Chemometrics:
      - Models & Best Practices: methods/chemometrics/models_and_best_practices.md
      - Classification & Regression: methods/chemometrics/classification_regression.md
      - PCA: methods/chemometrics/pca_and_dimensionality_reduction.md
      - Mixture Models: methods/chemometrics/mixture_models.md
      - Model Evaluation: methods/chemometrics/model_evaluation_and_validation.md
      - Interpretability: methods/chemometrics/model_interpretability.md
    - Validation:
      - Cross-Validation & Leakage: methods/validation/cross_validation_and_leakage.md
      - Metrics & Uncertainty: methods/validation/metrics_and_uncertainty.md
      - Robustness Checks: methods/validation/robustness_checks.md
      - Reporting Standards: methods/validation/reporting_standards.md
    - Statistics:
      - Introduction: methods/statistics/introduction_to_statistical_analysis.md
      - T-Tests & Power: methods/statistics/t_tests_effect_sizes_and_power.md
      - ANOVA & MANOVA: methods/statistics/anova_and_manova.md
      - Correlation: methods/statistics/correlation_and_mapping.md
      - Nonparametric: methods/statistics/nonparametric_methods_and_robustness.md
      - Hypothesis Testing: methods/statistics/hypothesis_testing_in_food_spectroscopy.md
      - Study Design: methods/statistics/study_design_and_data_requirements.md
  
  - API Reference:
    - Overview: api/index.md
    - Core: api/core.md
    - Datasets: api/datasets.md
    - Preprocessing: api/preprocessing.md
    - Chemometrics: api/chemometrics.md
    - Features: api/features.md
    - I/O & Data: api/io.md
    - Workflows: api/workflows.md
    - Machine Learning: api/ml.md
    - Metrics: api/metrics.md
    - Statistics: api/stats.md
  
  - Theory:
    - Spectroscopy Basics: theory/spectroscopy_basics.md
    - Food Applications: theory/food_spectroscopy_applications.md
    - Chemometrics & ML: theory/chemometrics_and_ml_basics.md
    - RQ Engine: theory/rq_engine_detailed.md
    - Harmonization: theory/harmonization_theory.md
    - MOATS: theory/moats_overview.md
    - Data Structures & FAIR: theory/data_structures_and_fair_principles.md
  
  - Help & Docs:
    - Troubleshooting: troubleshooting/troubleshooting_faq.md
    - Common Problems: troubleshooting/common_problems_and_solutions.md
    - Reporting Issues: troubleshooting/reporting_guidelines.md
    - Data Formats: reference/data_format.md
    - Reproducibility Guide: reproducibility.md
    - Reproducibility Checklist: protocols/reproducibility_checklist.md
    - Glossary: reference/glossary.md
    - Changelog: reference/changelog.md
    - Citing: reference/citing.md
    - Versioning: reference/versioning.md
```

---

## ⚠️ IMPORTANT: YAML Indentation

- Use **2 spaces** per indent level (NOT tabs)
- Each nested item must have exactly 2 more spaces than parent
- After pasting, run: `mkdocs build --strict` to verify syntax

---

## What Gets Archived (Still Searchable)

The following sections will be removed from nav but remain on disk and searchable:

- `tutorials/` - All beginner/intermediate/advanced tutorials
- `user-guide/cli.md`, `protocols_and_yaml.md`, etc. - User guide pages
- `developer-guide/` - Developer documentation
- `05-advanced-topics/` - Internal architecture
- `_internal/` - Project history
- Most of `reference/` - Supporting tables
- Duplicate entries in nav

---

## Size Comparison

**Before:** 96+ nav entries across 15 top-level sections
**After:** 48 nav entries across 7 top-level sections

**Reduction:** ~50% simpler while keeping all content accessible
