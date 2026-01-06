# Ready-to-Paste YAML Block

## HOW TO USE THIS FILE

1. Open your text editor with `mkdocs.yml`
2. Find the line that says `nav:` (around line 163)
3. Select from `nav:` all the way down to (but NOT including) `markdown_extensions:`
4. Delete the entire selected section
5. Paste the content below (starting with `nav:`)
6. Save the file
7. Run: `mkdocs build --strict`
8. Verify: No errors

---

## ⚠️ IMPORTANT FORMATTING RULES

- Use **2 SPACES** for each indent level (NOT TABS)
- Each item must be indented exactly 2 more spaces than its parent
- No trailing whitespace
- Each list item starts with `- ` (dash space)

---

## PASTE THIS ENTIRE BLOCK

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

## AFTER PASTING

1. **Save the file**

2. **Run validation:**
   ```bash
   mkdocs build --strict
   ```
   
   Expected output:
   ```
   INFO    -  Cleaning site directory
   INFO    -  Building documentation...
   INFO    -  Documentation built in 20.89 seconds
   ```

3. **Test locally:**
   ```bash
   mkdocs serve
   ```
   
   Then open: http://localhost:8000

4. **Verify each section loads:**
   - Click "Getting Started"
   - Click "Workflows" → "Oil Authentication"
   - Click "Methods" → "Validation"
   - Click "API Reference"
   - Click "Theory"
   - Click "Help & Docs"

5. **Test search:**
   - Search for "oil authentication" (should find it)
   - Search for "preprocessing" (should find it)
   - Search for "leakage" (should find it)

6. **Commit when satisfied:**
   ```bash
   git add mkdocs.yml
   git commit -m "refactor(docs): streamline nav for JOSS reviewers
   
   - Reduced nav from 96+ to 48 entries
   - Organized by reviewer priority
   - Archived tutorials/user-guide/developer-guide from nav (still searchable)
   - Added reproducibility guide to Help & Docs"
   ```

---

## IF SOMETHING BREAKS

**Immediate rollback:**
```bash
git checkout mkdocs.yml
mkdocs build --strict
```

That's it. You're back to the original.

---

## FILE PATHS TO VERIFY

The YAML above references these files. All should exist:

```bash
# Should all return 0 bytes or file info (not "No such file")
ls docs/getting-started/installation.md
ls docs/workflows/authentication/oil_authentication.md
ls docs/methods/validation/cross_validation_and_leakage.md
ls docs/api/core.md
ls docs/theory/spectroscopy_basics.md
ls docs/troubleshooting/troubleshooting_faq.md
ls docs/reproducibility.md
ls docs/protocols/reproducibility_checklist.md
```

If any file is missing, update the path in the YAML.

---

## NEED HELP?

See also:
- `MKDOCS_IMPLEMENTATION_GUIDE.md` — Detailed step-by-step
- `MKDOCS_NAVIGATION_PROPOSAL.md` — Full design rationale
- `MKDOCS_SUMMARY.md` — Executive summary
- `rollback_mkdocs_nav.sh` — Automated rollback

