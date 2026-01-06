# MkDocs Navigation Architecture Proposal for JOSS

## Design Rationale

### Current Issues
1. **96+ nav entries** - overwhelming for reviewers
2. **Duplicates & deep nesting** - e.g., Registry appears in both User Guide and Workflows
3. **Internal pages mixed with user content** - design docs, deployment info shouldn't be front-row
4. **5 different help sections** - FAQ, Help, Troubleshooting, Help & Support scattered

### Reviewer Priorities
JOSS reviewers need:
1. **Quick start** (Getting Started)
2. **Real working examples** (Workflows: Oil Auth, QC, Mixtures)
3. **Methods justification** (Methods: Preprocessing, Chemometrics, Validation)
4. **API reference** (for software engineering review)
5. **Theory background** (for scientific review)
6. **Support paths** (FAQ, reporting issues)

### Proposed Structure (6 Top-Level Sections)

```
Home → Getting Started → Workflows → Methods → API → Theory → Help
```

**Rationale:**
- Getting Started: First thing reviewers do
- Workflows: "Does it work in practice?" (oil, QC, mixtures)
- Methods: "Is the science sound?"
- API: "Is the code well-designed?"
- Theory: "What's the foundation?"
- Help: "How do I learn more?"

---

## Proposed mkdocs.yml nav Block

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
      - Overview: methods/preprocessing/index.md
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
      - Overview: methods/statistics/overview.md
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
    - Data Format Ref: reference/data_format.md
    - Reproducibility: reproducibility.md
    - Reproducibility Checklist: protocols/reproducibility_checklist.md
    - Glossary: reference/glossary.md
    - Changelog: reference/changelog.md
    - Citing FoodSpec: reference/citing.md
    - Versioning: reference/versioning.md

  # === INTERNAL ONLY (Not in nav, but kept as files) ===
  # The following sections/files are valuable but archived from nav for clarity:
  # - Tutorials/ (beginner/intermediate/advanced) - available via search
  # - User Guide/ (except Workflows) - available via search
  # - Developer Guide/ - for contributors, not reviewers
  # - Reference/ - supporting tables, not critical path
  # - 05-advanced-topics/ - internal design docs
  # - _internal/ - project history & audit logs
```

---

## Navigation Changes Summary

### Removed from Nav (Files Kept, Just Hidden)

| Section | Reason | How to Access |
|---------|--------|---------------|
| Tutorials (Beginner/Intermediate/Advanced) | Redundant with Workflows + Getting Started | Search or direct link |
| User Guide (CLI, Protocols, Logging, etc.) | Too detailed; focus on workflows instead | Search or direct link |
| Developer Guide | For contributors, not JOSS reviewers | Search or direct link |
| Advanced Topics (05-*) | Internal architecture/deployment | Search or direct link |
| Reference (Most) | Supporting material, not critical | Search or direct link |
| _internal/ | Project history, not user-facing | Git history only |

### Added to Nav

| Section | Why |
|---------|-----|
| End-to-End Pipeline | Shows complete workflow from start to finish |
| Reproducibility Guide | Critical for JOSS: shows how to ensure reproduciblity |
| Reproducibility Checklist | Practical checklist for reviewers to verify |
| Data Format Reference | Reviewers need to understand data model |

### Consolidated

| Before | After | Benefit |
|--------|-------|---------|
| 3x help sections + FAQ | Single "Help & Docs" | Clear where to go |
| Registry + Library sections | Removed (search accessible) | Reduces clutter for reviewers |
| User Guide (14 entries) | Removed (search accessible) | Workflows are the entry point, not CLI |

---

## Design Principles

### 1. **Reviewer-First Path**
```
Home → Examples → Getting Started → Pick a Workflow → Methods → API → Theory
```

### 2. **Horizontal Scrolling Discouraged**
- No section has >15 items at any level
- Most sections have 5-10 items (ideal cognitive load)

### 3. **Duplicate Elimination**
- Removed "Registry & Reporting" from User Guide (it's under Workflows)
- Consolidated Help sections
- Removed "Deployment Artifacts" (internal)

### 4. **Search-Accessible Archive**
All files remain on disk and in search. Just not in nav. Example:
- User wants CLI tips? Search "CLI" → gets `user-guide/cli.md`
- User wants plugin info? Search "plugin" → gets `developer-guide/writing_plugins.md`

---

## Top 6 Entry Points Highlighted

1. **Getting Started** (first section after Home)
2. **Workflows → Oil Authentication** (most common use case)
3. **Workflows → Quality & Monitoring** (second most common)
4. **Methods → Validation** (for reproducibility review)
5. **API Reference** (for code review)
6. **Theory → Spectroscopy Basics** (scientific foundation)

---

## Rollback Plan

### If Something Breaks

**Step 1: Immediate Revert**
```bash
cd /home/cs/FoodSpec
git diff mkdocs.yml  # See what changed
git checkout mkdocs.yml  # Restore original
```

**Step 2: Verify Original Builds**
```bash
mkdocs build --strict
# Should complete without errors
```

**Step 3: Rebuild Docs**
```bash
mkdocs serve
# Visit http://localhost:8000 to verify original nav
```

### If You Want to Keep Changes But Fix Issues

```bash
# Make edits to mkdocs.yml with your text editor
# Test incrementally:
mkdocs build --strict

# If build fails, check for:
# 1. File path typos (must match exactly)
# 2. YAML syntax (indent by 2 spaces, not tabs)
# 3. Duplicate entries (search grep -n ":" mkdocs.yml)

# Once working, commit:
git add mkdocs.yml
git commit -m "refactor: streamline nav for JOSS reviewers"
git log --oneline -5  # See commit history
```

---

## Testing Checklist

- [ ] Copy proposed nav block into mkdocs.yml (replace old nav section)
- [ ] Run `mkdocs build --strict` (should complete in ~21 seconds)
- [ ] Run `mkdocs serve` and navigate each top-level section
- [ ] Verify no 404 errors in console
- [ ] Search for "oil authentication" - should find it
- [ ] Search for "plugins" - should find developer-guide/writing_plugins.md (via search, not nav)
- [ ] Search for "tutorials" - should find tutorials/index.md (via search)

---

## Files NOT Modified

- All `.md` files remain intact
- No files deleted
- No files renamed
- Only `mkdocs.yml` nav section changed

---

## Benefits for JOSS Review

✅ **Cleaner first impression** - 6-8 top sections instead of 20+  
✅ **Focused on examples** - Workflows section is prominent  
✅ **Scientific rigor visible** - Methods section shows statistical validation  
✅ **Reproducibility emphasized** - Reproducibility docs in "Help & Docs"  
✅ **API documented** - Separate API Reference section for code reviewers  
✅ **Nothing hidden** - All content searchable, just not in main nav  
✅ **Easy rollback** - Single file to edit, git-backed for safety  

