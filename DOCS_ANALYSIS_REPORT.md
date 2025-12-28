# Documentation Analysis & Restructuring Proposal

**Analysis Date:** December 28, 2025  
**Total Markdown Files:** 196  
**Files in mkdocs.yml nav:** 106  
**Files NOT in nav:** 90 (46% of total)

---

## 1. CURRENT SITE MAP (Grouped by Folder)

### Published Folders (in nav)
```
docs/
├── 01-getting-started/ (9 files: 8 in nav, 1 index not in nav)
│   ├── quickstart_15min.md ✓
│   ├── installation.md ✓
│   ├── getting_started.md ✓
│   ├── quickstart_cli.md ✓
│   ├── quickstart_python.md ✓
│   ├── quickstart_protocol.md ✓
│   ├── first-steps_cli.md ✓
│   ├── faq_basic.md ✓
│   └── index.md ✗ (not in nav)
│
├── 02-tutorials/ (13 files: 11 in nav, 2 GUI files excluded)
│   ├── index.md ✓
│   ├── level1_load_and_plot.md ✓
│   ├── level1_baseline_and_smoothing.md ✓
│   ├── level1_simple_classification.md ✓
│   ├── oil_discrimination_basic.md ✓
│   ├── oil_vs_chips_matrix_effects.md ✓
│   ├── thermal_stability_tracking.md ✓
│   ├── level3_reproducible_pipelines.md ✓
│   ├── reference_analysis_oil_authentication.md ✓
│   ├── hsi_surface_mapping.md ✓
│   ├── end_to_end_notebooks.md ✓
│   ├── modeling_gui_foodspec_workflow.md ✗ (GUI - deprecated?)
│   └── raman_gui_quickstart.md ✗ (GUI - deprecated?)
│
├── 03-cookbook/ (14 files: 9 in nav, 5 not in nav)
│   ├── cookbook_intro.md ✓
│   ├── preprocessing_guide.md ✓
│   ├── chemometrics_guide.md ✓
│   ├── ftir_raman_preprocessing.md ✓
│   ├── cookbook_rq_questions.md ✓
│   ├── cookbook_validation.md ✓
│   ├── validation_baseline.md ✓
│   ├── validation_chemometrics_oils.md ✓
│   ├── validation_peak_ratios.md ✓
│   ├── protocol_cookbook.md ✓
│   ├── cookbook_registry_reporting.md ✓
│   ├── index.md ✗ (duplicate of cookbook_intro?)
│   ├── cookbook_preprocessing.md ✗ (vs preprocessing_guide?)
│   └── cookbook_troubleshooting.md ✗ (not in nav)
│
├── 04-user-guide/ (15 files: 12 in nav, 3 not in nav)
│   ├── cli.md ✓
│   ├── cli_help.md ✓
│   ├── protocols_and_yaml.md ✓
│   ├── automation.md ✓
│   ├── protocol_profiles.md ✓
│   ├── config_logging.md ✓
│   ├── data_governance.md ✓
│   ├── libraries.md ✓
│   ├── library_search.md ✓
│   ├── csv_to_library.md ✓
│   ├── vendor_io.md ✓
│   ├── data_formats_and_hdf5.md ✓
│   ├── registry_and_plugins.md ✓
│   ├── index.md ✗ (not in nav)
│   ├── cli_guide.md ✗ (duplicate of cli.md?)
│   └── logging.md ✗ (duplicate of config_logging?)
│
├── 05-advanced-topics/ (12 files: 7 in nav, 5 not in nav)
│   ├── validation_strategies.md ✓
│   ├── model_lifecycle.md ✓
│   ├── architecture.md ✓
│   ├── design_overview.md ✓
│   ├── model_registry.md ✓
│   ├── MOATS_IMPLEMENTATION.md ✓
│   ├── multimodal_workflows.md ✓
│   ├── deployment_artifact_versioning.md ✓
│   ├── deployment_hdf5_schema_versioning.md ✓
│   ├── index.md ✗ (not in nav)
│   ├── advanced_deep_learning.md ✗ (not in nav)
│   ├── hsi_and_harmonization.md ✗ (duplicate?)
│   └── model_lifecycle_and_prediction.md ✗ (duplicate?)
│
├── 06-developer-guide/ (11 files: 9 in nav, 2 not in nav)
│   ├── contributing.md ✓
│   ├── writing_plugins.md ✓
│   ├── extending_protocols_and_steps.md ✓
│   ├── documentation_guidelines.md ✓
│   ├── documentation_style_guide.md ✓
│   ├── documentation_maintainer_guide.md ✓
│   ├── testing_and_ci.md ✓
│   ├── testing_coverage.md ✓
│   ├── releasing.md ✓
│   ├── index.md ✗ (not in nav)
│   ├── RELEASE_CHECKLIST.md ✗ (internal)
│   └── RELEASING.md ✗ (duplicate of releasing.md?)
│
├── 06-tutorials/ (1 file: excluded from nav intentionally)
│   └── example_catalog.md ✗ (internal use)
│
├── 07-theory-and-background/ (8 files: 5 in nav, 3 not in nav)
│   ├── spectroscopy_basics.md ✓
│   ├── chemometrics_and_ml_basics.md ✓
│   ├── rq_engine_detailed.md ✓
│   ├── harmonization_theory.md ✓
│   ├── moats_overview.md ✓
│   ├── index.md ✗ (not in nav)
│   ├── domains_overview.md ✗ (not in nav)
│   └── rq_engine_theory.md ✗ (duplicate of rq_engine_detailed?)
│
├── 07-validation/ (5 files: 4 in nav, 1 not in nav)
│   ├── cross_validation_and_leakage.md ✓
│   ├── metrics_and_uncertainty.md ✓
│   ├── robustness_checks.md ✓
│   ├── reporting_standards.md ✓
│   └── index.md ✗ (not in nav)
│
├── 08-api/ (10 files: 7 in nav, 3 not in nav)
│   ├── index.md ✓
│   ├── core.md ✓
│   ├── datasets.md ✓
│   ├── features.md ✓
│   ├── io.md ✓
│   ├── workflows.md ✓
│   ├── ml.md ✓
│   ├── stats.md ✓
│   ├── chemometrics.md ✗ (not in nav - should be?)
│   ├── metrics.md ✗ (not in nav - should be?)
│   └── preprocessing.md ✗ (not in nav - should be?)
│
├── 09-reference/ (9 files: 7 in nav, 2 not in nav)
│   ├── metric_significance_tables.md ✓
│   ├── glossary.md ✓
│   ├── keyword_index.md ✓
│   ├── method_comparison.md ✓
│   ├── ml_model_vip_scores.md ✓
│   ├── changelog.md ✓
│   ├── citing.md ✓
│   ├── versioning.md ✓
│   ├── index.md ✗ (not in nav)
│   └── data_format.md ✗ (not in nav - should be?)
│
├── 10-help/ (3 files: 2 in nav, 1 not in nav)
│   ├── troubleshooting.md ✓
│   ├── faq.md ✓
│   └── index.md ✗ (not in nav)
│
├── foundations/ (5 files: 4 in nav, 1 not in nav)
│   ├── index.md ✓
│   ├── spectroscopy_basics.md ✓
│   ├── food_spectroscopy_applications.md ✓
│   ├── data_structures_and_fair_principles.md ✓
│   └── introduction.md ✗ (duplicate of index?)
│
├── workflows/ (12 files: 11 in nav, 1 not in nav)
│   ├── index.md ✓
│   ├── oil_authentication.md ✓
│   ├── domain_templates.md ✓
│   ├── heating_quality_monitoring.md ✓
│   ├── aging_workflows.md ✓
│   ├── batch_quality_control.md ✓
│   ├── mixture_analysis.md ✓
│   ├── calibration_regression_example.md ✓
│   ├── harmonization_automated_calibration.md ✓
│   ├── standard_templates.md ✓
│   ├── hyperspectral_mapping.md ✓
│   └── workflow_design_and_reporting.md ✓
│
├── protocols/ (8 files: 5 in nav, 3 not in nav)
│   ├── protocols_overview.md ✓
│   ├── reference_protocol.md ✓
│   ├── methods_text_generator.md ✓
│   ├── protocol_benchmarks.md ✓
│   ├── statistical_power_and_limits.md ✓
│   ├── benchmarking_framework.md ✗ (not in nav)
│   ├── decision_trees.md ✗ (not in nav)
│   └── reproducibility_checklist.md ✗ (not in nav)
│
├── preprocessing/ (5 files: all in nav)
│   ├── baseline_correction.md ✓
│   ├── normalization_smoothing.md ✓
│   ├── derivatives_and_feature_enhancement.md ✓
│   ├── scatter_correction_cosmic_ray_removal.md ✓
│   └── feature_extraction.md ✓
│
├── ml/ (6 files: all in nav)
│   ├── models_and_best_practices.md ✓
│   ├── classification_regression.md ✓
│   ├── pca_and_dimensionality_reduction.md ✓
│   ├── mixture_models.md ✓
│   ├── model_evaluation_and_validation.md ✓
│   └── model_interpretability.md ✓
│
├── stats/ (8 files: none in nav - orphaned?)
│   ├── overview.md ✗
│   ├── introduction_to_statistical_analysis.md ✗
│   ├── t_tests_effect_sizes_and_power.md ✗
│   ├── anova_and_manova.md ✗
│   ├── correlation_and_mapping.md ✗
│   ├── nonparametric_methods_and_robustness.md ✗
│   ├── hypothesis_testing_in_food_spectroscopy.md ✗
│   └── study_design_and_data_requirements.md ✗
│
├── metrics/ (1 file: orphaned)
│   └── metrics_and_evaluation.md ✗
│
├── troubleshooting/ (3 files: none in nav - orphaned?)
│   ├── common_problems_and_solutions.md ✗
│   ├── reporting_guidelines.md ✗
│   └── troubleshooting_faq.md ✗
│
├── user_guide/ (1 file: orphaned)
│   └── instrument_file_formats.md ✗
│
├── visualization/ (1 file: orphaned)
│   └── plotting_with_foodspec.md ✗
│
├── datasets/ (1 file: orphaned)
│   └── dataset_design.md ✗
│
├── design/ (1 file: orphaned)
│   └── 01_overview.md ✗
│
├── examples_gallery.md ✓ (new - just added)
├── index.md ✓
├── non_goals_and_limitations.md ✓
└── API_CONSISTENCY_REPORT.md ✗ (internal)
```

### Internal/Archive Folders (mostly in nav under "[INTERNAL]")
```
_internal/
├── archive/ (46 files - historical, audit reports, old API docs)
│   ├── README.md ✓ (in nav)
│   ├── api_*.md (8 files - superseded by 08-api/)
│   ├── *AUDIT*.md (6 files - project audits)
│   ├── *COMPLETE.md (5 files - completion reports)
│   └── project_history/ (7 files - historical)
│
├── developer-guide/ (7 files: 6 in nav under [INTERNAL])
│   ├── GAPS_AND_FUTURE_WORK.md ✓
│   ├── FEATURE_INVENTORY.md ✓
│   ├── integration_checklist.md ✓
│   ├── ci_troubleshooting.md ✓
│   ├── developer_notes.md ✓
│   └── design_stats_and_analysis.md ✓
│
├── dev/ (3 files - not in nav)
│   ├── design_stats_and_analysis.md
│   ├── developer_notes.md
│   └── smoke_test_results_2025-12-25.md
│
└── api-duplicate/ (10 files - exact duplicate of 08-api/)
    └── [all API files duplicated]
```

---

## 2. PAGES WITH PLACEHOLDERS (TODO/TBD/...)

### High-Priority Public Pages (need completion)

**index.md (Home)**
- Line 21: Table with "..." in "You are..." column
- **Action:** Replace with proper role descriptions or remove table

**01-getting-started/quickstart_cli.md**
- Lines 57, 72, 248, 252, 314: Command examples with `...` placeholders
- **Action:** Replace with full working examples or use explicit ellipsis like `[options]`

**04-user-guide/library_search.md**
- Line 11: Data format example with `...` in wavenumber list
- **Action:** Show actual example values: `1000,1005,1010,1015,...,1800`

**05-advanced-topics/model_lifecycle.md**
- Lines 21-33: API examples with `...` for parameters
- **Action:** Provide concrete parameter examples or mark as conceptual

**08-api/index.md**
- Line 146: Code block with `...` placeholder
- **Action:** Complete code example or remove incomplete snippet

**foundations/data_structures_and_fair_principles.md**
- Line 32: Constructor example with `...` placeholders
- **Action:** Show real example: `x=spectra_array, wavenumbers=wn, metadata=df, modality="raman"`

**02-tutorials/level1_simple_classification.md**
- Lines 246, 464: Code with `...` placeholders
- **Action:** Complete examples with actual array shapes/values

**ml/models_and_best_practices.md**
- Line 127: Train/test split with `...`
- **Action:** Show full train_test_split example

### Medium-Priority (mostly internal/archive)

**Multiple _internal/archive/ files** contain TODOs and incomplete sections
- **Action:** These are historical - leave as-is or clean up if archive is ever published

**_internal/developer-guide/ files** have some TODOs
- **Action:** Update roadmap/gaps docs or mark as living documents

---

## 3. PAGES THAT SHOULD NOT BE PUBLISHED

### Definitely Internal (95% confidence)

**Root-level internal reports:**
- `API_CONSISTENCY_REPORT.md` - internal quality report

**Entire folders to exclude:**
- `_internal/archive/` (46 files) - historical project documents
- `_internal/api-duplicate/` (10 files) - exact copy of 08-api/
- `_internal/dev/` (3 files) - developer scratch notes
- `_internal/developer-guide/` (7 files) - internal roadmaps (currently in nav but marked [INTERNAL])

**Orphaned/unused files:**
- `datasets/dataset_design.md` - incomplete design doc
- `design/01_overview.md` - early design sketch
- `06-tutorials/example_catalog.md` - internal inventory (intentionally excluded from nav)

### Probably Redundant/Deprecated (80% confidence)

**GUI-related (no GUI exists):**
- `02-tutorials/modeling_gui_foodspec_workflow.md`
- `02-tutorials/raman_gui_quickstart.md`

**Duplicate files (pick one version):**
- `03-cookbook/index.md` vs `03-cookbook/cookbook_intro.md`
- `03-cookbook/cookbook_preprocessing.md` vs `03-cookbook/preprocessing_guide.md`
- `04-user-guide/cli_guide.md` vs `04-user-guide/cli.md`
- `04-user-guide/logging.md` vs `04-user-guide/config_logging.md`
- `05-advanced-topics/model_lifecycle_and_prediction.md` vs `05-advanced-topics/model_lifecycle.md`
- `05-advanced-topics/hsi_and_harmonization.md` vs `07-theory-and-background/harmonization_theory.md`
- `07-theory-and-background/rq_engine_theory.md` vs `07-theory-and-background/rq_engine_detailed.md`
- `06-developer-guide/RELEASING.md` vs `06-developer-guide/releasing.md`
- `foundations/introduction.md` vs `foundations/index.md`

### Orphaned Content (needs integration or removal)

**stats/ folder (8 files)** - comprehensive statistics guides not linked anywhere
- **Decision:** Either add to nav under a "Statistics Deep Dive" section or merge into existing guides

**troubleshooting/ folder (3 files)** - separate from `10-help/troubleshooting.md`
- **Decision:** Merge into `10-help/troubleshooting.md` or link as supplementary

**Singleton orphaned files:**
- `metrics/metrics_and_evaluation.md` - should be in 08-api/ or 09-reference/
- `user_guide/instrument_file_formats.md` - should be in 04-user-guide/
- `visualization/plotting_with_foodspec.md` - should be in 04-user-guide/ or 08-api/

---

## 4. PROPOSED NEW INFORMATION ARCHITECTURE

### Philosophy: Three Clear Paths

1. **Beginner Path** - Zero to productive in <30 minutes
2. **Practitioner Path** - Domain experts using FoodSpec for research
3. **Developer Path** - Contributors and extenders

### Proposed Structure

```
docs/
│
├── index.md (Home with clear path signposting)
├── examples-gallery.md (Quick recipe cards)
│
├── getting-started/          [BEGINNER PATH START]
│   ├── index.md (Welcome - which path?)
│   ├── quickstart-15min.md
│   ├── installation.md
│   ├── first-workflow.md (merge quickstart_python + quickstart_cli)
│   ├── understanding-results.md (new - how to read outputs)
│   └── faq-basics.md
│
├── tutorials/                [BEGINNER → PRACTITIONER]
│   ├── index.md (Learning ladder)
│   ├── beginner/
│   │   ├── load-and-plot.md
│   │   ├── preprocess.md
│   │   └── classify.md
│   ├── intermediate/
│   │   ├── oil-authentication.md
│   │   ├── matrix-effects.md
│   │   └── validation.md
│   └── advanced/
│       ├── reproducible-pipelines.md
│       ├── reference-workflow.md
│       └── hsi-mapping.md
│
├── workflows/                [PRACTITIONER PATH]
│   ├── index.md (Domain overview)
│   ├── authentication/
│   │   ├── oil-authentication.md
│   │   └── domain-templates.md
│   ├── quality-monitoring/
│   │   ├── heating-quality.md
│   │   ├── aging.md
│   │   └── batch-qc.md
│   ├── quantification/
│   │   ├── mixture-analysis.md
│   │   └── calibration.md
│   ├── harmonization/
│   │   ├── multi-instrument.md
│   │   └── calibration-transfer.md
│   └── spatial/
│       └── hyperspectral-mapping.md
│
├── methods/                  [PRACTITIONER - TECHNICAL]
│   ├── preprocessing/
│   │   ├── baseline-correction.md
│   │   ├── normalization.md
│   │   ├── derivatives.md
│   │   ├── scatter-correction.md
│   │   └── feature-extraction.md
│   ├── chemometrics/
│   │   ├── pca.md
│   │   ├── classification.md
│   │   ├── regression.md
│   │   └── mixture-models.md
│   ├── validation/
│   │   ├── cross-validation.md
│   │   ├── metrics.md
│   │   ├── robustness.md
│   │   └── reporting.md
│   └── statistics/
│       ├── hypothesis-testing.md
│       ├── power-analysis.md
│       └── study-design.md
│
├── user-guide/               [PRACTITIONER - OPERATIONS]
│   ├── cli-reference.md
│   ├── python-api-guide.md
│   ├── protocols-and-yaml.md
│   ├── data-formats.md
│   ├── data-governance.md
│   ├── libraries.md
│   ├── automation.md
│   └── logging-config.md
│
├── theory/                   [DEEP UNDERSTANDING]
│   ├── spectroscopy-basics.md
│   ├── food-spectroscopy.md
│   ├── chemometrics-foundations.md
│   ├── rq-engine.md
│   ├── harmonization.md
│   ├── moats.md
│   └── fair-principles.md
│
├── api/                      [DEVELOPER PATH - CODE]
│   ├── index.md
│   ├── core.md
│   ├── datasets.md
│   ├── preprocessing.md
│   ├── chemometrics.md
│   ├── features.md
│   ├── ml.md
│   ├── stats.md
│   ├── metrics.md
│   ├── io.md
│   └── workflows.md
│
├── developer-guide/          [DEVELOPER PATH - EXTEND]
│   ├── contributing.md
│   ├── architecture.md
│   ├── writing-plugins.md
│   ├── extending-protocols.md
│   ├── testing.md
│   ├── documentation.md
│   └── releasing.md
│
├── reference/                [LOOKUP]
│   ├── glossary.md
│   ├── method-comparison.md
│   ├── metric-tables.md
│   ├── data-format-spec.md
│   ├── changelog.md
│   ├── citing.md
│   └── versioning.md
│
├── help/                     [SUPPORT]
│   ├── troubleshooting.md (comprehensive - merge orphaned)
│   ├── faq.md (full FAQ - merge faq-basics)
│   └── community.md (new - where to get help)
│
└── _internal/                [NEVER PUBLISHED]
    ├── archive/ (history)
    ├── dev-notes/
    └── reports/
```

---

## 5. CONCRETE ACTION PLAN

### Phase 1: Cleanup & Deduplication (No Breaking Changes)

**1.1 Remove Definitely Internal/Redundant Files**
```bash
# Delete exact duplicates
rm -rf docs/_internal/api-duplicate/
rm docs/API_CONSISTENCY_REPORT.md
rm docs/datasets/dataset_design.md
rm docs/design/01_overview.md

# Remove deprecated GUI tutorials
rm docs/02-tutorials/modeling_gui_foodspec_workflow.md
rm docs/02-tutorials/raman_gui_quickstart.md

# Remove duplicate releasing doc
rm docs/06-developer-guide/RELEASING.md  # Keep releasing.md
```

**1.2 Merge Duplicate Content (Keep Best Version)**
```bash
# Cookbook: Keep preprocessing_guide.md, delete cookbook_preprocessing.md
# Action: Merge any unique content from cookbook_preprocessing → preprocessing_guide
# Then: rm docs/03-cookbook/cookbook_preprocessing.md

# CLI Guide: Keep cli.md, delete cli_guide.md
# Action: Verify cli.md has all content from cli_guide.md
# Then: rm docs/04-user-guide/cli_guide.md

# Logging: Keep config_logging.md, delete logging.md
# Action: Merge logging.md → config_logging.md
# Then: rm docs/04-user-guide/logging.md

# RQ Engine: Keep rq_engine_detailed.md, delete rq_engine_theory.md
# Action: Ensure rq_engine_detailed.md has all theory content
# Then: rm docs/07-theory-and-background/rq_engine_theory.md

# Foundations: Keep index.md, delete introduction.md
# Action: Merge introduction.md → index.md if unique content
# Then: rm docs/foundations/introduction.md
```

**1.3 Add Missing API Pages to Nav**
```yaml
# In mkdocs.yml under "API Reference":
- API Reference:
    - Overview: 08-api/index.md
    - Core API: 08-api/core.md
    - Datasets: 08-api/datasets.md
    - Preprocessing: 08-api/preprocessing.md      # ADD
    - Chemometrics: 08-api/chemometrics.md        # ADD
    - Features: 08-api/features.md
    - Machine Learning: 08-api/ml.md
    - Statistics: 08-api/stats.md
    - Metrics: 08-api/metrics.md                  # ADD
    - I/O & Data: 08-api/io.md
    - Workflows: 08-api/workflows.md
```

**1.4 Integrate Orphaned Stats Content**
```yaml
# Decision: Add stats/ folder to nav under "Methods & Statistics"
# OR: Move stats/*.md → methods/statistics/ in Phase 2

# Quick fix for now: Add to nav:
- Statistics Deep Dive:
    - Overview: stats/overview.md
    - Introduction: stats/introduction_to_statistical_analysis.md
    - T-Tests & Power: stats/t_tests_effect_sizes_and_power.md
    - ANOVA & MANOVA: stats/anova_and_manova.md
    - Correlation: stats/correlation_and_mapping.md
    - Nonparametric Methods: stats/nonparametric_methods_and_robustness.md
    - Hypothesis Testing: stats/hypothesis_testing_in_food_spectroscopy.md
    - Study Design: stats/study_design_and_data_requirements.md
```

**1.5 Move Orphaned Singleton Files**
```bash
# Move to proper locations
mv docs/metrics/metrics_and_evaluation.md docs/09-reference/metrics-reference.md
mv docs/user_guide/instrument_file_formats.md docs/04-user-guide/vendor-formats.md
mv docs/visualization/plotting_with_foodspec.md docs/04-user-guide/visualization.md

# Update internal links after moves
# Clean up empty dirs: rmdir docs/metrics docs/user_guide docs/visualization
```

**1.6 Fix Placeholders in Public Pages**
```markdown
# Files to edit (with line numbers from grep):
1. docs/index.md (line 21) - Replace table "..." with roles
2. docs/01-getting-started/quickstart_cli.md (lines 57,72,248,252,314) - Complete command examples
3. docs/04-user-guide/library_search.md (line 11) - Show full wavenumber example
4. docs/05-advanced-topics/model_lifecycle.md (lines 21-33) - Add concrete params
5. docs/08-api/index.md (line 146) - Complete code snippet
6. docs/foundations/data_structures_and_fair_principles.md (line 32) - Full constructor
7. docs/02-tutorials/level1_simple_classification.md (lines 246,464) - Complete arrays
8. docs/ml/models_and_best_practices.md (line 127) - Full train_test_split
```

### Phase 2: Restructure (Breaking Changes - Requires Redirects)

**2.1 Rename Numbered Folders to Descriptive Names**
```bash
# Create new structure
mkdir -p docs/{getting-started,tutorials,workflows,methods,theory,api,developer-guide,reference,help}

# Move content (examples - not exhaustive)
mv docs/01-getting-started/* docs/getting-started/
mv docs/02-tutorials/* docs/tutorials/
mv docs/08-api/* docs/api/
mv docs/06-developer-guide/* docs/developer-guide/
mv docs/09-reference/* docs/reference/
mv docs/10-help/* docs/help/

# Theory consolidation
mv docs/07-theory-and-background/* docs/theory/
mv docs/foundations/* docs/theory/  # Merge foundations into theory

# Methods consolidation
mkdir -p docs/methods/{preprocessing,chemometrics,validation,statistics}
mv docs/preprocessing/* docs/methods/preprocessing/
mv docs/ml/* docs/methods/chemometrics/
mv docs/07-validation/* docs/methods/validation/
mv docs/stats/* docs/methods/statistics/

# Workflows already good
# mv docs/workflows/* docs/workflows/  # No change

# Clean up old dirs
rmdir docs/{01-getting-started,02-tutorials,03-cookbook,04-user-guide,05-advanced-topics,06-developer-guide,07-theory-and-background,07-validation,08-api,09-reference,10-help,preprocessing,ml,foundations}
```

**2.2 Reorganize Tutorials by Level**
```bash
mkdir -p docs/tutorials/{beginner,intermediate,advanced}

# Beginner
mv docs/tutorials/level1_load_and_plot.md docs/tutorials/beginner/01-load-and-plot.md
mv docs/tutorials/level1_baseline_and_smoothing.md docs/tutorials/beginner/02-preprocess.md
mv docs/tutorials/level1_simple_classification.md docs/tutorials/beginner/03-classify.md

# Intermediate
mv docs/tutorials/oil_discrimination_basic.md docs/tutorials/intermediate/01-oil-authentication.md
mv docs/tutorials/oil_vs_chips_matrix_effects.md docs/tutorials/intermediate/02-matrix-effects.md
mv docs/tutorials/thermal_stability_tracking.md docs/tutorials/intermediate/03-stability.md

# Advanced
mv docs/tutorials/level3_reproducible_pipelines.md docs/tutorials/advanced/01-reproducible-pipelines.md
mv docs/tutorials/reference_analysis_oil_authentication.md docs/tutorials/advanced/02-reference-workflow.md
mv docs/tutorials/hsi_surface_mapping.md docs/tutorials/advanced/03-hsi-mapping.md

# Keep supplementary at top level
# docs/tutorials/end_to_end_notebooks.md stays
```

**2.3 Reorganize Workflows by Domain**
```bash
mkdir -p docs/workflows/{authentication,quality-monitoring,quantification,harmonization,spatial}

# Authentication
mv docs/workflows/oil_authentication.md docs/workflows/authentication/
mv docs/workflows/domain_templates.md docs/workflows/authentication/

# Quality Monitoring
mv docs/workflows/heating_quality_monitoring.md docs/workflows/quality-monitoring/
mv docs/workflows/aging_workflows.md docs/workflows/quality-monitoring/
mv docs/workflows/batch_quality_control.md docs/workflows/quality-monitoring/

# Quantification
mv docs/workflows/mixture_analysis.md docs/workflows/quantification/
mv docs/workflows/calibration_regression_example.md docs/workflows/quantification/

# Harmonization
mv docs/workflows/harmonization_automated_calibration.md docs/workflows/harmonization/
mv docs/workflows/standard_templates.md docs/workflows/harmonization/

# Spatial
mv docs/workflows/hyperspectral_mapping.md docs/workflows/spatial/

# Keep at root: index.md, workflow_design_and_reporting.md
```

**2.4 Consolidate User Guide**
```bash
# Merge protocol_profiles.md content into protocols_and_yaml.md
# Merge csv_to_library.md content into libraries.md
# Result: Cleaner user-guide/ folder with ~8 essential guides
```

**2.5 Update mkdocs.yml with New Structure**
```yaml
nav:
  - Home: index.md
  - Examples Gallery: examples-gallery.md
  
  - Getting Started:
      - Welcome: getting-started/index.md
      - 15-Minute Quickstart: getting-started/quickstart-15min.md
      - Installation: getting-started/installation.md
      - Your First Workflow: getting-started/first-workflow.md
      - Understanding Results: getting-started/understanding-results.md
      - FAQ: getting-started/faq-basics.md
  
  - Tutorials:
      - Learning Path: tutorials/index.md
      - Beginner:
          - Load & Plot: tutorials/beginner/01-load-and-plot.md
          - Preprocessing: tutorials/beginner/02-preprocess.md
          - Classification: tutorials/beginner/03-classify.md
      - Intermediate:
          - Oil Authentication: tutorials/intermediate/01-oil-authentication.md
          - Matrix Effects: tutorials/intermediate/02-matrix-effects.md
          - Stability Tracking: tutorials/intermediate/03-stability.md
      - Advanced:
          - Reproducible Pipelines: tutorials/advanced/01-reproducible-pipelines.md
          - Reference Workflow: tutorials/advanced/02-reference-workflow.md
          - HSI Mapping: tutorials/advanced/03-hsi-mapping.md
      - Notebooks: tutorials/end-to-end-notebooks.md
  
  - Workflows:
      - Overview: workflows/index.md
      - Authentication:
          - Oil Authentication: workflows/authentication/oil-authentication.md
          - Domain Templates: workflows/authentication/domain-templates.md
      - Quality Monitoring:
          - Heating Quality: workflows/quality-monitoring/heating-quality.md
          - Aging Analysis: workflows/quality-monitoring/aging.md
          - Batch QC: workflows/quality-monitoring/batch-qc.md
      - Quantification:
          - Mixture Analysis: workflows/quantification/mixture-analysis.md
          - Calibration: workflows/quantification/calibration.md
      - Harmonization:
          - Multi-Instrument: workflows/harmonization/multi-instrument.md
          - Calibration Transfer: workflows/harmonization/calibration-transfer.md
      - Spatial Analysis:
          - Hyperspectral Mapping: workflows/spatial/hyperspectral-mapping.md
      - Design & Reporting: workflows/workflow-design.md
  
  - Methods:
      - Preprocessing:
          - Baseline Correction: methods/preprocessing/baseline-correction.md
          - Normalization: methods/preprocessing/normalization.md
          - Derivatives: methods/preprocessing/derivatives.md
          - Scatter Correction: methods/preprocessing/scatter-correction.md
          - Feature Extraction: methods/preprocessing/feature-extraction.md
      - Chemometrics:
          - PCA: methods/chemometrics/pca.md
          - Classification: methods/chemometrics/classification.md
          - Regression: methods/chemometrics/regression.md
          - Mixture Models: methods/chemometrics/mixtures.md
      - Validation:
          - Cross-Validation: methods/validation/cross-validation.md
          - Metrics: methods/validation/metrics.md
          - Robustness: methods/validation/robustness.md
          - Reporting Standards: methods/validation/reporting.md
      - Statistics:
          - Overview: methods/statistics/overview.md
          - Hypothesis Testing: methods/statistics/hypothesis-testing.md
          - Power Analysis: methods/statistics/power.md
          - Study Design: methods/statistics/study-design.md
  
  - User Guide:
      - CLI Reference: user-guide/cli-reference.md
      - Python API: user-guide/python-api.md
      - Protocols & YAML: user-guide/protocols-yaml.md
      - Data Formats: user-guide/data-formats.md
      - Data Governance: user-guide/data-governance.md
      - Libraries: user-guide/libraries.md
      - Automation: user-guide/automation.md
      - Logging & Config: user-guide/logging-config.md
      - Visualization: user-guide/visualization.md
  
  - Theory:
      - Spectroscopy Basics: theory/spectroscopy-basics.md
      - Food Spectroscopy: theory/food-spectroscopy.md
      - Chemometrics Foundations: theory/chemometrics-foundations.md
      - RQ Engine: theory/rq-engine.md
      - Harmonization: theory/harmonization.md
      - MOATS: theory/moats.md
      - FAIR Principles: theory/fair-principles.md
  
  - API Reference:
      - Overview: api/index.md
      - Core: api/core.md
      - Datasets: api/datasets.md
      - Preprocessing: api/preprocessing.md
      - Chemometrics: api/chemometrics.md
      - Features: api/features.md
      - ML: api/ml.md
      - Statistics: api/stats.md
      - Metrics: api/metrics.md
      - I/O: api/io.md
      - Workflows: api/workflows.md
  
  - Developer Guide:
      - Contributing: developer-guide/contributing.md
      - Architecture: developer-guide/architecture.md
      - Writing Plugins: developer-guide/plugins.md
      - Extending Protocols: developer-guide/protocols.md
      - Testing: developer-guide/testing.md
      - Documentation: developer-guide/documentation.md
      - Releasing: developer-guide/releasing.md
  
  - Reference:
      - Glossary: reference/glossary.md
      - Method Comparison: reference/method-comparison.md
      - Metric Tables: reference/metric-tables.md
      - Data Format Spec: reference/data-format.md
      - Changelog: reference/changelog.md
      - Citing FoodSpec: reference/citing.md
      - Versioning: reference/versioning.md
      - Limitations: reference/limitations.md
  
  - Help:
      - Troubleshooting: help/troubleshooting.md
      - FAQ: help/faq.md
      - Community: help/community.md
```

### Phase 3: Path Optimization

**3.1 Create Path Landing Pages**
```markdown
# Create docs/getting-started/index.md
---
# Welcome to FoodSpec

Choose your path:

## 🎓 Beginner Path
New to FoodSpec or food spectroscopy? Start here.
→ [15-Minute Quickstart](quickstart-15min.md)

## 🔬 Practitioner Path
Research scientist or QC analyst? Jump to workflows.
→ [Workflows Overview](../workflows/index.md)

## 💻 Developer Path
Want to extend FoodSpec or contribute?
→ [Developer Guide](../developer-guide/contributing.md)
---
```

**3.2 Add Path Signposts Throughout**
```markdown
# Add "Next Steps" to every tutorial:
- Beginner: "Next → [Intermediate Tutorials](../intermediate/index.md)"
- Intermediate: "Next → [Advanced Tutorials](../advanced/index.md)"
- Advanced: "Next → [Build Your Own Workflow](../../workflows/workflow-design.md)"

# Add "Prerequisites" to every page
# Add "Related" cross-links at bottom
```

### Phase 4: Final Cleanup

**4.1 Archive Management**
```yaml
# Decide: Keep _internal/ in repo but NEVER publish
# Option 1: Add to .gitignore for gh-pages branch
# Option 2: Exclude from mkdocs.yml explicitly
# Option 3: Move to separate docs-internal/ repo

# Recommended: Keep in repo, exclude from build
# In mkdocs.yml:
exclude_docs: |
  _internal/
  **/TODO.md
  **/*DRAFT*.md
```

**4.2 Redirect Map**
```yaml
# Create redirects.yml for Phase 2 moves
redirects:
  01-getting-started/quickstart_15min.md: getting-started/quickstart-15min.md
  02-tutorials/level1_load_and_plot.md: tutorials/beginner/01-load-and-plot.md
  08-api/index.md: api/index.md
  # ... (full map for all 106+ moved files)
```

**4.3 CI/CD Integration**
```yaml
# Add to GitHub Actions workflow
- name: Check for broken links
  run: python scripts/validate_docs.py --check-links
  
- name: Verify all nav entries exist
  run: python scripts/validate_docs.py --check-nav

- name: Check for TODO/TBD in public pages
  run: |
    ! grep -r "TODO\|TBD" docs/ \
      --exclude-dir=_internal \
      --include="*.md"
```

---

## SUMMARY OF RECOMMENDATIONS

### Immediate Actions (Phase 1 - No Breaking Changes)
1. ✅ **Delete 50+ redundant files** (_internal/api-duplicate/, deprecated GUI docs, exact duplicates)
2. ✅ **Merge 8 duplicate content pairs** (keep best version, merge unique content)
3. ✅ **Add 3 missing API pages to nav** (preprocessing, chemometrics, metrics)
4. ✅ **Fix 8 public pages with placeholders** (complete code examples, remove "...")
5. ✅ **Integrate orphaned stats/ folder** (add to nav or merge into methods/)
6. ✅ **Move 3 singleton orphaned files** (metrics, instrument_file_formats, plotting)

**Estimated Time:** 4-6 hours  
**Impact:** Clean up 25% of docs without breaking any links

### Strategic Actions (Phase 2 - Requires Planning)
1. 🔄 **Rename numbered folders** (01-getting-started → getting-started)
2. 🔄 **Reorganize tutorials by level** (beginner/intermediate/advanced subdirs)
3. 🔄 **Reorganize workflows by domain** (authentication/quality/quantification subdirs)
4. 🔄 **Consolidate methods** (preprocessing, ml, validation, stats → methods/)
5. 🔄 **Update all internal links** (use search-replace or script)
6. 🔄 **Create redirect map** (preserve old URLs for external links)
7. 🔄 **Rewrite mkdocs.yml** (new flat structure with clear paths)

**Estimated Time:** 12-16 hours  
**Impact:** Professional IA, easier navigation, better discoverability

### Path Optimization (Phase 3 - Content Enhancement)
1. ✍️ **Create path landing pages** (beginner/practitioner/developer entry points)
2. ✍️ **Add path signposts** ("Next Steps", "Prerequisites", "Related" sections)
3. ✍️ **Write missing guides** (understanding-results.md, community.md)

**Estimated Time:** 6-8 hours  
**Impact:** User-centric documentation, lower time-to-productivity

---

## DECISION POINTS FOR USER

**Before Phase 1:**
1. Confirm deletion of GUI tutorials (no GUI exists)?
2. Keep stats/ as separate section or merge into methods/?
3. Handle _internal/ how? (exclude from build, separate repo, or publish marked as internal?)

**Before Phase 2:**
1. Approve flat folder names vs numbered (getting-started vs 01-getting-started)?
2. Commit to subdirectory organization (tutorials/beginner/ vs tutorials/level1-...)?
3. Timeline for breaking changes (deploy with redirects)?

**Before Phase 3:**
1. Which paths to optimize first (beginner/practitioner/developer priority)?
2. Create new content or just reorganize existing?

---

**This analysis is complete and ready for review. No files have been modified.**
