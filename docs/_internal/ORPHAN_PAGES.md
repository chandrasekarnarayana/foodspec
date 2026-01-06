# Orphaned Pages Report

**Generated:** 2026-01-06  
**Source:** `mkdocs build --strict` output  
**Total Orphaned Pages:** 163 across 21 folders  
**Status:** Not in mkdocs.yml nav configuration

---

## Summary by Category

| Category | Count | Priority | Action |
|----------|-------|----------|--------|
| **Duplicate Folders** (perfect matches) | 25 | 🔴 Critical | MERGE into canonical folder, add redirects |
| **Duplicate Folders** (with variants) | 10 | 🔴 Critical | MERGE with deduplication, add redirects |
| **Internal/Archive** | 41 | 🟡 Medium | ARCHIVE (exclude from nav) |
| **Valid Content** (active folders) | 82 | 🟢 High | KEEP + add to nav |

---

## Detailed Breakdown by Folder

### 1. 📦 Duplicate Folders (High Priority)

#### **08-api/** (11 pages) → **MERGE into api/**
- **Status:** Perfect duplicate of `docs/api/`
- **Pages:** chemometrics.md, core.md, datasets.md, features.md, index.md, io.md, metrics.md, ml.md, preprocessing.md, stats.md, workflows.md
- **Recommendation:** 
  - **MERGE** into `docs/api/`
  - Add redirects: `08-api/*` → `api/*`
  - Keep single canonical location
- **Next Step:** See DOCS_MIGRATION_LEDGER.md Phase 1

#### **09-reference/** (11 pages) → **MERGE into reference/**
- **Status:** Perfect duplicate of `docs/reference/`
- **Pages:** changelog.md, citing.md, data_format.md, glossary.md, index.md, keyword_index.md, method_comparison.md, metric_significance_tables.md, metrics_reference.md, ml_model_vip_scores.md, versioning.md
- **Recommendation:**
  - **MERGE** into `docs/reference/`
  - Add redirects: `09-reference/*` → `reference/*`
  - Keep single canonical location
- **Next Step:** See DOCS_MIGRATION_LEDGER.md Phase 1

#### **10-help/** (3 pages) → **MERGE into help/** OR move to **troubleshooting/**
- **Status:** Duplicate of `docs/help/` + `docs/troubleshooting/`
- **Pages:** faq.md, index.md, troubleshooting.md
- **Recommendation:**
  - Check content differences (might have variant versions)
  - Consolidate with `docs/help/` (preferred) or `docs/troubleshooting/`
  - Add redirects: `10-help/*` → `help/*`
- **Next Step:** See DOCS_MIGRATION_LEDGER.md (help/troubleshooting triple)

#### **stats/** (8 pages) → **MERGE into methods/statistics/**
- **Status:** Perfect duplicate of `docs/methods/statistics/`
- **Pages:** anova_and_manova.md, correlation_and_mapping.md, hypothesis_testing_in_food_spectroscopy.md, introduction_to_statistical_analysis.md, nonparametric_methods_and_robustness.md, overview.md, study_design_and_data_requirements.md, t_tests_effect_sizes_and_power.md
- **Recommendation:**
  - **MERGE** into `docs/methods/statistics/`
  - Add redirects: `stats/*` → `methods/statistics/*`
  - Keep single canonical location
- **Next Step:** See DOCS_MIGRATION_LEDGER.md Phase 1

---

### 2. 🏗️ Numbered Prefix Folders (Organization Artifact)

#### **05-advanced-topics/** (12 pages)
- **Status:** Advanced/specialized content not in main nav
- **Pages:** MOATS_IMPLEMENTATION.md, advanced_deep_learning.md, architecture.md, deployment_artifact_versioning.md, deployment_hdf5_schema_versioning.md, design_overview.md, hsi_and_harmonization.md, index.md, model_lifecycle.md, model_registry.md, multimodal_workflows.md, validation_strategies.md
- **Recommendation:**
  - **KEEP** - valuable specialized content
  - **ADD to nav** under new "Advanced Topics" section (or merge into Theory/Methods)
  - Option A: Add entire section to mkdocs.yml
  - Option B: Merge pages into existing sections (theory, workflows, methods)
  - Suggested location: `nav > Theory > Advanced Topics` or separate top-level section
- **Next Step:** Prioritize for Phase 2 content consolidation

---

### 3. 📚 Valid Orphaned Content (Should Be in Nav)

These folders exist and have good content, but are not in mkdocs.yml navigation.

#### **developer-guide/** (11 pages) → **KEEP + ADD to nav**
- **Pages:** RELEASE_CHECKLIST.md, contributing.md, documentation_guidelines.md, documentation_maintainer_guide.md, documentation_style_guide.md, extending_protocols_and_steps.md, index.md, releasing.md, testing_and_ci.md, testing_coverage.md, writing_plugins.md
- **Recommendation:**
  - **KEEP** - already exists with good structure
  - **ADD to mkdocs.yml** under Developer Guide section
  - Verify index.md exists and is structured as section guide
- **Priority:** 🟢 High - move to nav immediately

#### **tutorials/** (11 pages) → **KEEP + ADD to nav**
- **Pages:** Beginner (01-load-and-plot.md, 02-preprocess.md, 03-classify.md), Intermediate (01-oil-authentication.md, 02-matrix-effects.md, 03-stability.md), Advanced (01-reproducible-pipelines.md, 02-reference-workflow.md, 03-hsi-mapping.md), end-to-end-notebooks.md, index.md
- **Recommendation:**
  - **KEEP** - well-organized by skill level
  - **ADD to mkdocs.yml** as "Tutorials" section with subsections: Beginner, Intermediate, Advanced
  - Link to example notebooks (in examples/ folder)
- **Priority:** 🟢 High - move to nav immediately

#### **user-guide/** (16 pages) → **KEEP + ADD to nav**
- **Pages:** automation.md, cli.md, cli_help.md, config_logging.md, csv_to_library.md, data_formats_and_hdf5.md, data_governance.md, index.md, libraries.md, library_search.md, protocol_profiles.md, protocols_and_yaml.md, registry_and_plugins.md, vendor_formats.md, vendor_io.md, visualization.md
- **Recommendation:**
  - **KEEP** - comprehensive user documentation
  - **ADD to mkdocs.yml** as "User Guide" section with subsections
  - Current nav appears to have many of these already - check for duplicates in nav structure
- **Priority:** 🟢 High - verify nav mapping, add any missing

#### **workflows/** (7 pages) → **KEEP + ADD to nav**
- **Pages:** aging_workflows.md, batch_quality_control.md, domain_templates.md, harmonization_automated_calibration.md, heating_quality_monitoring.md, mixture_analysis.md, standard_templates.md
- **Recommendation:**
  - **KEEP** - domain-specific workflows
  - **ADD to mkdocs.yml** under Workflows section (likely already partially there - verify)
  - Check if root-level pages duplicate subfolders
- **Priority:** 🟢 High - verify nav mapping, add any missing

#### **protocols/** (7 pages) → **KEEP + ADD to nav**
- **Pages:** benchmarking_framework.md, decision_trees.md, methods_text_generator.md, protocol_benchmarks.md, protocols_overview.md, reference_protocol.md, statistical_power_and_limits.md
- **Recommendation:**
  - **KEEP** - specialized protocol documentation
  - **ADD to mkdocs.yml** under Methods or Protocols section
  - Verify if already partially in nav
- **Priority:** 🟢 High - move to nav

#### **getting-started/** (3 pages) → **VERIFY in nav**
- **Pages:** quickstart_cli.md, quickstart_protocol.md, quickstart_python.md
- **Recommendation:**
  - **KEEP** - essential entry point content
  - **Likely already in nav** - verify and ensure no missing quickstart pages
  - If missing, add to Getting Started section
- **Priority:** 🟢 High - critical for JOSS reviewers

#### **foundations/** (4 pages) → **KEEP + ADD to nav**
- **Pages:** data_structures_and_fair_principles.md, food_spectroscopy_applications.md, index.md, spectroscopy_basics.md
- **Recommendation:**
  - **KEEP** - foundational theory content
  - **ADD to mkdocs.yml** under Theory or Foundations section
  - Verify not duplicated in other locations
- **Priority:** 🟢 Medium - add to nav

#### **theory/** (1 page) → **KEEP + ADD to nav**
- **Pages:** index.md
- **Recommendation:**
  - **KEEP** - section index
  - **ADD to mkdocs.yml** if Theory section exists
  - Verify all theory/* pages are discoverable
- **Priority:** 🟢 Medium - add if missing

#### **methods/** (3 pages) → **KEEP + ADD to nav**
- **Pages:** index.md, statistics/overview.md, validation/index.md
- **Recommendation:**
  - **KEEP** - method documentation structure
  - **Likely partially in nav** - verify and add any missing subsections
  - Check for duplicates with stats/ and reference/ folders
- **Priority:** 🟢 High - verify nav mapping

#### **help/** (4 pages) → **KEEP + VERIFY in nav**
- **Pages:** faq.md, how_to_cite.md, index.md, troubleshooting.md
- **Recommendation:**
  - **KEEP** - user support content
  - **Likely already in nav** - verify all pages included
  - Consolidate with 10-help/ (see above)
- **Priority:** 🟢 High - critical for users

#### **reference/** (6 pages) → **KEEP + VERIFY in nav**
- **Pages:** index.md, keyword_index.md, method_comparison.md, metric_significance_tables.md, metrics_reference.md, ml_model_vip_scores.md
- **Recommendation:**
  - **KEEP** - reference material
  - **MERGE with 09-reference/** (they're duplicates)
  - Consolidate to single reference/ location in nav
- **Priority:** 🔴 Critical - resolve 09-reference duplication

---

### 4. 🔬 Miscellaneous Single Pages

#### **concepts/why_foodspec.md** → **KEEP + ADD to nav**
- **Recommendation:** Keep; add to "Why FoodSpec?" intro or Foundations section
- **Priority:** 🟢 Medium

#### **datasets/dataset_design.md** → **KEEP + ADD to nav**
- **Recommendation:** Keep; add under User Guide or Datasets section
- **Priority:** 🟢 Medium

#### **design/01_overview.md** → **ARCHIVE or MERGE**
- **Recommendation:** Check if content is valuable. If yes, merge into Theory or Architecture documentation. Otherwise archive.
- **Priority:** 🟡 Medium

#### **(root) non_goals_and_limitations.md** → **KEEP + ADD to nav**
- **Recommendation:** Keep; add to Getting Started (Non-Goals & Limitations section) or index page
- **Priority:** 🟢 High - important for setting expectations

---

### 5. 🗂️ Internal/Archive Content (EXCLUDE from nav)

#### **_internal/archive/** (30+ pages)
- **Status:** Historical/deprecated documentation
- **Pages:** Various audit reports, migration guides, smoke tests, historical logs
- **Recommendation:**
  - **ARCHIVE** - do NOT add to nav
  - Already in `_internal/` (internal-only folder)
  - Exclude from mkdocs nav via mkdocs.yml filter
  - Verify `.gitignore` treats appropriately (if not production)
- **Next Step:** Ensure mkdocs.yml excludes _internal/ folder from nav

#### **_internal/developer-guide/** (6 pages)
- **Status:** Duplicate of `docs/developer-guide/` in public folder
- **Recommendation:**
  - **ARCHIVE** - use public folder only
  - Remove from _internal/ or consolidate duplicates
  - **MERGE** pages into `docs/developer-guide/`
- **Next Step:** Audit content, consolidate, delete redundant copies

#### **_internal/dev/** (3 pages)
- **Status:** Developer notes and test results
- **Recommendation:**
  - **ARCHIVE** - internal-only, do not include in nav
  - Keep as reference but exclude from public docs
- **Next Step:** Verify not needed in public docs

#### **_internal/DOCS_MIGRATION_LEDGER.md**
- **Status:** This audit document itself
- **Recommendation:**
  - **KEEP** in _internal/ (planning document, not user-facing)
  - **ARCHIVE** from nav
- **Next Step:** Already internal, exclude from nav

---

## Action Plan by Priority

### 🔴 **CRITICAL - Phase 1: Consolidate Duplicates** (2-3 days)
**Affects:** 25 pages (must complete for CI/CD to pass)

1. ✅ **08-api/** + **api/** → Keep `api/`, delete `08-api/`
   - Add mkdocs-redirects config
   - Test redirect paths
   
2. ✅ **09-reference/** + **reference/** → Keep `reference/`, delete `09-reference/`
   - Add mkdocs-redirects config
   - Test redirect paths

3. ✅ **stats/** + **methods/statistics/** → Keep `methods/statistics/`, delete `stats/`
   - Add mkdocs-redirects config
   - Test redirect paths

4. ✅ **10-help/** + **help/** + **troubleshooting/** → Consolidate (decide canonical)
   - Compare content
   - Merge into one
   - Add mkdocs-redirects for other two

### 🟢 **HIGH - Phase 2: Add Valid Content to Nav** (1-2 days)
**Affects:** 82 pages (improves discoverability)

1. Add **tutorials/** section
2. Add **protocols/** section
3. Verify **user-guide/**, **workflows/**, **methods/** are complete
4. Verify **foundations/**, **getting-started/**, **help/** are complete
5. Add **05-advanced-topics/** section
6. Add **developer-guide/** section (if missing)

### 🟡 **MEDIUM - Phase 3: Review & Archive** (1 day)
**Affects:** 41 pages (cleanup internal docs)

1. Audit **_internal/** content
2. Configure mkdocs.yml to exclude _internal/ from nav
3. Archive or delete obsolete audit reports

---

## How to Implement

### Step 1: Configure mkdocs.yml to Exclude _internal/
```yaml
plugins:
  - search
  - exclude:
      glob:
        - _internal/**
```

### Step 2: Execute Phase 1 (See DOCS_MIGRATION_LEDGER.md)
- Merge duplicate folders
- Add mkdocs-redirects entries
- Test with `mkdocs build --strict`

### Step 3: Execute Phase 2
- Edit mkdocs.yml to add new nav sections
- Verify all orphaned content is discoverable
- Test with `mkdocs build --strict`

### Step 4: Verify
```bash
mkdocs build --strict  # Should show 0 orphaned pages
python scripts/validate_docs.py --full
```

---

## Statistics

| Metric | Count |
|--------|-------|
| **Total Orphaned Pages** | 163 |
| **Duplicate Folder Pairs** | 4 (25 pages) |
| **Duplicate/Variant Folders** | 1 (10 pages) |
| **Valid Content to Add to Nav** | 82 pages |
| **Internal/Archive (exclude)** | 41 pages |
| **Single Misc Pages** | 5 pages |
| **Folders with Orphans** | 21 |
| **Folders Fully Orphaned** | 7 (05-advanced-topics, concepts, datasets, design, foundations, theory, 1 root) |
| **Folders Partially Orphaned** | 14 (partial pages missing from nav) |

---

## Related Documents

- DOCS_MIGRATION_LEDGER.md (in this folder) - Detailed migration plan with batch execution
- JOSS_DOCS_AUDIT_REPORT.md (repository root) - Full documentation assessment
- mkdocs.yml (repository root) - Navigation configuration (to be updated)

---

**Next Action:** Review categorization above. Execute Phase 1 consolidation using mkdocs-redirects plugin to prevent broken links for JOSS reviewers.
