# Documentation Migration Ledger

**Purpose:** Track the migration of duplicated/orphaned documentation pages to a canonical structure that aligns with scikit-learn standard.  
**Date Initiated:** January 6, 2026  
**Scope:** FoodSpec Repository  

---

## Rules of the Migration

1. **No broken links allowed after each batch**
   - Every page removal must have a redirect in mkdocs.yml
   - All internal cross-references must be updated
   - Run `mkdocs build --strict` before committing each batch

2. **Every moved page must have a redirect**
   - Old path → New path redirects defined in mkdocs.yml
   - Example: `stats/: methods/statistics/`
   - Redirects maintained for at least 2 releases

3. **Duplicate pages must be merged into one canonical page**
   - For identical files in two locations: pick canonical, delete other, add redirect
   - For similar but different files: merge content, keep canonical, delete duplicate
   - Document merge decisions and preserve content from all versions

4. **Archived pages must be removed from mkdocs.yml nav**
   - Pages moved to `docs/_internal/archive/` removed from navigation
   - These pages remain in repo for historical reference
   - Cannot be indexed by search

5. **API docs must not contain tutorials; examples go to examples gallery**
   - API reference pages: only auto-generated docstrings + signatures (no long tutorials)
   - Runnable examples: move to `/docs/examples/` with narrative + code + figures
   - Workflow tutorials: keep in `/docs/workflows/` with workflow focus (not API focus)

---

## Migration Ledger: Duplicate Folder Pairs

### Pair 1: `/stats/` ↔ `/methods/statistics/`

**Status:** ✅ COMPLETED (merged & deleted docs/stats/)  
**Completion Date:** 2026-01-06  
**Priority:** HIGH  
**Complexity:** LOW (perfect duplicates, identical files)

| Old Path | New Path | Action | Reason | Redirect? | Status |
|----------|----------|--------|--------|-----------|--------|
| `docs/stats/anova_and_manova.md` | `docs/methods/statistics/anova_and_manova.md` | DELETE | Identical file exists in canonical location; merged with enhanced metadata | YES | ✅ COMPLETED |
| `docs/stats/correlation_and_mapping.md` | `docs/methods/statistics/correlation_and_mapping.md` | DELETE | Identical file exists in canonical location; merged with enhanced metadata | YES | ✅ COMPLETED |
| `docs/stats/hypothesis_testing_in_food_spectroscopy.md` | `docs/methods/statistics/hypothesis_testing_in_food_spectroscopy.md` | DELETE | Identical file exists in canonical location; merged with enhanced metadata | YES | ✅ COMPLETED |
| `docs/stats/introduction_to_statistical_analysis.md` | `docs/methods/statistics/introduction_to_statistical_analysis.md` | DELETE | Identical file exists in canonical location; merged with enhanced metadata | YES | ✅ COMPLETED |
| `docs/stats/nonparametric_methods_and_robustness.md` | `docs/methods/statistics/nonparametric_methods_and_robustness.md` | DELETE | Identical file exists in canonical location; merged with enhanced metadata | YES | ✅ COMPLETED |
| `docs/stats/overview.md` | `docs/methods/statistics/overview.md` | DELETE | Identical file exists in canonical location; merged with enhanced metadata | YES | ✅ COMPLETED |
| `docs/stats/study_design_and_data_requirements.md` | `docs/methods/statistics/study_design_and_data_requirements.md` | DELETE | Identical file exists in canonical location; merged with enhanced metadata | YES | ✅ COMPLETED |
| `docs/stats/t_tests_effect_sizes_and_power.md` | `docs/methods/statistics/t_tests_effect_sizes_and_power.md` | DELETE | Identical file exists in canonical location; merged with enhanced metadata | YES | ✅ COMPLETED |
| `docs/stats/` (folder) | — | DELETE | Entire folder deprecated; canonical is `/methods/statistics/` | YES | ✅ COMPLETED |

**Canonical Choice:** `docs/methods/statistics/` (reason: hierarchical under Methods, matches mkdocs.yml nav structure)

**Execution Summary:**
1. ✅ Verified all 8 files are identical with Path differences
2. ✅ Enhanced canonical files with metadata headers and Next Steps sections from stats/ versions
3. ✅ Added 8 redirect rules to mkdocs.yml: `stats/*` → `methods/statistics/*`
4. ✅ Deleted `docs/stats/` folder
5. ✅ Fixed 3 docstring links in src/foodspec/stats/hypothesis_tests.py
6. ✅ Fixed references in docs/_internal/archive/stats_tests.md
7. ✅ Ran mkdocs build --strict: PASSED
8. ✅ Ran link checker: 0 broken links (227 files checked)

**Result:** All users accessing old stats/ URLs automatically redirected to methods/statistics/ via mkdocs-redirects plugin. No broken links. Single source of truth maintained.

**Risk:** None (all redirects tested; links verified)

---

### Pair 2: `/api/` ↔ `/08-api/`

**Status:** ✅ COMPLETED (deleted docs/08-api/ with redirects)  
**Completion Date:** 2026-01-06  
**Priority:** HIGH  
**Complexity:** LOW (perfect duplicates, identical structure)

| Old Path | New Path | Action | Reason | Redirect? | Status |
|----------|----------|--------|--------|-----------|--------|
| `docs/08-api/index.md` | `docs/api/index.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/core.md` | `docs/api/core.md` | DELETE | Identical file exists in canonical location; api/core.md has better structure | YES | ✅ COMPLETED |
| `docs/08-api/datasets.md` | `docs/api/datasets.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/features.md` | `docs/api/features.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/io.md` | `docs/api/io.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/metrics.md` | `docs/api/metrics.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/ml.md` | `docs/api/ml.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/preprocessing.md` | `docs/api/preprocessing.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/stats.md` | `docs/api/stats.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/workflows.md` | `docs/api/workflows.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/chemometrics.md` | `docs/api/chemometrics.md` | DELETE | Identical file exists in canonical location | YES | ✅ COMPLETED |
| `docs/08-api/` (folder) | — | DELETE | Entire folder deprecated; canonical is `/api/` | YES | ✅ COMPLETED |

**Canonical Choice:** `docs/api/` (reason: shorter path, already in active nav, matches mkdocs.yml structure)

**Execution Summary:**
1. ✅ Compared all 11 files: 10 identical, 1 differs (core.md - api/ version is better with Overview section)
2. ✅ Decided to keep api/core.md (it has better metadata and Overview section)
3. ✅ Added 11 redirect rules to mkdocs.yml: `08-api/*` → `api/*` (including index.md)
4. ✅ Deleted `docs/08-api/` folder via `git rm -r`
5. ✅ Fixed 11 broken references in active documentation:
   - src/foodspec/preprocess/baseline.py (1 docstring link)
   - docs/theory/data_structures_and_fair_principles.md (1 link)
   - docs/getting-started/quickstart_python.md (2 links)
   - docs/reference/data_format.md (2 links)
   - docs/reference/keyword_index.md (1 link)
   - docs/workflows/index.md (1 link)
   - docs/workflows/workflow_design_and_reporting.md (1 link)
   - docs/10-help/faq.md (1 link)
   - docs/help/faq.md (1 link)
6. ✅ Updated 11 archive files (docs/_internal/archive/*.md) to reference api/ instead of 08-api/
7. ✅ Ran mkdocs build --strict: PASSED (only pre-existing vendor_io anchor warning unrelated to consolidation)
8. ✅ Ran link checker: 0 broken links (216 markdown files checked)

**Git Commits:**
- Commit a6a56af: "docs: consolidate 08-api/ into api/ (canonical) with redirects" (12 files changed, 2,500 deletions)
- Commit e92a406: "docs: fix all 08-api references to api/ in active documentation" (11 files changed, 13 edits)
- Commit a5f9c80: "docs: update archive file references to use api/ (canonical path)" (11 files changed, 18 edits)

**Result:** All users accessing old 08-api/ URLs automatically redirected to api/ via mkdocs-redirects plugin. No broken links. Single source of truth maintained.

**Risk:** None (all redirects tested; links verified; build passes)

---

### Pair 3: `/reference/` ↔ `/09-reference/`

**Status:** IDENTIFY & PLAN (not started)  
**Priority:** HIGH  
**Complexity:** LOW (perfect duplicates, identical files)

| Old Path | New Path | Action | Reason | Redirect? | Status |
|----------|----------|--------|--------|-----------|--------|
| `docs/reference/changelog.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/citing.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/data_format.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/glossary.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/index.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/keyword_index.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/method_comparison.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/metric_significance_tables.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/metrics_reference.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/ml_model_vip_scores.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/reference/versioning.md` | (KEEP) | KEEP | Canonical location | NO | PLANNED |
| `docs/09-reference/` (folder) | — | DELETE | Entire folder deprecated; canonical is `/reference/` | YES | PLANNED |

**Canonical Choice:** `docs/reference/` (reason: already in active nav, shorter path, cleaner URL)

**Migration Steps:**
1. Verify all 11 files are identical (byte-for-byte)
2. Add redirects to mkdocs.yml: `09-reference/: reference/`
3. Delete `docs/09-reference/` folder
4. Run `mkdocs build --strict` to verify
5. Update git: `git rm -r docs/09-reference/`

**Risk:** None (files are duplicates; migration is straightforward)

---

### Triple: `/help/` ↔ `/troubleshooting/` ↔ `/10-help/`

**Status:** IDENTIFY & PLAN (not started)  
**Priority:** HIGH  
**Complexity:** MEDIUM (different content; need to merge carefully)

#### File Inventory

**In `/help/`:**
- `help/faq.md` — FAQ content
- `help/how_to_cite.md` — Citing instructions
- `help/index.md` — Landing page
- `help/troubleshooting.md` — General troubleshooting

**In `/troubleshooting/`:**
- `troubleshooting/common_problems_and_solutions.md` — Problem-solution pairs
- `troubleshooting/reporting_guidelines.md` — How to report issues
- `troubleshooting/troubleshooting_faq.md` — Troubleshooting FAQ

**In `/10-help/`:**
- `10-help/faq.md` — FAQ content (similar to `/help/faq.md`)
- `10-help/index.md` — Landing page (similar to `/help/index.md`)
- `10-help/troubleshooting.md` — Troubleshooting (similar to `/help/troubleshooting.md`)

#### Proposed Merge Strategy

| Old Path | Action | Target Location | Merge Strategy | Redirect? | Status |
|----------|--------|-----------------|-----------------|-----------|--------|
| `docs/help/faq.md` | KEEP | `docs/help/faq.md` | Canonical | NO | PLANNED |
| `docs/help/how_to_cite.md` | KEEP | `docs/help/how_to_cite.md` | Canonical | NO | PLANNED |
| `docs/help/index.md` | KEEP | `docs/help/index.md` | Canonical (update links) | NO | PLANNED |
| `docs/help/troubleshooting.md` | MERGE | `docs/help/troubleshooting.md` | Merge with content from `/troubleshooting/` pages | NO | PLANNED |
| `docs/troubleshooting/common_problems_and_solutions.md` | MERGE | `docs/help/troubleshooting.md` | Merge into canonical troubleshooting page (or keep separate as subsection) | YES | PLANNED |
| `docs/troubleshooting/reporting_guidelines.md` | MOVE | `docs/help/reporting_issues.md` | Move and rename to clarify intent | YES | PLANNED |
| `docs/troubleshooting/troubleshooting_faq.md` | MERGE | `docs/help/faq.md` | Merge into FAQ (or fold into troubleshooting) | YES | PLANNED |
| `docs/10-help/` (folder) | DELETE | — | Duplicate of `/help/`; deprecate | YES | PLANNED |
| `docs/troubleshooting/` (folder) | ARCHIVE or DELETE | `docs/_internal/archive/troubleshooting-backup/` | Archive as backup; content merged into `/help/` | YES | PLANNED |

**Canonical Choice:** `docs/help/` (reason: already in active nav, cleaner structure, consolidates all help topics)

**Migration Steps:**
1. Audit content differences between the three folders
2. Merge content:
   - `/help/faq.md` + `/10-help/faq.md` → keep in `/help/faq.md`
   - `/help/troubleshooting.md` + `/troubleshooting/troubleshooting_faq.md` → merge into single FAQ or troubleshooting page
   - `/troubleshooting/common_problems_and_solutions.md` → add to `/help/troubleshooting.md` or create subsection
   - `/troubleshooting/reporting_guidelines.md` → move to `/help/reporting_issues.md`
3. Update `/help/index.md` to reference all consolidated pages
4. Add redirects to mkdocs.yml:
   - `10-help/: help/`
   - `troubleshooting/: help/`
5. Delete `docs/10-help/` and `docs/troubleshooting/` folders (or archive)
6. Run `mkdocs build --strict` to verify

**Risk:** Medium (need careful content merge to avoid losing information)

**Acceptance Criteria:**
- All help topics discoverable from single landing page
- No orphaned pages
- Cross-references between help pages work correctly

---

### Workflows Duplicates

**Status:** IDENTIFY & PLAN (not started)  
**Priority:** MEDIUM  
**Complexity:** MEDIUM (complex hierarchy; some deliberate structure)

#### File Inventory & Duplicates Found

| Root Level (Deprecated) | Subfolder (Canonical) | Status |
|-------------------------|----------------------|--------|
| `docs/workflows/aging_workflows.md` | `docs/workflows/quality-monitoring/aging_workflows.md` | DUPLICATE |
| `docs/workflows/batch_quality_control.md` | `docs/workflows/quality-monitoring/batch_quality_control.md` | DUPLICATE |
| `docs/workflows/heating_quality_monitoring.md` | `docs/workflows/quality-monitoring/heating_quality_monitoring.md` | DUPLICATE |
| `docs/workflows/domain_templates.md` | `docs/workflows/authentication/domain_templates.md` | DUPLICATE |
| `docs/workflows/harmonization_automated_calibration.md` | `docs/workflows/harmonization/harmonization_automated_calibration.md` | DUPLICATE |
| `docs/workflows/mixture_analysis.md` | `docs/workflows/quantification/mixture_analysis.md` | DUPLICATE |
| `docs/workflows/standard_templates.md` | `docs/workflows/harmonization/standard_templates.md` | DUPLICATE |

#### Migration Table

| Old Path | New Path | Action | Reason | Redirect? | Status |
|----------|----------|--------|--------|-----------|--------|
| `docs/workflows/aging_workflows.md` | `docs/workflows/quality-monitoring/aging_workflows.md` | DELETE | Duplicate exists in organized subfolder | YES | PLANNED |
| `docs/workflows/batch_quality_control.md` | `docs/workflows/quality-monitoring/batch_quality_control.md` | DELETE | Duplicate exists in organized subfolder | YES | PLANNED |
| `docs/workflows/heating_quality_monitoring.md` | `docs/workflows/quality-monitoring/heating_quality_monitoring.md` | DELETE | Duplicate exists in organized subfolder | YES | PLANNED |
| `docs/workflows/domain_templates.md` | `docs/workflows/authentication/domain_templates.md` | DELETE | Duplicate exists in organized subfolder | YES | PLANNED |
| `docs/workflows/harmonization_automated_calibration.md` | `docs/workflows/harmonization/harmonization_automated_calibration.md` | DELETE | Duplicate exists in organized subfolder | YES | PLANNED |
| `docs/workflows/mixture_analysis.md` | `docs/workflows/quantification/mixture_analysis.md` | DELETE | Duplicate exists in organized subfolder | YES | PLANNED |
| `docs/workflows/standard_templates.md` | `docs/workflows/harmonization/standard_templates.md` | DELETE | Duplicate exists in organized subfolder | YES | PLANNED |
| `docs/workflows/workflow_design_and_reporting.md` | (KEEP) | KEEP | Unique page; conceptual overview not domain-specific | NO | PLANNED |
| `docs/workflows/end_to_end_pipeline.md` | (KEEP) | KEEP | Unique page; high-level overview | NO | PLANNED |
| `docs/workflows/index.md` | (UPDATE) | UPDATE | Update to reference subfolders cleanly; remove duplicated links | NO | PLANNED |

**Canonical Choice:** Subfolders under `/workflows/` (reason: organized by domain, already used in nav, reflects workflow intent)

**Migration Steps:**
1. Verify each root-level file is identical to subfolder version
2. Add redirects to mkdocs.yml:
   - `workflows/aging_workflows.md: workflows/quality-monitoring/aging_workflows.md`
   - `workflows/batch_quality_control.md: workflows/quality-monitoring/batch_quality_control.md`
   - ... (etc. for all 7 duplicates)
3. Delete root-level duplicated files (keep only if content differs)
4. Update `docs/workflows/index.md` to reference subfolders
5. Run `mkdocs build --strict` to verify

**Risk:** Low (files are organized; migration is straightforward)

---

## Other Orphaned Folders (Not Duplicates, but Untracked)

These folders exist in `/docs/` but are NOT in `mkdocs.yml` nav. Decision: **archive or integrate into active nav**.

| Folder | Files | In Nav? | Recommendation | Action | Status |
|--------|-------|---------|-----------------|--------|--------|
| `docs/05-advanced-topics/` | 12 | ❌ NO | Integrate high-value pages; archive rest | SELECTIVE ARCHIVE | PLANNED |
| `docs/tutorials/` | 10 | ❌ NO | Move examples to `/examples/` gallery; archive rest | MIGRATE → EXAMPLES | PLANNED |
| `docs/concepts/` | 1 | ❌ NO | Merge into theory/ or user-guide/ | MERGE → THEORY | PLANNED |
| `docs/design/` | 1 | ❌ NO | Move to developer-guide/ or archive | MOVE → DEV-GUIDE | PLANNED |
| `docs/datasets/` | 1 | ❌ NO | Merge into user-guide/ | MERGE → UG | PLANNED |
| `docs/protocols/` | 7 | ❌ NO | Already content in user-guide/; review and consolidate | CONSOLIDATE | PLANNED |
| `docs/foundations/` | 4 | ⚠️ PARTIAL | Should be in nav under Theory | INTEGRATE → THEORY | PLANNED |

---

## Batch Execution Plan

### Batch 1 (Quick Wins - Week 1)
- **Task:** Consolidate identical folder pairs
- **Scope:** `/stats/` + `/08-api/` + `/09-reference/`
- **Files Affected:** 30 files deleted, 30 redirects added
- **Effort:** 2–3 hours
- **Risk:** Low (identical files)
- **Status:** READY TO EXECUTE
- **Acceptance:** mkdocs build --strict passes, no broken links

### Batch 2 (Merge Help Folders - Week 1)
- **Task:** Merge `/help/` + `/troubleshooting/` + `/10-help/`
- **Scope:** 10 files merged into 5
- **Effort:** 2–3 hours (content review + merge)
- **Risk:** Medium (need careful merge to preserve all info)
- **Status:** READY TO PLAN (needs content audit first)
- **Acceptance:** All help topics discoverable from single page

### Batch 3 (Workflows Duplicates - Week 2)
- **Task:** Delete root-level workflow duplicates
- **Scope:** 7 files deleted, 7 redirects added
- **Effort:** 1–2 hours
- **Risk:** Low (files are duplicates)
- **Status:** READY TO EXECUTE
- **Acceptance:** All workflows accessible via subfolders

### Batch 4 (Untracked Folders - Week 2-3)
- **Task:** Integrate or archive orphaned folders
- **Scope:** 28+ files (decision: keep, move, or archive)
- **Effort:** 3–5 hours (content review + decisions)
- **Risk:** Medium (must review content before archiving)
- **Status:** BLOCKED ON CONTENT AUDIT
- **Acceptance:** All user-facing content in active nav; internal/archived content in `_internal/`

### Batch 5 (Auto-Generate API Reference - Phase 2, Week 3-4)
- **Task:** Replace manual `/api/` pages with auto-generated docstrings
- **Scope:** 11 files replaced
- **Effort:** Depends on Phase 2 (docstring fixes)
- **Risk:** Medium (depends on docstring quality)
- **Status:** BLOCKED ON PHASE 2
- **Acceptance:** mkdocstrings generates valid API pages from all docstrings

---

## Redirect Configuration (mkdocs.yml)

Once migrations are complete, add to `mkdocs.yml` redirects section:

```yaml
plugins:
  - redirects:
      redirect_maps:
        # Pair 1: stats → methods/statistics
        stats/: methods/statistics/
        stats/anova_and_manova.md: methods/statistics/anova_and_manova.md
        stats/correlation_and_mapping.md: methods/statistics/correlation_and_mapping.md
        # ... (all 8 files)
        
        # Pair 2: 08-api → api
        08-api/: api/
        08-api/index.md: api/index.md
        08-api/core.md: api/core.md
        # ... (all 11 files)
        
        # Pair 3: 09-reference → reference
        09-reference/: reference/
        09-reference/index.md: reference/index.md
        # ... (all 11 files)
        
        # Triple: help/troubleshooting/10-help consolidation
        10-help/: help/
        troubleshooting/: help/
        troubleshooting/common_problems_and_solutions.md: help/troubleshooting.md
        troubleshooting/reporting_guidelines.md: help/reporting_issues.md
        troubleshooting/troubleshooting_faq.md: help/faq.md
        
        # Workflows duplicates
        workflows/aging_workflows.md: workflows/quality-monitoring/aging_workflows.md
        workflows/batch_quality_control.md: workflows/quality-monitoring/batch_quality_control.md
        workflows/heating_quality_monitoring.md: workflows/quality-monitoring/heating_quality_monitoring.md
        workflows/domain_templates.md: workflows/authentication/domain_templates.md
        workflows/harmonization_automated_calibration.md: workflows/harmonization/harmonization_automated_calibration.md
        workflows/mixture_analysis.md: workflows/quantification/mixture_analysis.md
        workflows/standard_templates.md: workflows/harmonization/standard_templates.md
```

---

## Sign-Off Checklist

Before committing each batch:

- [ ] Run `mkdocs build --strict` — must pass with 0 warnings (except pre-existing)
- [ ] Verify all redirects work (test 3–5 old links manually)
- [ ] Check git diffs — only deletions + redirect additions, no content changes
- [ ] Review mkdocs.yml — no syntax errors, redirects correctly formatted
- [ ] Spot-check 2–3 pages in site/build — links render correctly
- [ ] Update this ledger with "Status: EXECUTED"

---

## Summary

| Duplicate Pair | Files Affected | Action | Effort | Status |
|---|---|---|---|---|
| `/stats/` ↔ `/methods/statistics/` | 8 | DELETE | 1–2 hrs | BATCH 1 |
| `/api/` ↔ `/08-api/` | 11 | DELETE | 1–2 hrs | BATCH 1 |
| `/reference/` ↔ `/09-reference/` | 11 | DELETE | 1–2 hrs | BATCH 1 |
| `/help/` ↔ `/troubleshooting/` ↔ `/10-help/` | 10 | MERGE | 2–3 hrs | BATCH 2 |
| Workflows (root vs subfolders) | 7 | DELETE | 1–2 hrs | BATCH 3 |
| Orphaned folders | 28+ | AUDIT & DECIDE | 3–5 hrs | BATCH 4 |
| **TOTAL** | **75+** | — | **10–15 hrs** | **PLANNED** |

---

**Last Updated:** January 6, 2026  
**Next Review:** After Batch 1 execution  
**Owner:** Docs Maintainer  

