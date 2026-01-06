# FoodSpec Documentation Audit Report
## Comparison vs scikit-learn Documentation Standard

**Date:** January 6, 2026  
**Scope:** Repository structure, documentation architecture, API reference generation, examples organization, and linkage completeness.

---

## A. Executive Summary

### Strengths
- ✅ **Well-organized MkDocs setup** with Material theme and mkdocstrings integration
- ✅ **Strong test coverage (78%)** with 689 passing tests
- ✅ **Comprehensive domain content** (workflows, methods, theory sections are substantial)
- ✅ **Redirects strategy in place** for old page paths (28 redirects defined)
- ✅ **Multiple quick-start options** (CLI, Python, Protocol)
- ✅ **Working examples** (12 Python scripts, 3 Jupyter notebooks in examples/)
- ✅ **Recent refactoring** successfully moved core API and demo package to public API

### Critical Gaps (Blocking Gold-Standard Quality)
- ❌ **Severe docstring gaps** in core API: ~60% of public functions missing Parameters/Returns
  - `FoodSpectrumSet.to_wide_dataframe()`, `subset()`, `apply()`, etc. lack formal docstrings
  - Blocks auto-generation of API reference (mkdocstrings relies on docstrings)
- ❌ **Duplicated docs pages** in multiple folders:
  - `stats/` (8 pages) + `methods/statistics/` (7 pages) both exist, not in nav
  - `api/` (11 pages) + `08-api/` (11 pages) duplicates untracked
  - `reference/` (11 pages) + `09-reference/` (11 pages) duplicates untracked
- ❌ **Information architecture chaos**: 130+ untracked pages not in mkdocs.yml nav
  - `05-advanced-topics/`, `08-api/`, `09-reference/`, `10-help/` folders exist but invisible
  - Confuses users and breaks discoverability
- ❌ **Examples scattered across 3 locations** (examples/, examples/notebooks/, docs/):
  - No unified "Examples Gallery" with narrative
  - Inline code blocks in docs not linked to runnable examples
  - Figures embedded in docs without code generation tie
- ❌ **API reference not auto-generated**: manual pages in docs/api/ don't reflect current public API
  - New `foodspec.demo` module not documented
  - Import paths may be stale (e.g., deprecated aliases not marked as such)
- ❌ **Missing copy-paste safety in docs**:
  - Recent fix for `heating_quality_monitoring.md` was necessary (removed risky `from examples` import)
  - Suggests other pages may have similar issues
- ❌ **No unified glossary or "when to use" guidance** (scikit-learn tone: assumptions, failure modes, alternatives)

### Immediate Priorities (Fix First)
1. **Eliminate duplicate pages** (consolidate stats/, reference/, api/, help/ pairs)
2. **Fix docstrings** in ~25 core public functions (FoodSpectrumSet, dataset operations, metrics, IO)
3. **Implement link checker** in CI/CD to catch broken anchors and missing pages
4. **Reorganize navigation** in mkdocs.yml: remove dead branches, move untracked pages into active nav
5. **Create examples gallery** with narrative + code + figures (3–5 flagship examples)

---

## B. Current State Map

### B.1 Documentation Pages Inventory (233 files)

| Category | Count | Status | Issues |
|----------|-------|--------|--------|
| **Getting Started** | 6 | ✅ Active (nav) | Clear entry points; good quickstarts |
| **Workflows** | 17 | ⚠️ Partial | Duplicated paths; good conceptual content |
| **Methods** (Preprocessing, Chemometrics, Validation, Statistics) | 23 | ✅ Active (nav) | Conceptually strong; some lack code examples |
| **API Reference** | 22 | ❌ Broken | Duplicates in `/api/` + `/08-api/`; manual, not auto-gen; stale |
| **Theory/Foundations** | 14 | ✅ Active (nav) | Renamed from `/theory/` + `/foundations/`; good coverage |
| **User Guide** | 13 | ✅ Active (nav) | Comprehensive; clear ownership |
| **Developer Guide** | 11 | ✅ Active (nav) | Good structure; covers plugins, testing, releasing |
| **Help/Troubleshooting** | 9 | ⚠️ Duplicate | Exists in `/help/` + `/troubleshooting/` + `/10-help/` |
| **Examples** | 1 | ❌ Minimal | Only `examples_gallery.md` in nav; no narrative structure |
| **Tutorials** | 10 | ⚠️ Orphaned | Pages exist (`/tutorials/`) but not in nav; no discovery path |
| **Advanced Topics** | 12 | ❌ Hidden | `/05-advanced-topics/` not in nav; orphaned |
| **Reference** (glossary, changelog, data formats) | 11 | ⚠️ Split | `/reference/` + `/09-reference/` duplicates |
| **Internal Archive** | 60+ | ❌ Dead | `/docs/_internal/archive/` should be removed |
| **Other** (design, concepts, datasets, protocols, stats) | 16 | ⚠️ Orphaned | Exist but not discoverable |

**Navigation Status:**
- **In mkdocs.yml nav:** ~100 pages (active, discoverable)
- **Exist but not in nav:** ~130 pages (orphaned, invisible)
- **Dead/archive:** 60+ pages (should be deleted from repo)

### B.2 Public API Inventory & Docstring Status

| Module | Objects | Docstring Status | Priority |
|--------|---------|------------------|----------|
| `foodspec.core.dataset.FoodSpectrumSet` | 25 methods | ⚠️ 60% missing | HIGH |
| `foodspec.core.spectrum.Spectrum` | 8 methods | ✅ 100% | — |
| `foodspec.metrics` | 8 functions | ⚠️ 50% missing | HIGH |
| `foodspec.stats` | 10+ functions | ❌ Not audited (complex module structure) | HIGH |
| `foodspec.io` (load_*, read_*, create_library) | 6 functions | ⚠️ Partial | HIGH |
| `foodspec.core.output_bundle.OutputBundle` | Unknown | Not checked | MEDIUM |
| `foodspec.demo.synthetic_heating_dataset` | 1 function | ✅ Added (new) | — |
| `foodspec.viz.*` (plotting) | 15+ functions | ❌ Not audited | MEDIUM |
| `foodspec.qc.*` (quality checks) | 8+ functions | ❌ Not audited | MEDIUM |
| `foodspec.repro.*` (reproducibility) | 4 functions | ❌ Not audited | MEDIUM |

**Key Finding:** Only **~40% of public API** has complete Google-style docstrings (Parameters + Returns + Examples). scikit-learn standard requires **100% with extensive parameter documentation, defaults, and "when to use" guidance.**

---

## C. Gap Analysis vs scikit-learn Standard

### C.1 Navigation & Information Architecture

**Gap:**
- scikit-learn: Clear 3-tier structure (Getting Started → User Guide → API Reference), with Examples as separate gallery
- FoodSpec: Hierarchical (13 top-level nav items) but with orphaned sub-branches and duplicates

**Specific Issues:**
1. **Duplicated content silos:**
   - `/help/` vs `/troubleshooting/` vs `/10-help/` (which is canonical?)
   - `/stats/` vs `/methods/statistics/` (8 page pair duplication)
   - `/api/` vs `/08-api/` (11 page pair duplication)
   - `/reference/` vs `/09-reference/` (11 page pair duplication)

2. **Orphaned high-value pages:**
   - `/tutorials/` (10 pages, includes beginner+intermediate+advanced) not in nav
   - `/05-advanced-topics/` (12 pages, includes MOATS, HSI, model registry) not in nav
   - `/foundations/` (4 pages) renamed but not synced in nav

3. **No unified examples gallery:** 
   - `examples_gallery.md` exists in nav but is minimal markdown, not narrative
   - Real examples scattered: `/examples/`, `/examples/notebooks/`, `/docs/tutorials/`
   - No visual index ("which example for my use case?")

### C.2 User Guide Conceptual Depth

**Gap:**
- scikit-learn: "When to use / When NOT to use", assumptions, common pitfalls, recommended defaults
- FoodSpec: Methods explained well, but lighter on "guidance" layer

**Specific Issues:**
1. **Missing anti-patterns:** E.g., "Preprocessing Normalization" page explains HOW, not WHEN normalization hurts (e.g., zero-variance columns, or over-smoothing)
2. **Conflicting options not compared:** Multiple baseline correction methods exist; no decision tree or recommendation
3. **No reproducibility burden discussion:** Data governance page exists, but light on "data leakage pitfalls in your workflow"

### C.3 Examples Placement & Quality

**Gap:**
- scikit-learn: Gallery with narrative, all examples auto-executed + figures committed
- FoodSpec: Examples in multiple places; no unified narrative; unclear which figures come from code

**Specific Issues:**
1. **Inline docs code blocks not tied to examples:**
   - `workflows/quality-monitoring/heating_quality_monitoring.md` code block recently fixed to not import from examples
   - But pattern suggests other pages may have unsafe imports
2. **Notebooks in `/examples/notebooks/` not mentioned in nav or linked from docs**
3. **Figure sources unclear:**
   - `docs/assets/workflows/heating_quality_monitoring/heating_ratio_vs_time.png` now committed (good!)
   - But do all other figures have traceable code?
4. **No metadata on examples:** Which skill level? Which domain (oil auth, heating, spatial)? Prerequisites?

### C.4 API Reference Generation

**Gap:**
- scikit-learn: Auto-generated from docstrings (100% sync with code)
- FoodSpec: Manual pages in `/api/` and `/08-api/` folders; duplicated; potentially stale

**Specific Issues:**
1. **Docstring coverage too low:** ~60% of public functions missing formal docstrings
2. **Manual pages can drift:** E.g., if function signature changes, manual page not updated
3. **New modules not documented:** `foodspec.demo` (just added) likely not in manual pages
4. **Deprecated imports not marked:** Backward compatibility shims in `__init__.py` (e.g., `foodspec.artifact`, `foodspec.calibration_transfer`) marked as deprecated in code but unclear if docs reflect this

### C.5 Cross-Linking & Discoverability

**Gap:**
- scikit-learn: Cross-links between API → Examples → User Guide → Theory (tight mesh)
- FoodSpec: Looser linkage; orphaned pages hard to discover

**Specific Issues:**
1. **Broken anchors:** At least 1 known (vendor_io.md links to missing troubleshooting anchor)
2. **No "Related Pages" sections:** E.g., Heating Quality workflow doesn't link to underlying Stats tests
3. **Orphaned tutorials:** Beginner tutorials exist but no discovery path from Getting Started → Tutorials

### C.6 Code Quality in Docs

**Gap:**
- scikit-learn: All code blocks copy-paste runnable, minimal, focused on 1 concept
- FoodSpec: Recent fixes (heating_quality_quickstart.py, heating_quality_monitoring.md) now safe; but pattern suggests others need audit

**Specific Issues:**
1. **Risky imports in docs:** Fixed for one file; need audit of all docs code blocks
2. **Reproducibility:** Docs code that saves to root/ or uses relative paths can fail in different working directories

---

## D. Recommended Target Structure

### D.1 Proposed Docs Folder Structure

```
docs/
├── index.md                                    # Home page (redesigned)
├── getting-started/
│   ├── index.md
│   ├── installation.md
│   ├── quickstart_15min.md
│   ├── quickstart_cli.md
│   ├── quickstart_python.md
│   ├── first_steps_cli.md
│   ├── understanding_results.md
│   └── faq.md
├── examples/                                   # NEW: Unified examples gallery
│   ├── index.md                                # (Narrative gallery + links)
│   ├── 01_oil_authentication.md                # (Link + embed code + figure)
│   ├── 02_heating_quality_monitoring.md
│   ├── 03_hyperspectral_spatial_mapping.md
│   ├── 04_mixture_analysis.md
│   ├── 05_classification_workflow.md
│   └── _code/                                  # (Runnable scripts)
│       ├── ex01_oil_auth.py
│       ├── ex02_heating.py
│       ├── ex03_hsi.py
│       └── ex04_mixture.py
├── user-guide/
│   ├── index.md
│   ├── core_concepts.md                        # NEW: unified overview
│   ├── data_formats_and_hdf5.md
│   ├── libraries.md
│   ├── library_search.md
│   ├── csv_to_library.md
│   ├── vendor_formats.md
│   ├── vendor_io.md
│   ├── cli.md
│   ├── cli_help.md
│   ├── protocols_and_yaml.md
│   ├── protocol_profiles.md
│   ├── automation.md
│   ├── config_logging.md
│   ├── registry_and_plugins.md
│   ├── visualization.md
│   └── data_governance.md
├── methods/                                    # Keep, but clean
│   ├── index.md
│   ├── preprocessing/
│   ├── chemometrics/
│   ├── validation/
│   ├── statistics/                             # (CONSOLIDATE: remove /stats/)
│   └── workflows/                              # (Move quality/auth/spatial here)
├── theory/                                     # Keep (renamed from foundations)
│   ├── index.md
│   ├── spectroscopy_basics.md
│   ├── food_applications.md
│   ├── chemometrics_and_ml.md
│   ├── rq_engine.md
│   ├── harmonization.md
│   ├── moats_overview.md
│   ├── data_structures_and_fair.md
│   └── when_to_use_guide.md                    # NEW: consolidated "when not to use"
├── api/                                        # REBUILD: auto-gen from docstrings
│   ├── index.md
│   ├── core.md                                 # ::: foodspec.core
│   ├── datasets.md
│   ├── io.md
│   ├── preprocessing.md
│   ├── chemometrics.md
│   ├── features.md
│   ├── metrics.md
│   ├── stats.md
│   ├── visualization.md
│   ├── qc.md                                   # NEW: data quality checks
│   ├── workflows.md
│   ├── ml.md
│   ├── reproducibility.md                      # NEW
│   ├── demo.md                                 # NEW: demo datasets
│   └── deploy.md
├── reference/                                  # CONSOLIDATE: merge /09-reference/
│   ├── index.md
│   ├── data_formats.md
│   ├── glossary.md
│   ├── keyword_index.md
│   ├── changelog.md
│   ├── versioning.md
│   ├── citing.md
│   └── metrics_reference.md
├── help/                                       # CONSOLIDATE: merge /troubleshooting/ + /10-help/
│   ├── index.md
│   ├── troubleshooting_faq.md
│   ├── common_problems.md
│   ├── reporting_issues.md
│   └── getting_support.md
├── developer-guide/
│   ├── index.md
│   ├── contributing.md
│   ├── testing_and_ci.md
│   ├── documentation_guidelines.md              # (Updated)
│   ├── writing_plugins.md
│   ├── extending_protocols.md
│   ├── releasing.md
│   └── style_guide.md
├── reproducibility.md                          # Keep
├── assets/
│   ├── workflows/
│   │   ├── heating_quality_monitoring/
│   │   │   └── heating_ratio_vs_time.png
│   │   └── ... (other workflow figures)
│   └── ... (other assets)
├── _internal/                                  # CLEANUP: docs internal archive only
│   └── archive/                                # (Thin out further)
└── non_goals_and_limitations.md                # Keep as is

# DELETE THE FOLLOWING (move to _internal/ or remove)
- docs/05-advanced-topics/   → CONSOLIDATE into methods/advanced/ or developer-guide/
- docs/08-api/               → DUPLICATE (keep /api/ only)
- docs/09-reference/         → DUPLICATE (keep /reference/ only)
- docs/10-help/              → DUPLICATE (keep /help/ only)
- docs/stats/                → DUPLICATE (keep /methods/statistics/ only)
- docs/help/ + docs/troubleshooting/ →  CONSOLIDATE into single /help/
- docs/concepts/             → MERGE into theory/
- docs/design/               → MOVE to _internal/
- docs/datasets/             → MERGE into user-guide/
- docs/protocols/            → CONSOLIDATE into user-guide/
- docs/tutorials/            → MOVE examples to /examples/, keep narrative gallery
```

### D.2 Proposed mkdocs.yml Navigation

```yaml
nav:
  - Home: index.md
  
  - Getting Started:
    - Overview: getting-started/index.md
    - Installation: getting-started/installation.md
    - 15-Min Quickstart: getting-started/quickstart_15min.md
    - CLI Quickstart: getting-started/quickstart_cli.md
    - Python Quickstart: getting-started/quickstart_python.md
    - Understanding Results: getting-started/understanding_results.md
    - FAQ: getting-started/faq.md
  
  - Examples:                                    # NEW
    - Gallery: examples/index.md
    - Oil Authentication: examples/01_oil_authentication.md
    - Heating Quality: examples/02_heating_quality.md
    - Hyperspectral Mapping: examples/03_hyperspectral_mapping.md
    - Mixture Analysis: examples/04_mixture_analysis.md
    - Classification: examples/05_classification_workflow.md
  
  - User Guide:
    - Overview: user-guide/index.md
    - Core Concepts: user-guide/core_concepts.md
    - Data Formats & HDF5: user-guide/data_formats_and_hdf5.md
    - Libraries: user-guide/libraries.md
    - Library Search: user-guide/library_search.md
    - Loading from CSV: user-guide/csv_to_library.md
    - Vendor Formats: user-guide/vendor_formats.md
    - Command Line: user-guide/cli.md
    - Protocols & YAML: user-guide/protocols_and_yaml.md
    - Automation: user-guide/automation.md
    - Configuration: user-guide/config_logging.md
    - Plugins & Registry: user-guide/registry_and_plugins.md
    - Visualization: user-guide/visualization.md
    - Data Governance: user-guide/data_governance.md
  
  - Methods:
    - Overview: methods/index.md
    - Preprocessing:
      - Baseline Correction: methods/preprocessing/baseline_correction.md
      - ... (others)
    - Chemometrics:
      - Models & Best Practices: methods/chemometrics/models_and_best_practices.md
      - ... (others)
    - Validation:
      - Cross-Validation: methods/validation/cross_validation_and_leakage.md
      - ... (others)
    - Statistics:
      - Introduction: methods/statistics/introduction.md
      - ... (others)
  
  - Theory:
    - Overview: theory/index.md
    - Spectroscopy Basics: theory/spectroscopy_basics.md
    - Food Applications: theory/food_applications.md
    - Chemometrics & ML: theory/chemometrics_and_ml.md
    - RQ Engine: theory/rq_engine.md
    - Harmonization: theory/harmonization.md
    - MOATS Overview: theory/moats_overview.md
    - Data Structures & FAIR: theory/data_structures_and_fair.md
    - When to Use Guide: theory/when_to_use_guide.md
  
  - API Reference:
    - Overview: api/index.md
    - Core: api/core.md
    - Datasets: api/datasets.md
    - I/O: api/io.md
    - Preprocessing: api/preprocessing.md
    - Chemometrics: api/chemometrics.md
    - Features: api/features.md
    - Metrics: api/metrics.md
    - Statistics: api/stats.md
    - Visualization: api/visualization.md
    - QC & Validation: api/qc.md
    - Workflows: api/workflows.md
    - ML: api/ml.md
    - Reproducibility: api/reproducibility.md
    - Demo Datasets: api/demo.md
    - Deploy: api/deploy.md
  
  - Reference:
    - Data Formats: reference/data_format.md
    - Glossary: reference/glossary.md
    - Changelog: reference/changelog.md
    - Versioning: reference/versioning.md
    - Citing: reference/citing.md
  
  - Help:
    - Troubleshooting: help/troubleshooting_faq.md
    - Common Problems: help/common_problems.md
    - Report Issues: help/reporting_issues.md
  
  - Developer Guide:
    - Contributing: developer-guide/contributing.md
    - Testing & CI: developer-guide/testing_and_ci.md
    - Documentation: developer-guide/documentation_guidelines.md
    - Plugins: developer-guide/writing_plugins.md
    - Releasing: developer-guide/releasing.md
```

### D.3 Naming & Style Conventions

**Page Naming:**
- Use kebab-case for filenames: `loading_data.md`, `oil_authentication.md`
- Use Title Case for page titles (shown in browser / nav)
- For examples: prefix with `01_`, `02_`, etc. for ordering

**Headings:**
- H1 for page title (auto-generated from markdown filename in nav)
- H2 for major sections (Prerequisites, Example, Expected Output)
- H3 for subsections (no deeper nesting in most cases)

**Code Blocks:**
- All code must be copy-paste runnable
- Always specify language (```python, ```bash, ```yaml)
- Avoid relative paths; use `Path("outputs")` instead of `./outputs/`
- No hard-coded paths (use `$HOME`, environment variables, or relative to project)

**Docstrings:**
- Google style (Parameters, Returns, Raises, Examples sections)
- Include parameter types: `ndarray (n_samples, n_wavenumbers)`
- Include units where relevant: `temperature : float, in Celsius`
- At least 1 short runnable example for every public function
- "When to use" section for non-obvious functions

---

## E. Staged Execution Plan

### Phase 0: Safety & Tooling (Effort: Small)

**Goal:** Prevent regressions; add guardrails for quality.

#### Phase 0.1: Link Checker in CI/CD

**File Actions:**
- [ ] Create `.github/workflows/docs-links.yml` with:
  - Run mkdocs build and extract broken anchor warnings
  - Fail on warnings
- [ ] Add to pre-commit hooks: `pre-commit run --hook-id mkdocs-build`

**Acceptance Criteria:**
- CI/CD workflow created and passing locally
- Pre-commit hooks configured
- Build fails on broken links/anchors

**Risk:** None (additive only)

#### Phase 0.2: Docstring Linter

**File Actions:**
- [ ] Create `scripts/check_docstrings.py`:
  - Scan public API (from `__all__`)
  - Check for Google-style docstrings (Parameters, Returns, Examples)
  - Report missing docstrings as errors
- [ ] Add to pre-commit

**Acceptance Criteria:**
- Script runs locally and reports gaps
- Identifies which 25 functions need docstrings
- All new code requires docstrings

**Risk:** May need tuning for false positives

**Estimated Effort:** Small (50–100 lines Python)

---

### Phase 1: Restructure Docs Navigation & Eliminate Duplicates (Effort: Medium)

**Goal:** Clear out orphaned pages, consolidate duplicates, fix nav structure.

#### Phase 1.1: Merge Duplicate Folders

**File Actions:**

1. **Consolidate `/stats/` → `/methods/statistics/`**
   - [ ] Copy unique content from `docs/stats/` into `docs/methods/statistics/`
   - [ ] Delete `docs/stats/`
   - [ ] Add redirect in mkdocs.yml: `stats/: methods/statistics/`

2. **Consolidate `/api/` + `/08-api/` → Keep only `/api/` (Manual pages for now)**
   - [ ] Review both folders side-by-side
   - [ ] Keep more complete versions in `/api/`
   - [ ] Delete `/08-api/`
   - [ ] Add redirect: `08-api/: api/`

3. **Consolidate `/reference/` + `/09-reference/` → Keep `/reference/`**
   - [ ] Merge content
   - [ ] Delete `/09-reference/`
   - [ ] Add redirect

4. **Consolidate `/help/`, `/troubleshooting/`, `/10-help/` → Keep `/help/`**
   - [ ] Merge all troubleshooting content
   - [ ] Keep single FAQ page
   - [ ] Delete `/troubleshooting/` and `/10-help/`
   - [ ] Add redirects

5. **Archive `/05-advanced-topics/` → Move to `docs/_internal/archived-advanced-topics/`**
   - [ ] Decide: keep 2–3 most important pages (MOATS, model registry) in `/developer-guide/advanced/`
   - [ ] Move rest to archive
   - [ ] Do NOT delete; preserve in version control for historical reference

6. **Clean `/docs/_internal/archive/` → Remove files not needed for historical reference**
   - [ ] Keep: old API pages (for historical comparison), project history logs
   - [ ] Delete: build logs, temporary working files, smoke test results
   - [ ] Compress size to <50 files

**File Operations Detail:**

```bash
# Example (consolidate stats)
mkdir -p docs/methods/statistics-merged
cp docs/stats/*.md docs/methods/statistics-merged/
# (manual review + merge)
rm -rf docs/stats/
# (add redirect to mkdocs.yml)

# Consolidate reference
# (similar pattern)
```

**Acceptance Criteria:**
- No duplicate folders remaining
- All `/docs/` subfolders either (a) in mkdocs.yml nav, or (b) in `_internal/` archive
- mkdocs build with no "orphaned pages" warnings
- All old paths have redirects

**Risk:** May miss interdependencies between duplicates; mitigate with careful manual review

**Estimated Effort:** Medium (2–4 hours manual review + file operations)

#### Phase 1.2: Add Orphaned Pages to mkdocs.yml Nav

**File Actions:**
- [ ] Add `/tutorials/` as new section:
  - `Examples: examples/index.md` (NEW section, replaces placeholder)
  - `Beginner Tutorials: tutorials/beginner/01-load-and-plot.md` (and 2–3 others)
  - `Intermediate Tutorials: tutorials/intermediate/01-oil-authentication.md` (and 2–3 others)
- [ ] Add foundations / theory pages as needed (likely already covered)
- [ ] Remove dead redirects from mkdocs.yml (ones pointing to deleted pages)

**Acceptance Criteria:**
- mkdocs build lists NO orphaned pages
- All pages in `/docs/` are either in nav or in `_internal/`
- Nav structure is clean and not overly deep (max 4 levels)

**Risk:** Navigation may become too wide; mitigate by grouping related topics

**Estimated Effort:** Small (1–2 hours)

---

### Phase 2: Docstrings & API Reference Generation (Effort: Large)

**Goal:** Achieve 100% docstring coverage for public API; auto-generate API reference from docstrings.

#### Phase 2.1: Audit & Fix Docstrings in Core Modules

**File Actions:**

For each module below, complete docstrings (Google style: Parameters, Returns, Raises, Examples):

1. **`src/foodspec/core/dataset.py`** (~20 methods need fixes)
   - [ ] `FoodSpectrumSet.to_wide_dataframe()`
   - [ ] `FoodSpectrumSet.subset()`
   - [ ] `FoodSpectrumSet.apply()`
   - [ ] `FoodSpectrumSet.scale()`
   - [ ] `FoodSpectrumSet.select_wavenumber_range()`
   - [ ] `FoodSpectrumSet.train_test_split()`
   - [ ] `FoodSpectrumSet.to_hdf5()` + `.from_hdf5()`
   - [ ] `FoodSpectrumSet.to_parquet()` + `.from_parquet()`
   - [ ] ... (others)

2. **`src/foodspec/metrics.py`** (~5 functions)
   - [ ] `compute_regression_metrics()`
   - [ ] `compute_roc_curve()`
   - [ ] `compute_pr_curve()`
   - [ ] ... (others)

3. **`src/foodspec/io/__init__.py` + submodules** (~6 functions)
   - [ ] `load_library()`
   - [ ] `create_library()`
   - [ ] `load_csv_spectra()`
   - [ ] `detect_format()`
   - [ ] ... (others)

4. **`src/foodspec/stats/` submodules** (audit structure first; ~10 functions)
   - [ ] `run_anova()`
   - [ ] `run_ttest()`
   - [ ] ... (others)

5. **`src/foodspec/qc/` submodules** (~8 functions)
   - [ ] `detect_leakage()`
   - [ ] `detect_batch_label_correlation()`
   - [ ] `compute_readiness_score()`
   - [ ] ... (others)

6. **`src/foodspec/viz/` submodules** (audit; ~10+ functions)
   - Sample 3–4 key functions; fix; then template for others

**Template for Each Function:**

```python
def function_name(param1: SomeType, param2: str = "default") -> ReturnType:
    """One-line summary (imperative mood).
    
    Longer description (1–2 sentences) explaining:
    - What the function does
    - High-level use case
    - Key assumptions or limitations
    
    Parameters
    ----------
    param1 : SomeType
        Description. Include units if relevant (e.g., "in Celsius").
    param2 : str, default="default"
        Description of choices or behavior.
    
    Returns
    -------
    ReturnType
        Description of return value(s). Include shape/structure if applicable.
    
    Raises
    ------
    ValueError
        If [specific condition].
    
    Examples
    --------
    >>> result = function_name(param1, param2="value")
    >>> print(result)
    [expected output]
    
    See Also
    --------
    related_function : Related functionality.
    
    Notes
    -----
    Any caveats or implementation notes.
    """
    # implementation...
```

**Acceptance Criteria:**
- All 60+ public functions have Google-style docstrings
- Each docstring includes: Parameters (with types), Returns, and at least 1 runnable example
- `scripts/check_docstrings.py` returns 0 errors
- mkdocstrings generates API pages from docstrings successfully
- Test: `mkdocs build` includes auto-generated API sections with no "missing docstring" warnings

**Risk:** Large effort; mitigate by batch processing (5–10 functions per day) and using templates

**Estimated Effort:** Large (15–25 hours across team)

#### Phase 2.2: Auto-Generate API Reference from Docstrings

**File Actions:**
- [ ] Verify mkdocstrings plugin is configured in `mkdocs.yml` (already is)
- [ ] Replace manual `/api/core.md` with auto-generated equivalent:
  ```markdown
  # Core API
  ::: foodspec.core.dataset.FoodSpectrumSet
  ::: foodspec.core.spectrum.Spectrum
  ::: foodspec.core.output_bundle.OutputBundle
  ::: foodspec.core.run_record.RunRecord
  ```
- [ ] Repeat for all API reference pages (io, preprocessing, metrics, stats, etc.)
- [ ] **Delete manual pages** once auto-gen is verified

**Acceptance Criteria:**
- All API pages auto-generated from docstrings
- API pages update automatically when docstrings change
- `mkdocs build --strict` passes (all docstrings valid)
- API pages match curated order (not alphabetical)

**Risk:** None (additive + verified before deleting manual pages)

**Estimated Effort:** Small (2–3 hours)

---

### Phase 3: Examples Gallery & Flagship Examples (Effort: Medium)

**Goal:** Create unified, narrative examples gallery with 3–5 flagship examples.

#### Phase 3.1: Restructure Examples Folder

**File Actions:**
- [ ] Create `docs/examples/` folder with structure:
  ```
  docs/examples/
  ├── index.md                    # Gallery landing page (narrative)
  ├── 01_oil_authentication.md    # Example page (narrative + embedded code + figure)
  ├── 02_heating_quality.md
  ├── 03_hyperspectral_mapping.md
  ├── 04_mixture_analysis.md
  ├── 05_classification_workflow.md
  └── _code/                      # Runnable code
      ├── ex01_oil_auth.py
      ├── ex02_heating.py
      ├── ex03_hsi.py
      ├── ex04_mixture.py
      └── ex05_classify.py
  ```

- [ ] Move/extract from `/examples/*.py`:
  - `examples/oil_authentication_quickstart.py` → `docs/examples/_code/ex01_oil_auth.py` (copy/link)
  - `examples/heating_quality_quickstart.py` → `docs/examples/_code/ex02_heating.py` (already refactored)
  - `examples/hyperspectral_demo.py` → `docs/examples/_code/ex03_hsi.py`
  - `examples/mixture_analysis_quickstart.py` → `docs/examples/_code/ex04_mixture.py`
  - Create new `ex05_classify.py` from tutorials

- [ ] Move notebooks from `/examples/notebooks/` → `docs/examples/_notebooks/`

**Acceptance Criteria:**
- All 5 flagship examples have corresponding markdown pages in `docs/examples/`
- Each markdown page has: title, intro, code block (copy-paste runnable), expected output, figure
- Code can be copy-pasted and run standalone
- All figures are generated from code (not decorative)

**Risk:** Code in examples may use older patterns; mitigate by modernizing before copying

**Estimated Effort:** Medium (4–6 hours)

#### Phase 3.2: Create Examples Gallery Landing Page

**File Actions:**
- [ ] Create `docs/examples/index.md` with:
  - Brief intro (what examples show, how to use them)
  - Table of contents with descriptions:
    | Example | Level | Domain | Time | Description |
    | --- | --- | --- | --- | --- |
    | Oil Authentication | Beginner | Authentication | 10 min | Load spectra, apply classification model, interpret results |
    | Heating Quality | Intermediate | Monitoring | 15 min | Track chemical changes during heating; detect outliers |
    | ... | ... | ... | ... | ... |
  - Links to each example
  - Prerequisites section (what you need to know before examples)
  - "Common use cases" index (e.g., "I want to classify spectra" → points to multiple examples)

**Acceptance Criteria:**
- Landing page is visually clear and easy to navigate
- Each example has metadata (level, domain, time estimate)
- Links to all examples work
- Pre-requisites and learning path clear

**Estimated Effort:** Small (1–2 hours)

---

### Phase 4: Polish & Consistency (Effort: Medium)

**Goal:** Refine docs tone, add "when to use" guidance, glossary, style consistency.

#### Phase 4.1: Add "When to Use" Guidance

**File Actions:**

1. **Create `docs/theory/when_to_use_guide.md`** with decision trees:
   - "I want to detect if samples are different" → ANOVA (parametric) vs Kruskal-Wallis (non-parametric)
   - "I want to build a classification model" → Decision tree vs SVM vs Neural Network (with trade-offs)
   - "I want to preprocess spectra" → Decision tree (normalize? smooth? baseline correct? order?)

2. **For each major method page, add subsection: "When NOT to use"**
   - `methods/preprocessing/normalization_smoothing.md`: Add "When NOT to use Normalization" (e.g., when zero-variance columns present)
   - `methods/chemometrics/models_and_best_practices.md`: Add anti-patterns

3. **For each API function with non-obvious behavior, add "Notes" or "Warnings" sections**
   - E.g., `FoodSpectrumSet.train_test_split()`: Note about stratification, random state, group leakage

**Acceptance Criteria:**
- Decision trees present and helpful
- Every major method has "When NOT to use" section
- Warnings/Notes in API docstrings for tricky functions
- Tone consistent with scikit-learn (practical, direct)

**Risk:** Requires domain expertise; mitigate by having authors review

**Estimated Effort:** Medium (4–8 hours)

#### Phase 4.2: Glossary & Terminology Consistency

**File Actions:**
- [ ] Create/expand `docs/reference/glossary.md` with:
  - Spectroscopy terms (wavenumber, transmission, absorbance, Raman shift, etc.)
  - Machine learning terms (stratification, leakage, cross-validation, etc.)
  - FoodSpec-specific terms (RQ engine, MOATS, HyperSpectralCube, etc.)
  - Link from glossary to relevant pages (e.g., "stratification" → link to validation methods)

- [ ] Audit docs for **terminology consistency**:
  - E.g., is it "spectral dataset" or "spectrum set" or "spectra"?
  - Use consistent terminology across all pages
  - Add glossary anchor links where terms first appear

**Acceptance Criteria:**
- Glossary covers 50+ terms
- Terminology consistent across all docs pages
- First mention of term links to glossary

**Risk:** None (purely additive)

**Estimated Effort:** Small–Medium (3–5 hours)

#### Phase 4.3: Style Guide & Tone Audit

**File Actions:**
- [ ] Create/expand `docs/developer-guide/documentation_guidelines.md` with:
  - Code block standards (copy-paste safe, language specified, no relative paths)
  - Heading hierarchy (use of H1, H2, H3)
  - Link format (relative paths, no `/docs/`, use `[text](path/to/page.md)`)
  - Example structure template (intro, code, expected output, figure)
  - Docstring template (Parameters, Returns, Examples)
  - Tone guidelines (active voice, second person for tutorials, neutral for reference)

- [ ] Audit 10–15 sample pages for compliance:
  - Check tone (is it "you" or "the user"? Is it friendly?)
  - Check code blocks (are they runnable?)
  - Check links (do they work?)
  - Flag for improvement

**Acceptance Criteria:**
- Style guide documented
- 10–15 pages reviewed and flagged for improvement
- Plan to fix flagged pages in future sprints

**Risk:** None (guidance only)

**Estimated Effort:** Small (2–3 hours for guide + spot-check)

---

### Phase 5: Pre-Commit & Validation (Effort: Small)

**Goal:** Automate quality checks to prevent regressions.

**File Actions:**
- [ ] Create `.pre-commit-config.yaml` with:
  - Docstring linter (Phase 0.2 script)
  - mkdocs build (to catch broken links)
  - Link checker (Phase 0.1)
  - Code formatter (black, if not already present)

- [ ] Update CI/CD (`.github/workflows/`) to run pre-commit checks on PR

**Acceptance Criteria:**
- Pre-commit hooks run on every commit
- CI/CD enforces checks on PR
- Developers cannot merge without passing checks

**Risk:** None (helpful guardrails)

**Estimated Effort:** Small (1–2 hours)

---

## F. Definition of Done Checklist

Use this checklist to verify you've reached the scikit-learn documentation standard:

### Navigation & Discovery
- [ ] All pages in `/docs/` are either in `mkdocs.yml` nav OR in `docs/_internal/` archive
- [ ] mkdocs build shows 0 "orphaned pages" warnings
- [ ] No duplicate content folders (stats + methods/statistics, api + 08-api, etc.)
- [ ] All broken links detected by CI/CD and fixed
- [ ] No broken anchors in cross-references

### Docstrings & API Reference
- [ ] 100% of public functions (in `__all__` or top-level imports) have Google-style docstrings
- [ ] Each docstring includes: Parameters (with types/units), Returns, at least 1 Example, Raises (if applicable)
- [ ] API reference auto-generated from docstrings (no manual pages for main API)
- [ ] New functions require docstrings before merge (enforced by pre-commit)
- [ ] mkdocstrings generates valid API pages for all modules

### Examples & Figures
- [ ] 3–5 flagship examples in `docs/examples/` with narrative + code + figures
- [ ] All figures in docs have traceable code (committed to repo, linked in docs)
- [ ] All code blocks in docs are copy-paste runnable and minimal (1 concept per block)
- [ ] Examples gallery has metadata (level, domain, time estimate) and discoverable index

### User Guide & "When to Use"
- [ ] Major methods have "When to use" and "When NOT to use" sections
- [ ] Core workflows have decision trees or recommended defaults
- [ ] Glossary covers 50+ terms with cross-links
- [ ] At least 3 pages have "anti-patterns" or "common pitfalls" sections
- [ ] Tone is consistent with scikit-learn (practical, direct, second-person in tutorials)

### Code Quality & Safety
- [ ] All code examples use `from pathlib import Path`, not hardcoded paths
- [ ] No imports from `examples/` folder in docs code blocks (all imports from public API)
- [ ] Pre-commit hooks enforce docstring + link checks
- [ ] CI/CD build fails on broken links or missing docstrings
- [ ] All new code reviewed for docs quality (inline docstrings, examples)

### Build & Validation
- [ ] `mkdocs build --strict` passes with 0 warnings (except pre-existing ones outside our control)
- [ ] `pytest` passes with 75%+ coverage maintained
- [ ] All examples run successfully (e.g., `python docs/examples/_code/ex01_oil_auth.py`)
- [ ] Link checker finds 0 broken external links (redirect chains if any)
- [ ] Docstring linter reports 0 errors

### Maintenance
- [ ] Documentation guidelines documented in `developer-guide/`
- [ ] Deprecations/breaking changes reflected in docs and API (with migration guides if needed)
- [ ] Release checklist includes "update API reference" or equivalent
- [ ] Every PR to `src/foodspec/` requires docs update (checked in PR review)

---

## G. Risk Mitigation & Next Steps

### Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Large docstring effort takes 20+ hours | Delay; burnout | Batch work; use templates; team effort |
| Consolidating duplicates breaks redirects | Broken links; users lost | Test all redirects; set up link checker first |
| Examples may use outdated patterns | Confusing; not copy-paste safe | Modernize before adding to gallery |
| Navigation changes confuse existing users | UX regression | Add prominent migration note; keep old nav in redirects |
| mkdocstrings auto-gen doesn't render well | Worse than manual pages | Stage changes; verify before deleting manual pages |

### Immediate Next Steps (To Start Today)

1. **Phase 0.1 (30 min):** Create link checker CI/CD workflow
2. **Phase 0.2 (1 hour):** Create docstring linter script
3. **Phase 1.1 (2 hours):** Start consolidating `/stats/` + `/methods/statistics/`
4. **Phase 2.1 (ongoing):** Pick 10 docstrings to fix as template examples
5. **Phase 3.1 (2 hours):** Extract 5 flagship examples and create landing page structure

### Estimated Total Effort

| Phase | Effort | Owner |
|-------|--------|-------|
| Phase 0: Safety & Tooling | Small (2–3 hours) | 1 person (engineer) |
| Phase 1: Restructure Docs | Medium (4–6 hours) | 1 person (tech writer / lead) |
| Phase 2: Docstrings | Large (15–25 hours) | Team (batch 5–10 per sprint) |
| Phase 3: Examples Gallery | Medium (4–6 hours) | 1–2 people |
| Phase 4: Polish | Medium (6–10 hours) | 1 person (tech writer) |
| Phase 5: Validation | Small (1–2 hours) | 1 person (engineer) |
| **Total** | **52–82 hours** | **2–4 weeks (if 2 people; 1 week if 4 people)** |

---

## H. Conclusion

FoodSpec has **strong domain content and solid build infrastructure**, but falls short of scikit-learn standard due to:
1. **Navigation chaos** (130+ orphaned pages)
2. **Docstring gaps** (~60% of public API missing formal docstrings)
3. **Scattered examples** (no unified gallery)
4. **Manual API reference** (prone to drift)

**Quick wins (Phase 0–1):** Link checker, docstring linter, consolidate duplicates. Can be done in **1–2 weeks** with 1–2 people.

**Main effort (Phase 2–4):** Fix docstrings (15–25 hours), rebuild API reference from docstrings, create examples gallery. Can be done in **3–4 weeks** with team batching.

**Long-term (Phase 5+):** Automated quality gates (pre-commit, CI/CD) to prevent regression.

**ROI:** Effort is front-loaded; once docstrings are complete and examples gallery is live, maintenance cost drops significantly. Reviews will see professional, discoverable, copy-paste-safe documentation that reflects the code exactly.

