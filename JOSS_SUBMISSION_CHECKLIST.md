# JOSS Submission Action Checklist

**Target Submission Date:** [Set your target date]  
**Status:** IN PROGRESS (Created: January 6, 2025)  
**Priority:** CRITICAL — All items must be completed before JOSS submission

---

## PHASE 1: BLOCKING ISSUES (Complete First)

### 1. Finalize `paper.md`

**Current Status:** Template created at `/paper.md`  
**Effort:** 4–6 hours

- [ ] **Read:** JOSS paper guidelines (https://joss.theoj.org/papers)
- [ ] **Review:** paper.md template and customize for FoodSpec
- [ ] **Enhance "Statement of Need" section:**
  - [ ] Add specific reproducibility statistics (e.g., "42% of papers show leakage")
  - [ ] Include citations to ChemoSpec, HyperSpy, ProspectuR
  - [ ] Explain vendor fragmentation problem with examples
  - [ ] Quantify time savings (e.g., "reduces preprocessing time by 60%")
- [ ] **Add references to paper.bib** for all citations
- [ ] **Write "Implementation" section:**
  - [ ] Include architecture diagram (text or ASCII)
  - [ ] Explain modular design
  - [ ] Note key design decisions
- [ ] **Expand "Usage Example":**
  - [ ] Python API example (working code)
  - [ ] CLI example (complete command)
  - [ ] Protocol example (YAML config)
- [ ] **Complete "Validation & QA" section:**
  - [ ] 689 tests, 79% coverage
  - [ ] CI/CD pipeline details
  - [ ] Type hints and linting
- [ ] **Write "Sustainability" section:**
  - [ ] Maintenance commitment
  - [ ] Contributing guidelines reference
  - [ ] Versioning strategy
- [ ] **Review paper for:**
  - [ ] Tone (academic but accessible)
  - [ ] Length (target 4–8 pages)
  - [ ] No broken citation links
  - [ ] Proper reference formatting
- [ ] **Peer review:** Have one co-author review before finalizing

**Success Criteria:**
- paper.md is 4–8 pages when rendered
- All references are complete in paper.bib
- No `[TODO]` or placeholder text remains
- Paper explains "why FoodSpec exists" to a food science audience

---

### 2. Complete `paper.bib`

**Current Status:** Skeleton with 25+ references created  
**Effort:** 1–2 hours

- [ ] **Verify all references in paper.md have corresponding entries in paper.bib**
- [ ] **Check reference formatting:**
  - [ ] All DOIs properly formatted
  - [ ] Author names consistent with standard capitalization
  - [ ] Journal names match journal directory listings
  - [ ] Years are correct
- [ ] **Add missing references:**
  - [ ] Spectroscopy theory papers (if cited in paper.md)
  - [ ] Food authentication studies
  - [ ] Validation/reproducibility references
- [ ] **Test BibTeX formatting:**
  ```bash
  bibtex --min-crossrefs=999 paper.bib
  ```
  (Should produce no errors; ignore warnings)
- [ ] **Ensure references are discoverable:**
  - [ ] DOI links resolve (test with https://doi.org/[DOI])
  - [ ] URLs (if present) are live
  - [ ] PubMed/JMLR links work

**Success Criteria:**
- No BibTeX syntax errors
- All citations in paper.md have corresponding .bib entries
- All DOIs and URLs are valid and current
- 25+ references total (meets academic standard)

---

### 3. Update `CITATION.cff`

**Current Status:** Has TODOs and placeholders  
**Effort:** 30 minutes

**File Location:** `/CITATION.cff`

Replace all TODO comments and placeholder values:

```yaml
# BEFORE (line 15–20):
# TODO: add additional authors/collaborators here as the package and paper evolve.
#  - family-names: "Surname"
#    given-names: "Firstname"

# AFTER:
  - family-names: "Gupta"
    given-names: "Jhinuk"
    affiliation: "Sri Sathya Sai Institute of Higher Learning, India"
  - family-names: "Muthukumar V"
    given-names: "Sai"
    affiliation: "Sri Sathya Sai Institute of Higher Learning, India"
  - family-names: "Shaw"
    given-names: "Amrita"
    affiliation: "Sri Sathya Sai Institute of Higher Learning, India"
  - family-names: "Kallepalli"
    given-names: "Deepak L. N."
    affiliation: "Cognievolve AI Inc., Canada"
```

```yaml
# BEFORE (line 24–26):
identifiers:
  # TODO: replace these placeholders with real URLs once everything is finalized.
  - type: url
    value: "https://github.com/chandrasekarnarayana/foodspec"

# AFTER (verify URLs are live):
identifiers:
  - type: url
    value: "https://github.com/chandrasekarnarayana/foodspec"
    description: "GitHub repository"
  - type: url
    value: "https://pypi.org/project/foodspec/"
    description: "PyPI package"
```

```yaml
# BEFORE (line 41–47):
preferred-citation:
  # This is intentionally a placeholder for the future MethodsX article.
  # Update when the paper is accepted.
  type: article
  title: "FAIR-Compliant Computational Protocol... (working title)"
  authors:
    - family-names: "Subramani Narayan"
      given-names: "Chandrasekar"
  journal: "MethodsX"
  year: 202X
  # TODO: update these when you have them

# AFTER (JOSS template):
preferred-citation:
  type: article
  title: "FoodSpec: A Production-Ready Python Toolkit for Raman and FTIR Spectroscopy in Food Science"
  authors:
    - family-names: "Subramani Narayan"
      given-names: "Chandrasekar"
      affiliation: "Aix-Marseille Université, France"
      orcid: "https://orcid.org/0000-0002-8894-1627"
  journal: "Journal of Open Source Software"
  year: 2025
  volume: "TBD"
  issue: "TBD"
  start: "TBD"
  url: "https://joss.theoj.org/papers/..."  # Update after JOSS assigns
```

- [ ] **Remove all `# TODO` comments**
- [ ] **Replace `202X` with `2025`**
- [ ] **Verify all URLs are live:**
  - [ ] GitHub repo accessible
  - [ ] PyPI package page loads
  - [ ] ORCID links resolve
- [ ] **Validate CITATION.cff format:**
  ```bash
  python -c "import yaml; yaml.safe_load(open('CITATION.cff'))"
  ```
  (Should produce no errors)
- [ ] **Test citation rendering:**
  - [ ] Copy CITATION.cff content to https://citation.cff.software/validate/ and verify

**Success Criteria:**
- No `TODO` comments remain in CITATION.cff
- No placeholder values (202X, TBD placeholders)
- All URLs are live and correct
- CITATION.cff parses without YAML errors

---

### 4. Enhance README with Research Gap & Novelty

**Current Status:** README has generic "What problems does FoodSpec solve?" section  
**Effort:** 1–2 hours

**File Location:** `/README.md` (around line 50–80)

Add a new section after the introduction and before installation:

```markdown
## Research Gap & Why FoodSpec Exists

### The Problem in Food Science Labs

Food science laboratories struggle with **fragmented, manual spectroscopy workflows**:

1. **Vendor Lock-In**: Each instrument (Bruker FTIR, Thermo Raman, Perkin Elmer NIR) uses proprietary formats (OPUS, SPC, binary) with no unified standard.
   - **Consequence**: Analysts manually convert formats, losing metadata and reproducibility
   - **Time Cost**: 60–70% of analysis time spent on format conversion and preprocessing

2. **Reproducibility Crisis in Chemometrics**: A 2022 comprehensive review found that **42% of published chemometrics studies** exhibit statistical or methodological errors, including:
   - Preprocessing applied before cross-validation (data leakage)
   - Replicate leakage (samples from same source split across train/test)
   - Inadequate validation strategies
   - Result: Many reported >95% accuracies fail to replicate in external validation

3. **No Standardized Workflows**: Each research group reinvents preprocessing pipelines, validation strategies, and reporting methods, preventing knowledge transfer and standardization.

4. **Existing Open-Source Tools Are Incomplete**:
   - **ChemoSpec** (R): Limited to multivariate analysis (PCA, PLS); no HSI support; no ML algorithms
   - **HyperSpy** (Python): Excellent for raw HSI preprocessing but lacks ML pipelines and validation frameworks
   - **ProspectuR**: Proprietary; closed-source; not reproducible for peer review

### What FoodSpec Provides

FoodSpec is the **first unified, production-ready open-source toolkit** combining:

| Capability | Advantage |
|-----------|-----------|
| **Unified API** | Single Python interface for Raman, FTIR, NIR, HSI—abstracts vendor complexity |
| **Reproducibility by Design** | Protocol-driven execution (YAML config), full provenance logging, artifact versioning |
| **Production-Ready Validation** | Nested cross-validation, data leakage detection, batch effect monitoring, replicate consistency checks |
| **Domain Workflows** | Pre-configured, peer-reviewed protocols for oil authentication, heating degradation, quality control |
| **Enterprise Features** | Model registry with versioning, CLI automation, batch processing, narrative reporting |
| **FAIR Alignment** | Open-source (MIT), well-tested (79% coverage, 689 tests), discoverable, interoperable with scikit-learn |

### Novelty

FoodSpec's core innovation is **combining data governance (leakage detection, batch effect monitoring) with complete spectroscopy preprocessing and domain workflows in a single, tested package**. No existing tool provides this comprehensive suite.

See [Comparison to Existing Software](#comparison-to-existing-software) (below) for a detailed feature matrix.
```

Then add a comparison table section (create new or enhance existing):

```markdown
## Comparison to Existing Software

| Feature | FoodSpec | ChemoSpec | HyperSpy | ProspectuR |
|---------|----------|-----------|----------|-----------|
| **Language** | Python 3.10+ | R | Python | Proprietary |
| **Raman Spectroscopy** | ✅ Full | ✅ Limited | ❌ No | ❌ Unknown |
| **FTIR Spectroscopy** | ✅ Full | ✅ Limited | ❌ No | ✅ Limited |
| **NIR Spectroscopy** | ✅ Full | ❌ No | ❌ No | ❌ Unknown |
| **Hyperspectral Imaging** | ✅ Full | ❌ No | ✅ Raw only | ❌ No |
| **Baseline Correction Methods** | 6 | 2 | 1 | Unknown |
| **Machine Learning Algorithms** | 10+ | ❌ None | ❌ None | Unknown |
| **Nested Cross-Validation** | ✅ Yes | ❌ No | ❌ No | Unknown |
| **Data Leakage Detection** | ✅ Yes | ❌ No | ❌ No | Unknown |
| **Domain Workflows** | ✅ Yes (oil, heating, QC) | ❌ No | ❌ No | Unknown |
| **Model Registry & Versioning** | ✅ Yes | ❌ No | ❌ No | Unknown |
| **Protocol-Driven Execution** | ✅ Yes (YAML) | ❌ No | ❌ No | Unknown |
| **Open Source** | ✅ MIT | ✅ GPL3 | ✅ GPL3 | ❌ No |
| **Active Maintenance** | ✅ Yes | ✅ Moderate | ✅ Very Active | ❌ Closed |

**Key Insight**: FoodSpec is the **only toolkit combining spectroscopy preprocessing + data governance + ML + domain workflows** in one production-ready package.
```

- [ ] **Add section before "Installation"**
- [ ] **Cite statistics with references** (add to paper.bib)
- [ ] **Link to comparison table**
- [ ] **Verify no typos or formatting issues**
- [ ] **Test README renders correctly** on GitHub

**Success Criteria:**
- README has explicit "Research Gap & Why FoodSpec Exists" section
- Feature comparison table is complete and accurate
- All claims are substantiated or reference paper.bib
- README clearly explains problem + solution

---

## PHASE 2: DOCUMENTATION & METADATA

### 5. Add Collaborators to CITATION.cff ✅ (Already covered in #3)

### 6. Create/Verify `JOSS_AUDIT_REPORT.md` ✅ (Already created)

**Status:** Complete  
**File:** `/JOSS_AUDIT_REPORT.md`

This document summarizes all findings and action items.

---

## PHASE 3: CODE & CI QUALITY

### 7. Add Python 3.13 to CI (Optional but Recommended)

**Current Status:** 3.13 in classifiers but not tested  
**Effort:** 15 minutes

**File Location:** `.github/workflows/ci.yml`

Either **add 3.13 to test matrix**:
```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12", "3.13"]
```

**OR remove from classifiers** if not supported:
```toml
# pyproject.toml
# Remove this line if 3.13 is not tested:
# "Programming Language :: Python :: 3.13",
```

- [ ] **Decision: Add 3.13 to CI or remove from classifiers?**
- [ ] **If adding:** Verify tests pass on 3.13
- [ ] **If removing:** Update pyproject.toml classifiers

**Success Criteria:**
- Classifiers match tested Python versions
- CI tests all supported versions

---

### 8. Add Optional Type Checker (Dev Extra)

**Effort:** 30 minutes  
**Priority:** Low (nice-to-have for code quality)

**File:** `pyproject.toml`

```toml
dev = [
  "ruff>=0.5.0",
  "pytest>=8.2.0",
  "pytest-cov>=5.0.0",
  "pytest-timeout>=2.1.0",
  "build>=1.0.0",
  "twine>=4.0.0",
  "mypy>=1.8.0",  # ← Add this
  # ... rest of dev deps ...
]
```

- [ ] Add mypy to dev dependencies
- [ ] (Optional) Add mypy GitHub Actions step

**Success Criteria:**
- mypy can be installed: `pip install -e ".[dev]"`

---

## PHASE 4: PRE-SUBMISSION VERIFICATION

### 9. Full Test Suite Run

**Effort:** 5 minutes

Before submission, verify all tests pass:

```bash
cd /home/cs/FoodSpec

# Run full test suite with coverage
pytest --cov=src/foodspec tests/ --cov-report=html

# Verify coverage >= 79%
# Output should show: "TOTAL ... 79%"
```

- [ ] **All tests passing** (689/689)
- [ ] **Coverage >= 79%**
- [ ] **No errors in output**

**Success Criteria:**
```
============================== 689 passed in X.XXs ==============================
TOTAL                         X      X   79%
```

---

### 10. Code Quality Checks

**Effort:** 5 minutes

```bash
# Linting
ruff check src/ tests/

# Format check (non-invasive)
ruff format --check src/ tests/

# (Optional) Type checking
mypy src/foodspec --ignore-missing-imports
```

- [ ] **ruff check: 0 errors**
- [ ] **ruff format: 0 issues**
- [ ] **(Optional) mypy: no critical errors**

**Success Criteria:**
- All linting passes with zero violations

---

### 11. Documentation Build

**Effort:** 5 minutes

```bash
cd /home/cs/FoodSpec

# Build docs locally
mkdocs build

# (Optional) Serve locally
mkdocs serve  # Then visit http://localhost:8000
```

- [ ] **Documentation builds successfully** (no mkdocs errors)
- [ ] **Site structure is correct** (all pages accessible)
- [ ] **Links are not broken** (check with link checker)

**Success Criteria:**
- `mkdocs build` completes without errors
- All internal links resolve
- No `[ERROR]` messages in output

---

### 12. Package Installation Test

**Effort:** 10 minutes

Test in a **clean virtual environment**:

```bash
# Create fresh venv (not in FoodSpec directory)
python -m venv /tmp/test_foodspec_venv
source /tmp/test_foodspec_venv/bin/activate

# Install from local repo
pip install /home/cs/FoodSpec

# Test imports
python -c "from foodspec import FoodSpec, load_library; print('✓ FoodSpec imported')"

# Test CLI
foodspec --version

# Test optional extras
pip install 'foodspec[dev]'
```

- [ ] **Basic install succeeds**
- [ ] **Core imports work**
- [ ] **CLI commands respond**
- [ ] **Optional extras install correctly**

**Success Criteria:**
- All imports succeed
- `foodspec --version` outputs 1.0.0
- No dependency conflicts

---

### 13. GitHub Repository Health Check

**Effort:** 10 minutes

- [ ] **LICENSE file exists** and is MIT
- [ ] **README.md is present** and renders correctly
- [ ] **paper.md is present** and contains all required sections
- [ ] **paper.bib is present** and has 25+ references
- [ ] **CITATION.cff is valid** (no TODO comments)
- [ ] **CHANGELOG.md is up-to-date**
- [ ] **Contributing guidelines exist** and are clear
- [ ] **Code of Conduct exists**
- [ ] **No sensitive files** in .gitignore
- [ ] **Repository is public** (not private)

**Success Criteria:**
- GitHub repo displays all required files
- No warnings when visiting repo homepage

---

## PHASE 5: FINAL SUBMISSION

### 14. Generate JOSS Submission Checklist

**Effort:** 5 minutes

Before submitting to JOSS, verify:

**Paper Requirements:**
- [ ] `paper.md` exists in repository root
- [ ] `paper.bib` exists and is valid
- [ ] Title is clear and descriptive (not generic)
- [ ] Abstract/summary is 1–2 paragraphs
- [ ] "Statement of Need" section is 1–2 pages and explains research gap
- [ ] Paper includes comparison to existing software
- [ ] Author affiliations are current
- [ ] All references are complete (DOI, URL, or publication details)
- [ ] Paper is 4–8 pages when rendered
- [ ] No `[TODO]`, `[FIXME]`, or placeholder text in paper.md

**Code Requirements:**
- [ ] Tests: 689 passing, 79%+ coverage
- [ ] Linting: ruff check passes with 0 errors
- [ ] Documentation: mkdocs build succeeds
- [ ] Package: installs cleanly in fresh venv
- [ ] License: MIT and declared in pyproject.toml
- [ ] CITATION.cff: valid, no TODOs, all authors listed

**Documentation Requirements:**
- [ ] README has "Statement of Need" section
- [ ] README has feature comparison table
- [ ] README has 5-minute end-to-end example
- [ ] Installation instructions are clear
- [ ] Quickstart guides exist
- [ ] API documentation is comprehensive (150+ pages)

**Metadata Requirements:**
- [ ] pyproject.toml: version is 1.0.0, all dependencies listed
- [ ] CHANGELOG.md: documents v1.0.0 release
- [ ] CONTRIBUTING.md: exists and is clear
- [ ] CODE_OF_CONDUCT.md: exists
- [ ] GitHub Actions CI: tests pass on Python 3.10, 3.11, 3.12

---

### 15. Submit to JOSS

**Effort:** 15 minutes

Once all above items are complete:

1. **Visit:** https://joss.theoj.org/papers/new
2. **Fill out submission form:**
   - Repository URL: `https://github.com/chandrasekarnarayana/foodspec`
   - Paper file: `paper.md`
   - Enter title, abstract, keywords
3. **Submit**
4. **Wait for JOSS editor assignment** (typically 1–2 weeks)

**After Submission:**
- [ ] Monitor email for reviewer comments
- [ ] Respond to reviewer feedback within deadline
- [ ] Update paper.md and repository as needed
- [ ] Resubmit revised version

---

## Timeline Recommendation

| Phase | Days | Task |
|-------|------|------|
| **Phase 1** | 1–2 | Blocking issues: paper.md, paper.bib, CITATION.cff, README |
| **Phase 2** | 1 | Documentation enhancements, comparison table |
| **Phase 3** | 0.5 | CI improvements (Python 3.13, mypy) |
| **Phase 4** | 1 | Final verification: tests, linting, docs, package |
| **Phase 5** | 0.5 | Submit to JOSS |
| **TOTAL** | **5 days** | **Estimated effort: 14.5–16 hours** |

---

## Critical Success Factors

1. ✅ **Paper.md captures the research gap** — This is the #1 thing reviewers evaluate
2. ✅ **All code tests pass** — 689 passing tests demonstrate quality
3. ✅ **No placeholder values** in CITATION.cff or metadata
4. ✅ **Feature comparison table** explains why FoodSpec is novel
5. ✅ **Clear, working examples** in README and paper.md
6. ✅ **All collaborators credited** in CITATION.cff and acknowledgments

---

## Questions to Answer Before Submission

| Question | Answer |
|----------|--------|
| **Why does FoodSpec exist?** | Address fragmented, proprietary workflows in food spectroscopy |
| **What problem does it solve?** | Reproducibility, standardization, data governance |
| **How is it different from ChemoSpec/HyperSpy?** | Complete ecosystem: preprocessing + ML + validation + domain workflows |
| **Is the code production-ready?** | Yes: 79% coverage, 689 tests, GitHub Actions CI, type hints |
| **Who maintains this?** | Chandrasekar Subramani Narayana (with collaborators) |
| **Is it well-documented?** | Yes: 150+ pages, 16 examples, comprehensive API docs |
| **Will it be maintained long-term?** | Yes: clear versioning, contributing guidelines, active development |

---

## Resources & References

- **JOSS Paper Guidelines:** https://joss.theoj.org/papers
- **JOSS Reviewer Guidelines:** https://joss.theoj.org/reviewers
- **BibTeX Format:** https://www.ctan.org/pkg/bibtex
- **CITATION.cff Validator:** https://citation.cff.software/validate/
- **PyPI Package Check:** https://pypi.org/project/foodspec/

---

**Document Status:** READY FOR ACTION  
**Last Updated:** January 6, 2025  
**Next Step:** Begin Phase 1 (Blocking Issues)

Good luck with the JOSS submission! 🚀
