# Documentation Definition of Done

**Purpose:** Quality checklist for documentation readiness before releases.  
**Last Updated:** January 6, 2026  
**Status:** Active

---

## Release Checklist

Use this checklist to verify documentation quality before each release. All items must pass for documentation to be considered "done."

---

### 🗂️ Structure & Organization

- [ ] **Navigation has no duplicates**
  - No page appears more than once in `mkdocs.yml` nav
  - Command: `grep -E "^\s+- [^:]+:" mkdocs.yml | sort | uniq -d`
  - Expected: Empty output (no duplicates)

- [ ] **No orphan non-archived pages**
  - All user-facing `.md` files either in nav or explicitly archived
  - Command: `python scripts/check_orphans.py` or manual audit
  - Expected: Only `_internal/` and `examples_gallery.md` (special case) allowed

- [ ] **Redirects cover all moved pages**
  - All historical URLs have redirects in `mkdocs.yml`
  - Check: Review `redirect_maps` section completeness
  - Expected: All Phase 1-3 migrations have redirects

---

### 🏗️ Build Quality

- [ ] **mkdocs build produces 0 warnings**
  - Command: `mkdocs build 2>&1 | grep -c "WARNING"`
  - Expected: `0`
  - Notes: Ignore INFO-level unrecognized links (external)

- [ ] **mkdocs build produces 0 errors**
  - Command: `mkdocs build 2>&1 | grep -c "ERROR"`
  - Expected: `0`

- [ ] **Build completes in reasonable time**
  - Command: `mkdocs build 2>&1 | tail -1`
  - Expected: < 15 seconds

---

### 🔗 Links & References

- [ ] **Link checker passes**
  - Command: `python scripts/check_docs_links.py`
  - Expected: "ALL CHECKS PASSED" (0 broken links)
  - Scope: All internal markdown links and image references

- [ ] **No broken anchors**
  - Command: `mkdocs build 2>&1 | grep "does not contain an anchor"`
  - Expected: Empty output (0 broken anchors)

- [ ] **External links functional** (manual spot check)
  - GitHub repo links work
  - External references accessible
  - Expected: Manual verification of key external links

---

### 📚 Content Completeness

- [ ] **5+ flagship examples exist and are linked**
  - Examples:
    1. Oil Authentication
    2. Heating Quality Monitoring
    3. Mixture Analysis
    4. Hyperspectral Mapping
    5. End-to-End Protocol
  - Verification: Check `docs/examples/` directory
  - Linkage: Verify examples appear in nav and examples gallery

- [ ] **Examples have runnable code**
  - All example scripts in `examples/` directory executable
  - All teaching notebooks available
  - Expected: 5+ Python scripts + 5+ Jupyter notebooks

- [ ] **Key workflows documented**
  - Authentication workflow
  - Quality monitoring workflow
  - Quantification workflow
  - Harmonization workflow
  - Expected: 4+ workflow pages in `docs/workflows/`

---

### 📖 API Documentation

- [ ] **API reference is mkdocstrings-based**
  - All `docs/api/*.md` files use `:::` mkdocstrings directives
  - No manual copy-paste of docstrings
  - Expected: 11 API pages with mkdocstrings integration

- [ ] **API pages are minimal (no tutorials)**
  - Each API page: brief intro (1-2 paragraphs) + mkdocstrings
  - No extended tutorials or walkthroughs
  - Expected: API pages < 200 lines each

- [ ] **Public API docstrings coverage >= 95%**
  - All public functions/classes in `src/foodspec/__init__.py` have docstrings
  - Docstrings follow Google style
  - Expected: Manual audit or coverage tool >= 95%

- [ ] **Docstrings include examples**
  - All major public functions have code examples in docstrings
  - Examples are executable
  - Expected: Spot check 10+ public APIs for examples

---

### 🎨 Visuals & Assets

- [ ] **Figures have generator scripts**
  - All figures in `docs/assets/*.png` have corresponding scripts
  - Scripts located in `docs/examples/` or `scripts/`
  - Expected: No orphan figures without generation method

- [ ] **Figures are properly referenced**
  - All figures in `docs/assets/` used in at least one page
  - All `![image](...)` references resolve correctly
  - Expected: No unused assets, no missing images

- [ ] **Logo and branding consistent**
  - Logo appears in theme configuration
  - Favicon configured
  - Expected: Check `theme:` section in mkdocs.yml

---

### 🧪 Methods & Preprocessing

- [ ] **Preprocessing methods complete**
  - Baseline correction documented
  - Normalization documented
  - Derivatives documented
  - Scatter correction documented
  - Feature extraction documented
  - Expected: 5 preprocessing method pages

- [ ] **Each method page includes:**
  - "When to use" section
  - "When NOT to use" section
  - "Recommended defaults" section
  - "See also" links (API + Examples)
  - Expected: Verify all preprocessing pages have these sections

---

### 🧭 Navigation & Discovery

- [ ] **Decision guide exists**
  - Located at `docs/user-guide/decision_guide.md`
  - Includes mermaid flowchart
  - Links to methods, examples, and API
  - Expected: Single source for "how do I..."

- [ ] **Getting started flow complete**
  - Installation instructions
  - 15-minute quickstart
  - First steps guide
  - FAQ
  - Expected: 4+ pages in getting-started/

- [ ] **Cross-references complete**
  - Methods pages link to examples
  - Examples link to API reference
  - API pages link to methods
  - Expected: Spot check 10+ pages for cross-refs

---

### ✅ Validation & Testing

- [ ] **Tests pass**
  - Command: `pytest tests/`
  - Expected: All tests pass, coverage >= 75%

- [ ] **Examples run without errors**
  - Command: `python examples/*.py` (sample)
  - Expected: All flagship examples execute successfully

- [ ] **Notebooks execute**
  - Command: `jupyter nbconvert --execute notebooks/*.ipynb` (sample)
  - Expected: All teaching notebooks run to completion

---

### 📦 Packaging & Deployment

- [ ] **Version numbers consistent**
  - `pyproject.toml` version matches `src/foodspec/__init__.py`
  - CHANGELOG.md updated for version
  - Expected: Version sync across all files

- [ ] **CHANGELOG.md updated**
  - Current release documented
  - Migration notes if needed
  - Expected: Entry for current version

- [ ] **README.md accurate**
  - Installation instructions current
  - Links to documentation work
  - Quick example runs
  - Expected: README reflects current state

---

## Verification Commands

Run these commands to check each criterion:

```bash
# 1. Check for duplicate nav entries
grep -E "^\s+- [^:]+:" mkdocs.yml | sort | uniq -d

# 2. Count build warnings
mkdocs build 2>&1 | grep -c "WARNING"

# 3. Count build errors  
mkdocs build 2>&1 | grep -c "ERROR"

# 4. Check build time
mkdocs build 2>&1 | tail -1

# 5. Run link checker
python scripts/check_docs_links.py

# 6. Check for broken anchors
mkdocs build 2>&1 | grep "does not contain an anchor" | wc -l

# 7. Count example files
ls -1 docs/examples/*.md | wc -l

# 8. Count API pages
ls -1 docs/api/*.md | wc -l

# 9. Verify mkdocstrings usage in API
grep -c ":::" docs/api/*.md

# 10. Count preprocessing pages
ls -1 docs/methods/preprocessing/*.md | wc -l

# 11. Check decision guide exists
test -f docs/user-guide/decision_guide.md && echo "EXISTS" || echo "MISSING"

# 12. Run test suite
pytest tests/ -v --cov=foodspec --cov-report=term-missing

# 13. Check version consistency
grep "version" pyproject.toml src/foodspec/__init__.py
```

---

## Status Report Template

Use this template when reporting DoD status:

```markdown
## Documentation DoD Status Report

**Date:** YYYY-MM-DD
**Version:** X.Y.Z
**Reporter:** [Name]

### Passing ✅

- [x] Item 1
- [x] Item 2

### Failing ❌

- [ ] Item 3: [Description of issue]
- [ ] Item 4: [Description of issue]

### Action Items

1. Fix item 3: [Proposed solution]
2. Fix item 4: [Proposed solution]

### Sign-Off

Ready for release: YES / NO
```

---

## Exceptions & Notes

### Known Exceptions

1. **examples_gallery.md** - Not in nav by design (linked from examples/index.md)
2. **_internal/** - Maintainer docs, properly excluded
3. **External links** - May have INFO warnings (unrecognized paths), acceptable

### Quality Thresholds

- **Build warnings:** 0 (strict)
- **Build errors:** 0 (strict)
- **Broken links:** 0 (strict)
- **Docstring coverage:** >= 95% (target)
- **Test coverage:** >= 75% (required)
- **Build time:** < 15 seconds (performance)

---

## Maintenance

- **Review frequency:** Before each release
- **Update frequency:** When new quality criteria added
- **Owner:** Release manager + documentation maintainer
- **Last full audit:** 2026-01-06

---

## Related Documents

- [DOCS_MIGRATION_LEDGER.md](DOCS_MIGRATION_LEDGER.md) - Migration tracking
- [ORPHAN_PAGES.md](ORPHAN_PAGES.md) - Orphan analysis
- [CLEANUP_REPORT_2026-01-06.md](CLEANUP_REPORT_2026-01-06.md) - Latest cleanup
- [../developer-guide/documentation_guidelines.md](../developer-guide/documentation_guidelines.md) - Style guide
