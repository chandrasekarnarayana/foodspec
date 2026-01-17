# FoodSpec Changelog Policy

**Version:** 1.0  
**Effective Date:** January 6, 2026  
**Scope:** All releases from v1.0.0 onwards

---

## 1. Format and Standards

FoodSpec follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

### Version Numbers
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes to public API or file formats
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

### Release Entry Structure
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality

### Deprecated
- Features to be removed in future versions

### Removed
- Features removed in this version

### Fixed
- Bug fixes

### Security
- Vulnerability patches
```

---

## 2. Category Definitions for FoodSpec

### 📦 Added
**New functionality that extends capabilities without altering existing behavior.**

Qualifies:
- New preprocessing methods (e.g., new baseline correction algorithm)
- New file format support (e.g., vendor reader for new instrument)
- New statistical tests or chemometric models
- New CLI commands or API classes
- New protocol step types
- New domain workflows (e.g., dairy analysis)
- New datasets in `foodspec.data`
- New documentation sections (if substantial, >500 words)

Does NOT qualify:
- Internal refactoring without API changes
- Test improvements
- Documentation clarifications (<500 words)

**Marking conventions:**
- 🎓 `[Experimental]` - API may change in next minor release
- 📚 `[Docs-only]` - Documentation addition without code changes
- 🔬 `[Research]` - Research-grade feature (not production-validated)

**Example:**
```markdown
### Added
- 🔬 New OPLS-DA algorithm for discriminant analysis (api/chemometrics)
- 📚 Tutorial: "Meat Authentication with Hyperspectral Imaging"
- CSV export for PCA loadings via `pca_results.to_csv()`
```

---

### 🔄 Changed
**Modifications to existing features that remain backward-compatible.**

Qualifies:
- Performance improvements (>10% speedup)
- Default parameter changes (if non-breaking)
- Output format enhancements (e.g., adding optional fields)
- Refactored internals with identical public API
- Improved error messages
- CLI output formatting changes
- Documentation restructuring

Does NOT qualify:
- Bug fixes (use **Fixed**)
- Breaking changes (mark with ⚠️, may trigger MAJOR bump)

**Marking conventions:**
- ⚡ `[Performance]` - >10% speedup or memory reduction
- 🔁 `[Reproducibility]` - Changes that affect numerical outputs (even if better)
- ⚙️ `[Internal]` - Refactoring with no API changes

**Example:**
```markdown
### Changed
- ⚡ [Performance] Vectorized peak detection: 3x faster on 1000+ spectra
- 🔁 [Reproducibility] Updated ALS baseline to use SciPy 1.11 solver (results may differ by <0.1%)
- Default `n_components` for PCA changed from 5 to "auto" (90% variance explained)
```

---

### ⚠️ Deprecated
**Features marked for removal in a future MAJOR version.**

Qualifies:
- Modules moved to new locations
- Parameters renamed or replaced
- File format versions being phased out
- CLI commands being consolidated

**Deprecation timeline:**
- Deprecated in v1.x → Removed in v2.0
- Deprecation warnings logged at runtime
- Migration guide MUST be provided

**Marking conventions:**
- ⏱️ `[Until vX.0.0]` - Removal target version
- 📖 `[Migration: see docs/...]` - Link to migration guide

**Example:**
```markdown
### Deprecated
- ⏱️ [Until v2.0.0] `foodspec.artifact.save_artifact()` → Use `foodspec.deploy.save_artifact()`
- ⏱️ [Until v2.0.0] `baseline_method="als_old"` → Use `baseline_method="als"` (📖 Migration: see docs/user-guide/migration_v1_to_v2.md)
```

---

### 🗑️ Removed
**Features removed in this version (triggers MAJOR version bump).**

Qualifies:
- Removal of previously deprecated features
- Dropped support for Python versions
- Removed file format support
- Deleted CLI commands

**Requirements:**
- MUST have been deprecated in previous MINOR release
- MUST include migration path in docs

**Marking conventions:**
- ⚠️ `[BREAKING]` - All removals are breaking changes
- 🔄 `[Replaced by: ...]` - Indicate replacement API

**Example:**
```markdown
### Removed
- ⚠️ [BREAKING] Dropped Python 3.8 support (minimum now 3.9)
- ⚠️ [BREAKING] Removed `foodspec.rq` module (🔄 Replaced by: `foodspec.features.rq`)
```

---

### 🐛 Fixed
**Bug fixes that restore intended behavior.**

Qualifies:
- Crashes or exceptions
- Incorrect calculations or outputs
- Memory leaks
- File I/O errors
- Broken links or rendering in documentation
- CLI argument parsing issues
- Test failures

Does NOT qualify:
- Enhancements (use **Changed** or **Added**)
- Performance improvements (use **Changed** with ⚡)

**Marking conventions:**
- 🔢 `[Accuracy]` - Fixes to numerical correctness
- 📄 `[Docs]` - Documentation fixes
- 🖥️ `[CLI]` - Command-line interface fixes

**Example:**
```markdown
### Fixed
- 🔢 [Accuracy] Corrected Savitzky-Golay edge handling (was using 'nearest', now 'mirror')
- 📄 [Docs] Fixed broken links in 12 preprocessing method pages
- Crash when loading HDF5 files with missing metadata (Issue #42)
```

---

### 🔒 Security
**Patches for vulnerabilities (may trigger PATCH bump even without other changes).**

Qualifies:
- Dependency updates for CVEs
- Input validation fixes preventing injection
- Authentication/authorization fixes
- Data exposure vulnerabilities

**Marking conventions:**
- 🚨 `[CVE-YYYY-XXXXX]` - Reference CVE number if applicable
- 🔐 `[Severity: High/Medium/Low]` - CVSS-based severity

**Example:**
```markdown
### Security
- 🚨 [CVE-2024-12345] Updated NumPy to 1.24.4 (🔐 Severity: High)
- Added input sanitization for file paths in `load_library()` to prevent directory traversal
```

---

## 3. Special Change Types

### 📚 Documentation-Only Changes
**When to mention in changelog:**
- Major section additions (>500 words) → **Added** with 📚
- Restructuring/reorganization → **Changed**
- Broken link fixes → **Fixed** with 📄

**When to skip:**
- Typo fixes
- Docstring clarifications
- Minor formatting changes

### ⚡ Performance Changes
**Must document if:**
- >10% speedup or memory reduction
- Changes algorithmic complexity (e.g., O(n²) → O(n log n))

**Category:** **Changed** with ⚡ marker

**Example:**
```markdown
### Changed
- ⚡ [Performance] PCA computation now uses randomized SVD: 10x faster on high-dimensional data (>1000 features)
```

### 🔁 Reproducibility Changes
**Changes that alter numerical outputs (even if improvements).**

**Must document if:**
- Algorithm updates (e.g., SciPy solver version)
- Random seed handling changes
- Floating-point precision adjustments
- Rounding/truncation behavior

**Category:** **Changed** with 🔁 marker  
**Requirement:** Quantify typical magnitude of change (e.g., "<0.1% difference")

**Example:**
```markdown
### Changed
- 🔁 [Reproducibility] ALS baseline now uses sparse matrices: results differ by <0.01% but 2x faster
```

---

## 4. Breaking Change Markers

### When is a change BREAKING?

**API Breaking:**
- Removing public functions/classes
- Renaming parameters
- Changing return types
- Changing default behavior significantly

**Data Breaking:**
- HDF5 schema changes (incompatible reads)
- Output file format changes

**Dependency Breaking:**
- Dropping Python version support
- Requiring new system dependencies

### How to Mark
1. Prefix with ⚠️ `[BREAKING]`
2. Place in appropriate category (**Changed**, **Removed**, **Fixed**)
3. Add migration note with 📖 icon

**Example:**
```markdown
### Changed
- ⚠️ [BREAKING] `make_classifier(model="svm")` now requires explicit `kernel=` parameter (📖 Migration: see docs/user-guide/migration_v2.md)
```

---

## 5. Acceptance Criteria for Release Entries

### Checklist for Each Release

**Before adding to CHANGELOG.md:**

#### Content Completeness
- [ ] All user-visible changes documented
- [ ] Breaking changes marked with ⚠️ `[BREAKING]`
- [ ] Reproducibility impacts marked with 🔁
- [ ] Performance changes quantified (e.g., "3x faster")
- [ ] Security fixes marked with 🔒
- [ ] Deprecations include removal target version

#### Cross-References
- [ ] API changes link to documentation (e.g., `api/chemometrics`)
- [ ] Breaking changes link to migration guide
- [ ] Fixed issues reference GitHub issue numbers (e.g., `#42`)
- [ ] New features link to examples or tutorials

#### Scientific Integrity
- [ ] Reproducibility changes quantify magnitude of difference
- [ ] Algorithm changes reference papers/sources if applicable
- [ ] Statistical test changes note assumption differences

#### Format Compliance
- [ ] Version follows SemVer (X.Y.Z)
- [ ] Date in ISO format (YYYY-MM-DD)
- [ ] Categories in order: Added → Changed → Deprecated → Removed → Fixed → Security
- [ ] Bullet points use consistent style (start with verb or noun phrase)

#### Audience Appropriateness
- [ ] Language accessible to food scientists (not just developers)
- [ ] Avoid jargon where possible (or explain on first use)
- [ ] Impact described ("what does this mean for users?")

---

## 6. Example Full Release Entry

```markdown
## [1.2.0] - 2026-01-15

### Added
- 🔬 [Experimental] OPLS-DA (Orthogonal PLS Discriminant Analysis) for improved classification (api/chemometrics)
- CSV export for confusion matrices via `results.confusion_matrix.to_csv()` (Issue #67)
- 📚 Tutorial: "Dairy Product Authentication with FTIR" (docs/examples/06_dairy_authentication.md)
- Support for Shimadzu IRSpirit file format (.spc)

### Changed
- ⚡ [Performance] Peak detection 5x faster using Cython-accelerated scipy.signal.find_peaks
- 🔁 [Reproducibility] Updated MSC (Multiplicative Scatter Correction) reference spectrum calculation to use median instead of mean (typical difference: <0.5%)
- Default `cv` parameter in `make_classifier()` changed from 5 to 10 for more robust validation
- CLI progress bars now show ETA for long operations

### Deprecated
- ⏱️ [Until v2.0.0] `baseline_correction(..., method="polynomial_old")` → Use `method="polynomial"` (📖 Migration: see docs/user-guide/migration_v1_to_v2.md)

### Fixed
- 🔢 [Accuracy] Corrected Savitzky-Golay filter at spectrum edges (was extrapolating incorrectly, now uses 'mirror' mode)
- 📄 [Docs] Fixed 8 broken cross-references in preprocessing method pages
- Crash when processing empty spectral datasets (now raises descriptive ValueError)
- HDF5 files >2GB now load correctly on Windows (Issue #71)

### Security
- 🚨 Updated Pillow to 10.1.0 to address image processing vulnerability

**Migration Notes:** See docs/user-guide/migration_v1.1_to_v1.2.md

**Contributors:** @chandrasekarnarayana, @contributor2

[1.2.0]: https://github.com/chandrasekarnarayana/foodspec/releases/tag/v1.2.0
```

---

## 7. Workflow Integration

### When to Update CHANGELOG.md
- **During PR review** for significant changes
- **Before release tagging** (final review)
- **After release** (add link to GitHub release)

### Who Updates
- **Developers** add entries during feature development
- **Maintainers** review and reorganize before release
- **Release manager** finalizes and adds comparison link

### Tools
- Use `git log --oneline v1.1.0..v1.2.0` to audit changes
- Use `git diff v1.1.0 docs/` to identify doc changes
- Use pytest coverage reports to identify test improvements

---

## 8. Version Bump Decision Tree

```
Does this change...
├─ Remove public APIs or break backward compatibility? → MAJOR (X.0.0)
├─ Add new features (backward-compatible)? → MINOR (x.Y.0)
├─ Fix bugs only (no new features)? → PATCH (x.y.Z)
└─ Documentation/test-only changes? → No version bump (or PATCH if bundled)
```

---

**Policy Maintained By:** FoodSpec Core Team  
**Review Cycle:** Annually or before MAJOR releases  
**Questions:** Open issue with label `documentation/changelog`
