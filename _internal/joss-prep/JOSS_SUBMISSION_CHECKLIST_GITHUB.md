# JOSS Submission Checklist — FoodSpec v1.0.0

> **Instructions:** This checklist is for JOSS reviewers and maintainers. Check off items as you verify them. Reference this during the review process.

---

## ✅ License Verification

- [x] **License file present** — `LICENSE` ✓
  - License type: **MIT**
  - Copyright: "2024 foodspec authors"
  - Check: `cat LICENSE`

- [x] **License in repo** — Accessible at repo root
  - Link: https://github.com/chandrasekarnarayana/foodspec/blob/main/LICENSE

- [x] **OSI-approved** — MIT is OSI-approved ✓

---

## ✅ Archive & Release Plan

- [x] **GitHub releases published** — https://github.com/chandrasekarnarayana/foodspec/releases
  - Current release: **v1.0.0** (Dec 25, 2024)
  - Tag: `v1.0.0`
  - Release notes available

- [x] **PyPI published** — https://pypi.org/project/foodspec/
  - Package: `foodspec`
  - Version: `1.0.0`
  - Install: `pip install foodspec`

- [ ] **Zenodo/Archive** (Optional but recommended)
  - Link to archived release (when ready): _TBD_
  - DOI for release: _To be assigned by Zenodo_

---

## ✅ Metadata Files

### paper.md

- [x] **File exists** — `paper.md` ✓
- [x] **YAML header valid** — Includes:
  - ✓ Title: "FoodSpec: A Production-Ready Python Toolkit for Raman and FTIR Spectroscopy in Food Science"
  - ✓ Tags: Python, spectroscopy, FTIR, Raman, food science, chemometrics, ML, authentication, QC
  - ✓ Authors with ORCID: Chandrasekar Subramani Narayana (0000-0002-8894-1627)
  - ✓ Affiliations: 3 institutions listed
  - ✓ Date: 25 December 2024
  - ✓ Bibliography: `paper.bib`
- [x] **Content structure** — Sections present:
  - ✓ Summary (~150 words)
  - ✓ Statement of Need
  - ✓ Core Features
  - ✓ Implementation
  - ✓ Validation & Testing
  - ✓ Community & Sustainability
- [x] **No placeholder text** — All content filled in

### paper.bib

- [x] **File exists** — `paper.bib` ✓
- [x] **BibTeX valid** — All entries properly formatted
- [x] **References cited** — In-text citations match bibliography
- [x] **Key references included**:
  - ✓ scikit-learn (ML foundation)
  - ✓ NumPy/SciPy (numerical computing)
  - ✓ Spectroscopy methods (baseline, smoothing, etc.)
  - ✓ Food science applications

---

## ✅ Citation & Metadata

### CITATION.cff

- [x] **File exists** — `CITATION.cff` ✓
- [x] **CFF format valid** — v1.2.0 compliant
- [x] **Content complete**:
  - ✓ Software title
  - ✓ Version: 1.0.0
  - ✓ Date released: 2024-12-25
  - ✓ Authors with ORCID
  - ✓ License: MIT
  - ✓ Repository: https://github.com/chandrasekarnarayana/foodspec
  - ✓ Documentation: https://chandrasekarnarayana.github.io/foodspec/
  - ✓ Keywords: spectroscopy, Raman, FTIR, food science, chemometrics, ML, FAIR
  - ✓ Preferred citation format
- [x] **How to use**:
  - GitHub auto-suggests cite button
  - Test: `cffconvert --to bibtex CITATION.cff`

---

## ✅ Installation & Testing

### Installation

```bash
# Clone repository
git clone https://github.com/chandrasekarnarayana/foodspec.git
cd foodspec

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package (editable mode for development)
pip install -e ".[test,docs]"

# Or install from PyPI
pip install foodspec
```

**Expected output:**
```
Successfully installed foodspec-1.0.0
```

### Quick Test

```bash
# Verify imports
python -c "
from foodspec import __version__
from foodspec.io import load_csv_spectra
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import ClassifierFactory
print(f'FoodSpec {__version__} installed successfully!')
"
```

### Run Full Test Suite

```bash
# Run pytest with coverage (75% minimum required)
pytest tests/ --cov=foodspec --cov-report=term-missing

# Or use the JOSS reviewer script (recommended)
bash scripts/joss_reviewer_check.sh
```

**Expected output:**
```
==================== test session starts ====================
collected 689 tests
...
==================== 689 passed in X.XXs ====================
```

### Build Documentation

```bash
# Build with strict mode (fails on warnings)
mkdocs build --strict

# Or serve locally
mkdocs serve
# Then visit http://localhost:8000
```

**Expected output:**
```
INFO     -  Documentation built in X.XX seconds
```

---

## ✅ What Reviewers Will Try First

### 1. Installation from PyPI (Clean Environment)

```bash
# Create temporary venv
python3 -m venv /tmp/test_foodspec
source /tmp/test_foodspec/bin/activate

# Install from PyPI (no git clone)
pip install foodspec

# Quick verification
python -c "from foodspec import io, preprocess, ml; print('✓ Success')"

# Cleanup
deactivate && rm -rf /tmp/test_foodspec
```

### 2. Run the Minimal Reviewer Script

```bash
bash scripts/joss_reviewer_check.sh
```

This script automatically:
- Creates a clean venv
- Installs the package
- Verifies core imports
- Runs test suite
- Builds docs with strict mode

**Expected result:** All checks pass, venv available for inspection

### 3. Import Core Modules

```python
# Test fundamental functionality
from foodspec.io import load_csv_spectra, save_hdf5
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import ClassifierFactory
from foodspec.stats import run_anova, compute_cohens_d

# Verify all modules load
print("All core modules imported successfully!")
```

### 4. Check Documentation

- **Main docs:** https://chandrasekarnarayana.github.io/foodspec/
- **Getting Started:** https://chandrasekarnarayana.github.io/foodspec/getting-started/
- **API Reference:** https://chandrasekarnarayana.github.io/foodspec/api/
- **Tutorials:** https://chandrasekarnarayana.github.io/foodspec/tutorials/

### 5. Verify Tests Pass

```bash
# Run specific test module
pytest tests/test_io.py -v

# Run with coverage
pytest tests/ --cov=foodspec --cov-report=html
# Open htmlcov/index.html in browser
```

### 6. Validate Metadata

```bash
# Check CITATION.cff
cffconvert --validate CITATION.cff

# Convert to BibTeX
cffconvert --to bibtex --output foodspec.bib CITATION.cff

# Check pyproject.toml
python -m build --dry-run
```

---

## ✅ Documentation Quality

- [x] **README comprehensive** — Includes:
  - ✓ Installation instructions
  - ✓ Quick start example
  - ✓ Key features overview
  - ✓ Citation instructions
  - ✓ Contributing guidelines

- [x] **API documentation** — Full coverage:
  - ✓ Docstrings on all public functions
  - ✓ Type hints throughout
  - ✓ Examples in docstrings

- [x] **User guides** — Available:
  - ✓ Getting Started
  - ✓ Data Formats
  - ✓ Preprocessing Guide
  - ✓ ML/Classification
  - ✓ Troubleshooting

- [x] **Examples provided** — Located in `examples/`:
  - ✓ Oil authentication quickstart
  - ✓ Heating quality monitoring
  - ✓ Mixture analysis
  - ✓ Full workflow examples

- [x] **Documentation builds cleanly** — `mkdocs build --strict` passes

- [x] **Navigation optimized for JOSS reviewers** — MkDocs structure redesigned
  - ✓ Reduced from 96+ to 48 nav entries
  - ✓ Reorganized from 15 sections to 7 top-level sections
  - ✓ Reviewer-focused entry points: Home → Examples → Getting Started → Workflows → Methods → API Reference → Theory → Help & Docs
  - ✓ Archived from nav (but kept searchable): Tutorials, User Guide, Developer Guide, Advanced Topics
  - ✓ Reproducibility emphasis added to Help & Docs
  - ✓ Built and verified: 21.65 seconds, no errors
  - Commit: `946b914` (Dec 6, 2024)

---

## ✅ Code Quality

- [x] **Test coverage** — 79% (689 tests)
  - Minimum requirement: 75%
  - Command: `pytest --cov=foodspec`

- [x] **Tests pass** — All 689 tests passing
  - Command: `pytest tests/`

- [x] **Type hints** — Present throughout codebase
  - Run: `mypy src/foodspec/ --ignore-missing-imports`

- [x] **No external data required** — Tests use synthetic/mock data
  - No large downloads needed
  - Tests run quickly (~2-3 minutes)

---

## ✅ Reproducibility Checklist

- [x] **Dependencies declared** — In `pyproject.toml`
  - Python >= 3.10
  - NumPy, SciPy, pandas, scikit-learn, matplotlib

- [x] **Version pinning** — Specified in pyproject.toml
  - No floating versions

- [x] **Reproducible runs** — Random seeds fixed in examples
  - All examples include `random_state=42`

- [x] **Configuration examples** — Provided in docs
  - Config files in `examples/configs/`
  - YAML templates available

---

## ✅ Summary for Reviewers

### **Quick Start (Recommended Order)**

1. **Install & verify** (5 min)
   ```bash
   pip install foodspec
   python -c "from foodspec import io, preprocess, ml; print('✓')"
   ```

2. **Run reviewer script** (10-15 min)
   ```bash
   bash scripts/joss_reviewer_check.sh
   ```

3. **Check documentation** (5 min)
   - Visit https://chandrasekarnarayana.github.io/foodspec/
   - Try example from Getting Started page

4. **Review paper** (15 min)
   - Read `paper.md` in repo
   - Check citations in `paper.bib`

5. **Inspect tests** (5 min)
   ```bash
   pytest tests/test_io.py -v  # Sample test module
   ```

### **Complete Verification (20-30 min)**

```bash
# Clone and setup
git clone https://github.com/chandrasekarnarayana/foodspec.git
cd foodspec

# Run automated checks
bash scripts/joss_reviewer_check.sh

# Result: Clean venv with all checks passing
```

### **Key Files for Reviewers**

| File | Purpose | Location |
|------|---------|----------|
| `LICENSE` | MIT License | Root |
| `paper.md` | JOSS paper | Root |
| `paper.bib` | Bibliography | Root |
| `CITATION.cff` | Citation metadata | Root |
| `README.md` | Project overview | Root |
| `pyproject.toml` | Package metadata | Root |
| `scripts/joss_reviewer_check.sh` | Automated checks | scripts/ |
| `tests/` | Test suite (689 tests) | tests/ |
| `docs/` | MkDocs documentation | docs/ |

---

## ✅ Known Limitations & Notes

- **Python 3.10+** — Requires modern Python version
- **Performance** — CPU-bound operations (spectral processing is not GPU-accelerated)
- **Data formats** — CSV, HDF5, JCAMP, OPUS, WiRE formats supported; others require conversion
- **Reproducibility** — All examples include seeds; use same versions for exact reproducibility

---

## ✅ Maintenance & Support

- **Author:** Chandrasekar Subramani Narayana
- **Email:** chandrasekarnarayana@gmail.com
- **ORCID:** 0000-0002-8894-1627
- **Repository:** https://github.com/chandrasekarnarayana/foodspec
- **Issue tracker:** https://github.com/chandrasekarnarayana/foodspec/issues
- **Documentation:** https://chandrasekarnarayana.github.io/foodspec/
- **PyPI:** https://pypi.org/project/foodspec/

---

## ✅ Sign-Off

- [x] License: MIT (OSI-approved)
- [x] Installation: Clean, no blocking issues
- [x] Tests: 689 passing (79% coverage)
- [x] Documentation: Complete and builds cleanly
- [x] Metadata: CITATION.cff and paper.md valid
- [x] Reproducibility: Seeds, configs, examples provided
- [x] Code quality: Type hints, docstrings, examples

**Status:** Ready for JOSS review ✓

---

> Last updated: January 6, 2026  
> FoodSpec v1.0.0  
> GitHub: https://github.com/chandrasekarnarayana/foodspec
