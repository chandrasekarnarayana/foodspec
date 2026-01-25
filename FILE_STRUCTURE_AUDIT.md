# FoodSpec File Structure Audit & Reorganization Plan

**Audit Date**: January 25, 2026  
**Repository**: chandrasekarnarayana/foodspec  
**Branch**: main  
**Total Size**: ~75MB (excluding site/ and .git/)

---

## Executive Summary

### Critical Issues Found

1. **🔴 CRITICAL: Dual Source Trees** - Two complete implementations exist
   - `src/foodspec/` (3.4MB) - Legacy codebase
   - `foodspec_rewrite/foodspec/` (within 21MB) - New architecture
   
2. **🔴 CRITICAL: Duplicate pyproject.toml** - Two configuration files
   - `./pyproject.toml` (132 lines) - Root configuration
   - `./foodspec_rewrite/pyproject.toml` (91 lines) - Subdirectory configuration

3. **🟡 HIGH: Scattered Documentation** - 40+ phase/completion documents across 4 locations
   - Root level: 9 PHASE*.md files
   - `foodspec_rewrite/`: 17 phase documents
   - `docs/_internal/archive/`: 3 phase documents
   - `docs/developer-guide/`: 2 phase documents

4. **🟡 HIGH: Multiple Output Directories** - 6+ directories with demo/test outputs
   - `outputs/` (18MB) - Main output directory
   - `comparison_output/` (296KB)
   - `demo_runs/` (84KB)
   - `demo_export/` (136KB)
   - `demo_pdf_export/` (68KB)
   - `protocol_runs_test/` (900KB)
   - `foodspec_rewrite/outputs/` (within 21MB)

5. **🟡 MEDIUM: 642+ Cache Files** - __pycache__, .pytest_cache directories

6. **🟡 MEDIUM: Built Documentation** - `site/` directory (27MB) included in repo

---

## Detailed Inventory

### 1. Source Code Structure

#### Current State: DUAL IMPLEMENTATIONS

```
FoodSpec/
├── src/foodspec/              # 3.4MB - LEGACY CODEBASE
│   ├── apps/
│   ├── chemometrics/
│   ├── cli/
│   ├── core/
│   ├── data/
│   ├── deploy/
│   ├── features/
│   ├── io/
│   ├── ml/
│   ├── preprocess/
│   ├── protocol/
│   ├── qc/
│   ├── reporting/          # Has new modules mixed in
│   ├── trust/              # New architecture
│   ├── viz/                # New architecture
│   └── ... (22 modules)
│
└── foodspec_rewrite/          # 21MB - NEW ARCHITECTURE
    ├── foodspec/              # Complete reimplementation
    │   ├── cli/
    │   ├── core/
    │   ├── deploy/
    │   ├── features/
    │   ├── io/
    │   ├── models/
    │   ├── preprocess/
    │   ├── qc/
    │   ├── reporting/
    │   ├── trust/
    │   ├── validation/
    │   └── viz/
    ├── tests/                 # Separate test suite
    ├── examples/
    ├── docs/
    └── pyproject.toml         # Separate config!
```

**Problem**: Import ambiguity - which `foodspec` gets imported?

#### Import Resolution Analysis
```python
import foodspec  # Imports from: /home/cs/FoodSpec/foodspec_rewrite/foodspec/
                 # Because foodspec_rewrite/ is in sys.path first
```

**Current Behavior**:
- Package installed from `src/` (legacy)
- But `foodspec_rewrite/` shadows it during development
- Tests may run against wrong codebase

---

### 2. Configuration Files

#### Root Level Configuration (7 files)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `pyproject.toml` | 132 | Main package config | ✅ Active |
| `mkdocs.yml` | ? | Documentation config | ✅ Active |
| `.gitignore` | ? | Git ignore rules | ✅ Active |
| `.markdownlint.json` | ? | Markdown linting | ✅ Active |
| `CITATION.cff` | ? | Citation metadata | ✅ Active |
| `foodspec_rewrite/pyproject.toml` | 91 | Rewrite config | ⚠️ DUPLICATE |

**Problem**: Two `pyproject.toml` files can cause pip confusion

---

### 3. Documentation Structure

#### Phase/Completion Documents (40 files)

**Root Level** (9 files):
```
./PHASE10_PROMPT17_COMPLETION.md
./PHASE11_REPORTING_COMPLETION.md
./PHASE12_PAPER_PRESETS_COMPLETION.md
./PHASE13_DOSSIER_COMPLETION.md
./PHASE_14_SUMMARY.md
./PHASE_1_IMPLEMENTATION.md
./PHASE_7_SUMMARY.md
./PHASE8_COMPLETION_REPORT.md
./PHASE9_COMPLETION_REPORT.md
```

**foodspec_rewrite/** (17 files):
```
foodspec_rewrite/IMPLEMENTATION_COMPLETE.md
foodspec_rewrite/IMPLEMENTATION_SUMMARY.md
foodspec_rewrite/PHASE_1_RELIABILITY_SUMMARY.md
foodspec_rewrite/PHASE_2_1_COMPLETE.md
foodspec_rewrite/PHASE_2_2_COMPLETE.md
foodspec_rewrite/PHASE_2_2_FINAL_SUMMARY.md
foodspec_rewrite/PHASE_2_CALIBRATION_SUMMARY.md
foodspec_rewrite/PHASE3_METRICS_SUMMARY.md
foodspec_rewrite/PHASE4_EVALUATION_SUMMARY.md
foodspec_rewrite/PHASE5_NESTED_CV_SUMMARY.md
foodspec_rewrite/PHASE6_COMPLETION_REPORT.md
foodspec_rewrite/PHASE6_SUMMARY.md
foodspec_rewrite/PHASE7_COMPLETION_REPORT.md
foodspec_rewrite/PHASE_8B_COMPLETION_REPORT.md
foodspec_rewrite/PHASE_8B_SUMMARY.md
foodspec_rewrite/PHASE_9_COMPLETION_REPORT.md
foodspec_rewrite/PHASE_9_SUMMARY.md
```

**docs/** (14+ architecture/design files):
```
foodspec_rewrite/ARCHITECTURE.md
foodspec_rewrite/DRIFT_VISUALIZATIONS.md
foodspec_rewrite/IMPLEMENTATION_CHECKLIST.md
foodspec_rewrite/INTERPRETABILITY_VISUALIZATIONS.md
foodspec_rewrite/PARAMETER_LINEAGE_VISUALIZATIONS.md
foodspec_rewrite/PIPELINE_DAG_VISUALIZER.md
foodspec_rewrite/REPLICATE_TEMPORAL_VISUALIZATIONS.md
foodspec_rewrite/REPORTING_BASE_IMPLEMENTATION.md
foodspec_rewrite/TEST_SUITE_COMPLETION_REPORT.md
foodspec_rewrite/TRUST_IMPLEMENTATION_SUMMARY.md
foodspec_rewrite/TRUST_SUBSYSTEM_COMPLETE.md
foodspec_rewrite/TRUST_VISUALIZATION_COMPLETE.md
foodspec_rewrite/VISUALIZATION_SUITE_SUMMARY.md
foodspec_rewrite/WARNING_SUPPRESSION_GUIDE.md
```

**Problem**: No single source of truth for implementation status

---

### 4. Output & Demo Directories

#### Test/Demo Output Directories (6+ locations)

| Directory | Size | Files | Purpose | Issue |
|-----------|------|-------|---------|-------|
| `outputs/` | 18MB | Many | Example outputs | Should be .gitignore |
| `comparison_output/` | 296KB | 5 | Multi-run comparison demo | Temporary |
| `demo_runs/` | 84KB | 16 | Demo analysis runs | Temporary |
| `demo_export/` | 136KB | Many | Export demo outputs | Temporary |
| `demo_pdf_export/` | 68KB | 9 | PDF export demo | Temporary |
| `protocol_runs_test/` | 900KB | 14 runs | Protocol testing | Should be in tests/ |
| `foodspec_rewrite/outputs/` | ? | Many | Rewrite outputs | Duplicate |

**Problem**: Mixing ephemeral outputs with version-controlled code

---

### 5. Built Documentation

#### MkDocs Site (27MB)

```
site/                          # 27MB - BUILT DOCUMENTATION
├── 404.html
├── api/
├── assets/
├── concepts/
├── datasets/
├── design/
├── developer-guide/
├── examples/
├── getting-started/
├── help/
├── methods/
├── metrics/
├── reference/
├── theory/
├── troubleshooting/
├── tutorials/
├── user-guide/
├── visualization/
└── workflows/
```

**Problem**: Built documentation committed to repo (should be in .gitignore)

---

### 6. Cache Files

#### Python & Test Caches (642 files)

```
./src/foodspec/**/__pycache__/
./foodspec_rewrite/**/__pycache__/
./tests/**/__pycache__/
./.pytest_cache/
./foodspec_rewrite/.pytest_cache/
./.ruff_cache/
./.benchmarks/
./foodspec_rewrite/.benchmarks/
./.foodspec_cache/
./foodspec_rewrite/.foodspec_cache/
.coverage
```

**Problem**: Should all be in .gitignore

---

### 7. Root-Level Markdown Files (20 files)

#### Current Root Files

**Essential** (Keep at root):
- ✅ `README.md` - Main project README
- ✅ `LICENSE` - MIT license
- ✅ `CHANGELOG.md` - Version history
- ✅ `CONTRIBUTING.md` - Contribution guidelines
- ✅ `CODE_OF_CONDUCT.md` - Code of conduct
- ✅ `CITATION.cff` - Citation metadata

**Release/Deployment** (Keep at root):
- ✅ `DEPLOYMENT_SUMMARY_v1.1.0-rc1.md` - Deployment record
- ✅ `RELEASE_CHECKLIST_v1.1.0-rc1.md` - Release process
- ✅ `RELEASE_NOTES_v1.0.0.md` - Historical release
- ✅ `BRANCH_MIGRATION_PLAN.md` - Migration strategy

**JOSS Submission** (Keep at root or move to docs/):
- ⚠️ `paper.md` - JOSS paper (keep at root per JOSS requirements)
- ⚠️ `paper.bib` - Bibliography (keep at root)
- ⚠️ `JOSS_DOCS_AUDIT_REPORT.md` - Move to `_internal/joss-prep/`
- ⚠️ `JOSS_SUBMISSION_CHECKLIST.md` - Move to `_internal/joss-prep/`

**Phase Documents** (Move to archive):
- 🔄 `PHASE*.md` (9 files) - Move to `_internal/phase-history/`

**Utility Files** (Delete or archive):
- ⚠️ `file_structure.txt` - Outdated snapshot, delete or regenerate

---

## Proposed Reorganization

### Phase 1: Immediate Cleanup (High Priority)

#### 1.1 Remove Duplicate foodspec_rewrite/ Directory

**Rationale**: Merge completed, no longer needed

**Actions**:
```bash
# Archive important docs first
mkdir -p _internal/phase-history/architecture-docs
mv foodspec_rewrite/*.md _internal/phase-history/architecture-docs/
mv foodspec_rewrite/docs/*.md _internal/phase-history/architecture-docs/

# Remove foodspec_rewrite/ entirely
git rm -r foodspec_rewrite/
```

**Impact**: 
- ✅ Eliminates import ambiguity
- ✅ Reduces repo size by 21MB
- ✅ Removes duplicate pyproject.toml
- ✅ Single source tree: `src/foodspec/`

#### 1.2 Update .gitignore

**Add to .gitignore**:
```gitignore
# Build outputs
site/
dist/
build/
*.egg-info/

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
.coverage
.coverage.*
htmlcov/

# Application cache
.foodspec_cache/
.benchmarks/

# Demo/test outputs
outputs/
demo_*/
comparison_output/
protocol_runs_test/
*_runs/
*_output/
*_export/

# Temporary files
*.tmp
*.log
.DS_Store
```

#### 1.3 Clean Existing Untracked Files

```bash
# Remove all cache directories
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
rm -rf .ruff_cache/
rm -rf .benchmarks/
rm -rf .foodspec_cache/
rm -rf site/

# Remove demo outputs
rm -rf outputs/
rm -rf comparison_output/
rm -rf demo_runs/
rm -rf demo_export/
rm -rf demo_pdf_export/
rm -rf protocol_runs_test/
```

---

### Phase 2: Documentation Consolidation

#### 2.1 Reorganize Phase Documents

**Create Archive Structure**:
```
_internal/
└── phase-history/
    ├── README.md                    # Index of all phases
    ├── phase-1-8/                   # Numbered phases
    │   ├── PHASE_1_IMPLEMENTATION.md
    │   ├── PHASE_7_SUMMARY.md
    │   ├── PHASE8_COMPLETION_REPORT.md
    │   ├── PHASE9_COMPLETION_REPORT.md
    │   ├── PHASE10_PROMPT17_COMPLETION.md
    │   ├── PHASE11_REPORTING_COMPLETION.md
    │   ├── PHASE12_PAPER_PRESETS_COMPLETION.md
    │   ├── PHASE13_DOSSIER_COMPLETION.md
    │   └── PHASE_14_SUMMARY.md
    ├── architecture-docs/           # Technical specs
    │   ├── ARCHITECTURE.md
    │   ├── IMPLEMENTATION_CHECKLIST.md
    │   ├── IMPLEMENTATION_COMPLETE.md
    │   └── ... (all foodspec_rewrite/*.md)
    ├── subsystem-docs/              # Component docs
    │   ├── trust/
    │   ├── reporting/
    │   ├── visualization/
    │   └── validation/
    └── joss-prep/                   # JOSS submission docs
        ├── JOSS_DOCS_AUDIT_REPORT.md
        ├── JOSS_SUBMISSION_CHECKLIST.md
        └── ... (existing joss-prep/)
```

**Actions**:
```bash
# Create structure
mkdir -p _internal/phase-history/{phase-1-8,architecture-docs,subsystem-docs,joss-prep}

# Move root-level phase docs
mv PHASE*.md _internal/phase-history/phase-1-8/

# Move JOSS docs
mv JOSS_*.md _internal/phase-history/joss-prep/

# Create index
cat > _internal/phase-history/README.md << 'EOF'
# FoodSpec Implementation Phase History

Complete record of the 8-phase rewrite from v1.0.0 to v1.1.0.

See BRANCH_MIGRATION_PLAN.md in project root for migration strategy.
EOF
```

#### 2.2 Documentation Structure

**Final docs/ Structure**:
```
docs/
├── index.md                    # Landing page
├── getting-started/            # Quick starts, installation
├── user-guide/                 # User documentation
├── developer-guide/            # Developer docs
├── api/                        # API reference (auto-generated)
├── examples/                   # Example use cases
├── tutorials/                  # Step-by-step guides
├── concepts/                   # Conceptual documentation
├── theory/                     # Scientific background
├── help/                       # Troubleshooting, FAQs
├── migration/                  # v1.0 → v2.0 migration
│   └── v1-to-v2.md
├── workflows/                  # Common workflows
├── visualization/              # Visualization guides
├── protocols/                  # Protocol documentation
├── methods/                    # Method documentation
├── metrics/                    # Metrics documentation
├── datasets/                   # Dataset documentation
├── assets/                     # Images, logos, etc.
└── _internal/                  # Internal/archived docs
    ├── archive/                # Historical docs
    └── phase-history/          # Phase completion docs
```

---

### Phase 3: Source Code Organization

#### 3.1 Clean Single Source Tree

**Final src/ Structure**:
```
src/foodspec/
├── __init__.py                 # Public API exports
├── __version__.py              # Version string
│
├── core/                       # Core data structures & API
│   ├── api.py                  # Main FoodSpec unified API
│   ├── dataset.py              # SpectralDataset
│   ├── artifacts.py            # Artifact management
│   └── manifest.py             # Provenance tracking
│
├── data/                       # Data loading
│   ├── loader.py
│   └── public.py               # Public datasets
│
├── io/                         # File I/O
│   ├── readers.py              # Format readers
│   └── writers.py              # Format writers
│
├── preprocess/                 # Preprocessing
│   ├── baseline.py
│   ├── smoothing.py
│   ├── normalization.py
│   ├── derivatives.py
│   └── recipes.py              # Common pipelines
│
├── features/                   # Feature extraction
│   ├── peaks.py
│   ├── bands.py
│   ├── chemometrics.py
│   └── selection.py
│
├── ml/                         # Machine learning
│   ├── models.py
│   ├── calibration.py
│   └── fusion.py
│
├── chemometrics/               # Chemometric methods
│   ├── pls.py
│   ├── mcr.py
│   ├── simca.py
│   └── vip.py
│
├── stats/                      # Statistical methods
│   ├── hypothesis.py
│   └── multivariate.py
│
├── qc/                         # Quality control
│   ├── engine.py
│   ├── drift.py
│   └── governance.py
│
├── trust/                      # Trust subsystem (NEW)
│   ├── abstain.py
│   ├── conformal.py
│   ├── coverage.py
│   ├── calibration.py
│   ├── evaluator.py
│   └── reliability.py
│
├── reporting/                  # Reporting (NEW)
│   ├── base.py
│   ├── cards.py
│   ├── dossier.py
│   ├── export.py
│   ├── pdf.py
│   └── templates/
│
├── viz/                        # Visualization (NEW)
│   ├── compare.py
│   ├── uncertainty.py
│   ├── embeddings.py
│   ├── processing_stages.py
│   ├── coefficients.py
│   ├── stability.py
│   └── paper.py
│
├── protocol/                   # Protocol system
│   ├── engine.py
│   └── steps/
│
├── cli/                        # Command-line interface
│   ├── main.py
│   └── commands/
│
├── deploy/                     # Deployment utilities
│   ├── artifact.py
│   └── registry.py
│
├── utils/                      # Utilities
│   ├── deprecation.py          # NEW: Deprecation utilities
│   ├── validation.py
│   └── troubleshooting.py
│
└── plugins/                    # Plugin system
    ├── loaders/
    ├── workflows/
    └── indices/
```

**Deprecated Root Modules** (Kept for compatibility, emit warnings):
```
src/foodspec/
├── spectral_dataset.py         # → foodspec.data.SpectralDataset
├── output_bundle.py            # → foodspec.core.OutputBundle
├── model_lifecycle.py          # → foodspec.ml.*
├── preprocessing_pipeline.py   # → foodspec.preprocess.*
├── spectral_io.py              # → foodspec.io.*
├── library_search.py           # → foodspec.features.*
├── validation.py               # → foodspec.stats.*
├── harmonization.py            # → foodspec.preprocess.*
├── narrative.py                # → foodspec.reporting.*
├── reporting.py                # → foodspec.reporting.*
├── rq.py                       # → foodspec.trust.reliability
├── cli_plugin.py               # → foodspec.cli.*
├── cli_predict.py              # → foodspec.cli.*
├── cli_protocol.py             # → foodspec.cli.*
├── cli_registry.py             # → foodspec.cli.*
└── model_registry.py           # → foodspec.core.registry
```

---

### Phase 4: Testing Structure

#### 4.1 Organize Test Suite

**Final tests/ Structure**:
```
tests/
├── conftest.py                 # Shared fixtures
├── __init__.py
│
├── unit/                       # Unit tests (fast)
│   ├── core/
│   ├── preprocess/
│   ├── features/
│   ├── ml/
│   ├── chemometrics/
│   ├── stats/
│   ├── qc/
│   ├── trust/
│   ├── reporting/
│   └── viz/
│
├── integration/                # Integration tests (slower)
│   ├── test_pipelines.py
│   ├── test_protocols.py
│   └── test_workflows.py
│
├── regression/                 # Regression tests
│   └── test_backward_compat.py
│
├── fixtures/                   # Test data & fixtures
│   ├── spectra/
│   ├── protocols/
│   └── expected_outputs/
│
└── benchmarks/                 # Performance benchmarks
    ├── benchmark_heating_quality.py
    └── benchmark_oil_authentication.py
```

**Current tests/ Issues**:
- ✅ Good: Well-organized by module
- ⚠️ Issue: Some tests in `tests/` root, should be in subdirs
- ⚠️ Issue: Duplicate test files between root and subdirs

---

### Phase 5: Examples & Scripts

#### 5.1 Examples Organization

**Final examples/ Structure**:
```
examples/
├── README.md                   # Index of all examples
│
├── quickstarts/                # Quick start scripts
│   ├── oil_authentication_quickstart.py
│   ├── heating_quality_quickstart.py
│   ├── mixture_analysis_quickstart.py
│   ├── aging_quickstart.py
│   ├── phase1_quickstart.py
│   └── qc_quickstart.py
│
├── advanced/                   # Advanced examples
│   ├── foodspec_auto_analysis_script.py
│   ├── governance_demo.py
│   ├── hyperspectral_demo.py
│   ├── moats_demo.py
│   ├── multimodal_fusion_demo.py
│   ├── spectral_dataset_demo.py
│   └── vip_demo.py
│
├── validation/                 # Validation examples
│   ├── validation_chemometrics_oils.py
│   ├── validation_peak_ratios.py
│   └── validation_preprocessing_baseline.py
│
├── new-features/               # v1.1.0 features
│   ├── multi_run_comparison_demo.py
│   ├── uncertainty_demo.py
│   ├── export_demo.py
│   ├── pdf_export_demo.py
│   ├── paper_presets_demo.py
│   ├── embeddings_demo.py
│   ├── processing_stages_demo.py
│   └── coefficients_stability_demo.py
│
├── notebooks/                  # Jupyter notebooks
│   └── trust_visualization_workflow.ipynb
│
├── protocols/                  # Example protocols
│   └── *.yaml
│
├── data/                       # Example data
│   └── *.csv
│
└── configs/                    # Example configs
    └── *.toml
```

#### 5.2 Scripts Organization

**Final scripts/ Structure**:
```
scripts/
├── README.md                   # What each script does
│
├── development/                # Development utilities
│   ├── audit_imports.py
│   ├── test_examples_imports.py
│   └── execute_migration.py
│
├── documentation/              # Doc generation/validation
│   ├── generate_docs_figures.py
│   ├── generate_workflow_figure.py
│   ├── validate_docs.py
│   ├── check_docs_links.py
│   └── bulk_update_links.py
│
├── maintenance/                # Code maintenance
│   ├── fix_codeblock_languages.py
│   ├── fix_methods_depth.py
│   ├── fix_tutorials_depth.py
│   └── fix_workflows_depth.py
│
└── workflows/                  # Example workflows
    └── raman_workflow_foodspec.py
```

---

## Implementation Plan

### Step 1: Backup & Branch
```bash
# Create reorganization branch
git checkout -b reorganize-file-structure

# Create backup
tar -czf foodspec-backup-$(date +%Y%m%d).tar.gz . --exclude='.git'
```

### Step 2: Update .gitignore
```bash
# Add comprehensive .gitignore entries
cat >> .gitignore << 'EOF'

# Build outputs
site/
dist/
build/
*.egg-info/

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
.coverage
.coverage.*
htmlcov/
.benchmarks/

# Application cache
.foodspec_cache/

# Demo/test outputs
outputs/
demo_*/
comparison_output/
protocol_runs_test/
*_runs/
*_output/
*_export/

# Temporary files
*.tmp
*.log
EOF
```

### Step 3: Clean Ignored Files
```bash
# Remove all ignored files
git clean -fdX
```

### Step 4: Archive Phase Documents
```bash
# Create archive structure
mkdir -p _internal/phase-history/{phase-1-8,architecture-docs,joss-prep}

# Move phase documents
mv PHASE*.md _internal/phase-history/phase-1-8/
mv JOSS_*.md _internal/phase-history/joss-prep/

# Move foodspec_rewrite/ docs before deletion
mv foodspec_rewrite/*.md _internal/phase-history/architecture-docs/
mv foodspec_rewrite/docs/*.md _internal/phase-history/architecture-docs/
```

### Step 5: Remove foodspec_rewrite/
```bash
# Remove the entire foodspec_rewrite/ directory
git rm -r foodspec_rewrite/
```

### Step 6: Reorganize Examples
```bash
# Create new structure
mkdir -p examples/{quickstarts,advanced,validation,new-features}

# Move files
mv examples/oil_authentication_quickstart.py examples/quickstarts/
mv examples/heating_quality_quickstart.py examples/quickstarts/
# ... (continue for all files)
```

### Step 7: Reorganize Scripts
```bash
# Create new structure
mkdir -p scripts/{development,documentation,maintenance,workflows}

# Move files
mv scripts/audit_imports.py scripts/development/
mv scripts/generate_docs_figures.py scripts/documentation/
# ... (continue for all files)
```

### Step 8: Update Imports & Tests
```bash
# Run tests to verify nothing broke
pytest tests/

# Fix any broken imports
# (Manual step - update import paths in affected files)
```

### Step 9: Update Documentation
```bash
# Update README.md with new structure
# Update CONTRIBUTING.md with new paths
# Update developer docs with new organization
```

### Step 10: Commit & Push
```bash
# Stage all changes
git add -A

# Commit with detailed message
git commit -m "refactor: Reorganize repository file structure

BREAKING CHANGES:
- Remove foodspec_rewrite/ directory (merge complete)
- Move phase documents to _internal/phase-history/
- Reorganize examples/ into subdirectories
- Reorganize scripts/ into subdirectories
- Update .gitignore to exclude build/demo outputs

Changes:
- Consolidate dual source trees into src/foodspec/
- Archive 40+ phase documents in logical structure
- Clean up 642 cache files via .gitignore
- Remove 27MB built documentation (site/)
- Remove 18MB demo outputs
- Improve discoverability of examples and scripts

Impact:
- Eliminates import ambiguity
- Reduces repo size by ~50MB
- Single source of truth for docs
- Cleaner git history going forward

See: FILE_STRUCTURE_AUDIT.md for full rationale"

# Push to remote
git push origin reorganize-file-structure
```

---

## Expected Outcomes

### Before Reorganization
- **Total Size**: ~75MB (excluding .git)
- **Source Trees**: 2 (ambiguous)
- **Config Files**: 2 pyproject.toml
- **Phase Docs**: 40+ scattered files
- **Demo Outputs**: 6+ directories (25MB)
- **Cache Files**: 642+ files
- **Built Docs**: site/ (27MB)

### After Reorganization
- **Total Size**: ~20MB (excluding .git) - **73% reduction**
- **Source Trees**: 1 (src/foodspec/)
- **Config Files**: 1 pyproject.toml
- **Phase Docs**: Archived in _internal/phase-history/
- **Demo Outputs**: .gitignored (not in repo)
- **Cache Files**: .gitignored (not in repo)
- **Built Docs**: .gitignored (generated on demand)

### Benefits
1. ✅ **Eliminates Import Ambiguity**: Single source tree
2. ✅ **Reduces Repo Size**: 73% size reduction
3. ✅ **Improves Discoverability**: Logical organization
4. ✅ **Cleaner Git History**: No more build artifacts
5. ✅ **Better Maintainability**: Single source of truth
6. ✅ **Faster Clones**: Smaller repo size
7. ✅ **Clearer Structure**: Obvious where things belong

---

## Risk Assessment

### Low Risk
- ✅ Moving documentation files (no code impact)
- ✅ Adding .gitignore entries (no code impact)
- ✅ Removing cache files (regenerated automatically)
- ✅ Removing site/ (regenerated by mkdocs)

### Medium Risk
- ⚠️ Removing foodspec_rewrite/ (verify all imports work)
- ⚠️ Reorganizing examples/ (update documentation links)
- ⚠️ Reorganizing scripts/ (update CI/CD if used)

### High Risk
- None (all changes are organizational, not functional)

### Mitigation
- Create backup before starting
- Work in feature branch
- Run full test suite after each major change
- Verify examples still work
- Update all documentation links
- Review PR carefully before merging

---

## Rollback Plan

If issues are discovered:

1. **Immediate**: Revert merge commit
   ```bash
   git revert <commit-sha>
   git push origin main
   ```

2. **Complete**: Reset to pre-reorganization state
   ```bash
   git checkout main
   git reset --hard <pre-reorganization-sha>
   git push --force origin main
   ```

3. **Partial**: Cherry-pick successful changes
   ```bash
   git cherry-pick <specific-good-commits>
   ```

---

## Timeline Estimate

- **Step 1-2** (Backup & .gitignore): 10 minutes
- **Step 3** (Clean ignored files): 5 minutes
- **Step 4** (Archive phase docs): 20 minutes
- **Step 5** (Remove foodspec_rewrite/): 5 minutes
- **Step 6-7** (Reorganize examples/scripts): 30 minutes
- **Step 8** (Update imports/tests): 60 minutes
- **Step 9** (Update documentation): 30 minutes
- **Step 10** (Commit & review): 20 minutes

**Total**: ~3 hours of focused work

---

## Next Steps

1. **Review this audit** with team/maintainer
2. **Get approval** for proposed reorganization
3. **Schedule maintenance window** (if applicable)
4. **Execute reorganization** following implementation plan
5. **Verify all tests pass**
6. **Update CI/CD** if needed
7. **Merge to main**
8. **Communicate changes** to users

---

## Appendix: Detailed File Counts

### Source Code
- Python files: 234 (as of last count)
- Test files: 150+
- Total lines of code: ~100,000 (including tests)

### Documentation
- Markdown files: 100+
- Phase/completion docs: 40
- User guides: 20+
- Examples: 30+

### Configuration
- YAML files: 10+
- TOML files: 2
- JSON files: 20+

---

*This audit was generated automatically on January 25, 2026, following the v1.1.0-rc1 deployment.*
