# FoodSpec Implementation Audit

**Date:** December 24, 2025  
**Goal:** Validate codebase alignment with the Implementation Outline  
**Scope:** Folder structure, entry point UX, triple output (metrics/diagnostics/provenance/artifacts), missing modules, code-quality guardrails

---

## Executive Summary

The FoodSpec codebase now ships the unified entry point and artifact bundling, but still needs standardization and code-quality hardening. Key findings:

- ✅ **Unified entry point delivered**: `FoodSpec` lives in `core/api.py`; CLI `run-exp` executes `exp.yml` end to end
- ✅ **Artifact path in place**: `artifact.py` supports save/load + `Predictor`; `OutputBundle` captures metrics/diagnostics/provenance
- ⚠️ **Triple output still uneven**: Workflows emit metrics/plots, but provenance + artifact structure need consistency
- ⚠️ **Presets/experiments/reporting**: Preset library, experiment diffing, and PDF reporting remain incomplete
- ⚠️ **Code-quality gaps**: Docstring coverage, PEP8 enforcement, large-file splits, and comment guidelines need CI-backed controls


---

## 1. Folder Structure Audit

### Expected vs. Actual

| Module | Expected | Actual | Status | Notes |
|--------|----------|--------|--------|-------|
| `core/` | Spectrum, SpectraSet, schema, units | `dataset.py`, `hyperspectral.py`, `api.py` (FoodSpec) | ✅ Partial | Unified entry point shipped; schema/units modules still missing |
| `io/` | loaders, writers, registry | `loaders.py`, `csv_import.py`, `text_formats.py`, `vendor_formats.py`, `core.py` | ✅ Complete | Good coverage; vendors (SPC, OPUS, JCAMP) supported |
| `preprocess/` | steps, pipeline, auto-preprocess | `baseline.py`, `cropping.py`, `normalization.py`, `smoothing.py`, `preprocessing_pipeline.py` | ✅ Complete | Pipeline infrastructure exists |
| `qc/` | QC metrics, drift, outliers | Limited; `apps/qc.py` for training/inference | ⚠️ Partial | QC logic embedded in apps; not a standalone module |
| `features/` | bands, peaks, ratios, index library | `peaks.py`, `ratios.py` | ✅ Partial | Core feature extraction; index library missing |
| `chemometrics/` | PCA/PLS/PLS-DA/SIMCA/MCR-ALS wrappers | `pca.py`, `mixture.py`, `models.py`, `validation.py` | ✅ Good | Good coverage; some models missing (SIMCA, detailed PLS-DA) |
| `ml/` | train/eval, suites, calibration, uncertainty | Distributed across `model_lifecycle.py`, `apps/`, chemometrics/ | ⚠️ Fragmented | No dedicated ML module; logic scattered |
| `stats/` | hypothesis tests, bootstrap, CI | `stats/` folder with hypothesis_tests.py, robustness.py | ✅ Complete | Comprehensive stats module |
| `exp/` | YAML experiments, run records, diff | Experiment engine + `run-exp` CLI + `artifact.py` | ⚠️ Partial | `exp.yml` execution + artifacts present; diffing/querying missing |
| `report/` | pdf/html reports | `reporting.py`, `viz/report.py` | ⚠️ Partial | Markdown/JSON reports; PDF missing |
| `deploy/` | artifact export + predictor | `artifact.py`, `model_lifecycle.py`, `model_registry.py` | ⚠️ Partial | Artifact bundler + Predictor exist; serving remains | 
| `cli/` | CLI commands | `cli.py`, `cli_*.py` files | ✅ Complete | Good CLI coverage |
| `presets/` | YAML presets library | Not implemented | ❌ Missing | Hardcoded presets in code |

### Missing/Fragmented Modules

1. **`deploy/`** (Priority: High)
   - Artifact export (pickle, joblib, ONNX?)
   - Predictors with uncertainty quantification
   - Model serving utilities

2. **`presets/`** (Priority: Medium)
   - Centralized YAML/JSON presets for preprocessing, features, ML
   - Versioned preset library
   - Composable preset chains

3. **`exp/`** (Priority: Medium)
   - Experiment tracking (run records, diffs, comparisons)
   - Reproducibility metadata
   - Experiment querying/filtering

4. **`ml/`** (Priority: Medium)
   - Dedicated module for ML training/eval
   - Consolidate from `model_lifecycle.py`, `chemometrics/models.py`, `apps/`

5. **`report/`** (Priority: Low)
   - PDF report generation (currently markdown/JSON only)
   - Comprehensive HTML templates

---

## 2. Entry Point UX Audit

### Current State

- `FoodSpec` implemented in `core/api.py` with chainable `qc()`, `preprocess()`, `features()`, `train()`, `export()` flow and `OutputBundle` tracking
- CLI `run-exp` executes `exp.yml` end to end (qc → preprocess → features → train → artifact export)
- Artifact bundling (`artifact.py`) + `Predictor` allow deployable `.foodspec` exports

### Desired State (per outline)

```python
fs = FoodSpec(path_or_dataset)
fs.qc()
fs.preprocess(preset)
fs.features(preset)
fs.train(...)
bundle = fs.export(...)
```

### Gap Analysis

- ⚠️ **App parity** → Some legacy workflows bypass `FoodSpec`; migrate to the unified entry point
- ⚠️ **Presets** → No centralized preset library to power `preprocess()` / `features()` defaults
- ⚠️ **Run records** → Ensure every step logs provenance + seed/versions; add experiment diffing
- ⚠️ **Quality gates** → Enforce docstrings, PEP8, and file-size limits on `core/api.py` and dependent modules

### Gap Analysis

- ❌ **No `FoodSpec()` class** → User must learn multiple entry points
- ❌ **Not chainable** → No fluent API
- ❌ **No preset system** → Users hardcode parameters
- ⚠️ **Output handling unclear** → No single "artifact bundle" pattern

### Recommendation

**Create `foodspec/core/api.py` with `FoodSpec` class** that wraps data, provides chainable methods, and manages the output bundle:

```python
class FoodSpec:
    def __init__(self, path_or_dataset):
        self.data = self._load(path_or_dataset)
        self._output_dir = None
        self._artifacts = OutputBundle()
    
    def qc(self, **kwargs):
        metrics, plots = apply_qc(self.data, **kwargs)
        self._artifacts.add("qc_metrics", metrics)
        return self
    
    def preprocess(self, preset_name):
        self.data = self.data.apply_pipeline(preset_name)
        return self
    
    def train(self, algorithm, label_col, **kwargs):
        self.model = train_model(self.data, algorithm, label_col, **kwargs)
        return self
    
    def export(self, path):
        return self._artifacts.save(path)
```

---

## 3. Triple Output Audit

Each workflow should produce:  
1. **Metrics** (numbers: accuracy, F1, RMSE, etc.)
2. **Diagnostics** (plots + tables: confusion matrix, feature importance, etc.)
3. **Provenance** (what was done, parameters, versions)
4. **Artifacts** (portable export: model, preprocessor, etc.)

### Current Implementations

| Workflow | Metrics | Diagnostics | Provenance | Artifacts | Notes |
|----------|---------|-------------|-----------|-----------|-------|
| Oil auth | ✅ cv_metrics DataFrame | ✅ confusion_matrix, feature_importance | ⚠️ partial | ⚠️ partial | Reports saved but inconsistent |
| Heating | ✅ regression_metrics | ✅ plots + tables | ⚠️ partial | ⚠️ partial | |
| QC | ✅ train/eval metrics | ✅ plots | ⚠️ partial | ✅ model_registry | |
| Mixture | ✅ coefficients, errors | ✅ plots | ❌ missing | ⚠️ partial | Least comprehensive |
| Protocol validation | ✅ benchmarks | ✅ summary tables | ❌ missing | ⚠️ partial | |

### Gaps

1. **Inconsistent provenance logging**
   - Some workflows use `log_run_metadata()` (good)
   - Others don't track parameters, versions, execution time
   - No centralized provenance schema

2. **Artifacts scattered**
   - Model registry exists but not always used
   - No unified export format (pickle vs joblib vs custom)
   - Preprocessors often not saved with models

3. **Diagnostics incomplete**
   - Most generate plots but don't ensure they're saved
   - Missing: feature correlation, residual analysis, CV fold details
   - No PDF reports (only markdown/HTML)

---

## 4. Module Status by Category

### ✅ Complete

- `io/` → All loaders functional
- `preprocess/` → Full pipeline infrastructure
- `cli/` → Good command coverage
- `stats/` → Comprehensive hypothesis tests and robustness
- `viz/` → Core plotting utilities
- `features/peaks` & `features/ratios` → Peak/ratio extraction

### ⚠️ Partial/Fragmented

- `qc/` → Logic in `apps/qc.py`, not standalone module
- `ml/` → Training scattered across `model_lifecycle.py`, `chemometrics/models.py`, `apps/`
- `chemometrics/` → Good core; missing advanced methods (SIMCA, detailed PLS-DA)
- `reporting/` → Markdown/JSON only; no PDF
- `core/` → Missing schema/units modules
- `deploy/` → Artifact bundler + Predictor exist (`artifact.py`); serving + ONNX export not shipped
- `exp/` → `run-exp` executes `exp.yml`; diffing/querying not implemented
- `style/quality` → PEP8/docstring enforcement and file-length guardrails not yet in CI

### ❌ Missing

- `presets/` → Centralized YAML preset library
- `report/pdf` → PDF report generation

---

## 5. Implementation Recommendations (Priority Order)

### Completed: Phase 1 - Unified Entry Point (Delivered)

- `FoodSpec` implemented (`core/api.py`) and exported
- CLI `run-exp` uses the unified pipeline and produces artifacts via `OutputBundle` + `artifact.py`
- Run record + artifact saver/loader in place

### Phase 0: Code Quality & Style Hardening (High Priority)

**Goal:** Enforce readability, PEP8, and documentation standards repo-wide.

**Work:**
1. Add formatter/linter (e.g., `ruff` + `black`) with CI gating and pre-commit
2. Require module-level docstrings and function docstrings (inputs/outputs/logic) across `src/` and `tests/`
3. Add file-length guard (<~600 lines) and refactor oversized modules
4. Increase meaningful comments on complex blocks; document invariants and assumptions
5. Add type-check step (pyright/mypy) to improve debuggability and modularity

**Effort:** ~2–3 days  
**Impact:** Consistent style, easier reviews, maintainability

---

### Phase 2: Triple Output Standardization (High Priority)

**Goal:** Ensure all workflows produce consistent metrics/diagnostics/provenance/artifacts.

**Work:**
1. Define `WorkflowResult` dataclass with guaranteed fields
2. Refactor all app functions to return `WorkflowResult`
3. Add centralized provenance logging (version, parameters, timing, random seed)
4. Ensure all outputs saved under `OutputBundle` structure
5. Add validation tests for each workflow

**Effort:** ~3–4 days  
**Impact:** Reproducibility and consistency

---

### Phase 3: Deploy Module (High Priority)

**Goal:** Export models and preprocessors as portable artifacts.

**Work:**
1. Create `src/foodspec/deploy/` folder
2. Implement artifact bundler (model + preprocessor + metadata)
3. Add ONNX export option (if applicable)
4. Implement predictor class with uncertainty quantification
5. Add serving utilities (REST API skeleton)

**Effort:** ~2–3 days  
**Impact:** Production readiness

---

### Phase 4: Presets Library (Medium Priority)

**Goal:** Centralize configuration as versioned YAML presets.

**Work:**
1. Create `src/foodspec/presets/` folder
2. Write preset schema (preprocessing, features, ML)
3. Implement preset loader/validator
4. Add 5–10 reference presets (quick, standard, publication-ready)
5. Integrate into `FoodSpec.preprocess()` and `FoodSpec.train()`

**Effort:** ~1–2 days  
**Impact:** Usability and reproducibility

---

### Phase 5: Experiment Tracking (Medium Priority)

**Goal:** Track experiments, log results, enable comparison.

**Work:**
1. Create `src/foodspec/exp/` folder
2. Implement experiment record schema
3. Add experiment storage (local sqlite or remote)
4. Implement experiment query/comparison utilities
5. Add experiment diffing (what changed between runs?)

**Effort:** ~2–3 days  
**Impact:** Research workflow support

---

### Phase 6: PDF Reports (Low Priority)

**Goal:** Generate publication-ready PDF reports.

**Work:**
1. Integrate `reportlab` or `weasyprint`
2. Add PDF templates for each workflow
3. Implement report builder
4. Add to `OutputBundle.export()`

**Effort:** ~2–3 days  
**Impact:** Ease of sharing results

---

## 6. File Organization Proposal

```
src/foodspec/
├── core/
│   ├── __init__.py
│   ├── dataset.py          # FoodSpectrumSet
│   ├── api.py              # FoodSpec (NEW - unified entry point)
│   ├── schema.py            # (NEW) Data schema definitions
│   ├── units.py             # (NEW) Unit handling (wavenumbers, etc.)
│   └── hyperspectral.py
├── deploy/                  # (NEW)
│   ├── __init__.py
│   ├── bundler.py           # Artifact bundler
│   ├── predictor.py         # Predictor + uncertainty
│   └── serving.py           # REST/gRPC skeleton
├── presets/                 # (NEW)
│   ├── __init__.py
│   ├── loader.py
│   ├── validator.py
│   └── configs/             # YAML preset files
│       ├── preprocess_quick.yaml
│       ├── preprocess_standard.yaml
│       ├── features_standard.yaml
│       ├── ml_rf.yaml
│       └── ...
├── exp/                     # (NEW)
│   ├── __init__.py
│   ├── record.py            # Experiment record schema
│   ├── storage.py           # Storage backend (sqlite, remote)
│   └── query.py             # Experiment query/diff utilities
├── io/
│   ├── __init__.py
│   ├── ...existing files...
│   └── output_bundle.py     # (NEW) Artifact output management
├── ml/                      # (REFACTOR)
│   ├── __init__.py
│   ├── train.py             # Consolidate training logic
│   ├── calibration.py       # Calibration utilities
│   └── uncertainty.py       # Uncertainty quantification
├── ...other modules...
```

---

## 7. Code Examples (Post-Refactoring)

### Before (Current)
```python
from foodspec import load_folder
from foodspec.apps.oils import run_oil_authentication_workflow

fs = load_folder("data/", metadata_csv="meta.csv")
result = run_oil_authentication_workflow(fs, label_column="oil_type")
print(result.cv_metrics)
# No clear artifact export path
```

### After (Proposed)
```python
from foodspec import FoodSpec

fs = FoodSpec("data/oils.csv", modality="raman")
fs.qc(method="isolation_forest")
fs.preprocess(preset="standard")
fs.features(preset="oil_auth")
model = fs.train(algorithm="rf", label_column="oil_type")
artifacts = model.export(path="./results/", format="bundle")
# artifacts.metrics -> metrics.json
# artifacts.plots -> plots/
# artifacts.model -> model.joblib
# artifacts.provenance -> provenance.json
```

---

## 8. Testing & Validation Plan

1. **Entry Point Tests** → Verify `FoodSpec()` works with paths, datasets, and configs
2. **Triple Output Tests** → Check all workflows produce metrics/diagnostics/provenance/artifacts
3. **Preset Tests** → Validate preset loading, composition, versioning
4. **Deploy Tests** → Export/reimport cycles without loss
5. **Experiment Tests** → Record creation, querying, diffing
6. **Integration Tests** → End-to-end user workflows

---

## 9. Timeline Estimate

| Phase | Effort | Estimated Duration |
|-------|--------|-------------------|
| Phase 1 (Entry Point) | COMPLETE | — |
| Phase 0 (Code Quality & Style) | High | 2–3 days |
| Phase 2 (Triple Output) | High | 3–4 days |
| Phase 3 (Deploy) | High | 2–3 days |
| Phase 4 (Presets) | Medium | 1–2 days |
| Phase 5 (Experiments) | Medium | 2–3 days |
| Phase 6 (PDF Reports) | Low | 2–3 days |
| **Remaining Total** | - | **10–17 days** |

| Phase | Effort | Estimated Duration |
|-------|--------|-------------------|
| Phase 1 (Entry Point) | High | 2–3 days |
| Phase 2 (Triple Output) | High | 3–4 days |
| Phase 3 (Deploy) | High | 2–3 days |
| Phase 4 (Presets) | Medium | 1–2 days |
| Phase 5 (Experiments) | Medium | 2–3 days |
| Phase 6 (PDF Reports) | Low | 2–3 days |
| **Total** | - | **12–18 days** |

---

## 10. Conclusion

FoodSpec has a strong foundation but needs refactoring to match the Implementation Outline. The main gaps are:

1. **No unified entry point** → Implement `FoodSpec()` class
2. **Inconsistent outputs** → Standardize metrics/diagnostics/provenance/artifacts
3. **Missing modules** → Add `deploy/`, `presets/`, `exp/`

Once complete, users will experience a cleaner, more discoverable API and better reproducibility support.

---

**Next Steps:**
1. Enforce code-quality guardrails (formatter, linter, docstring policy, file-length check) via CI
2. Migrate legacy workflows to `FoodSpec` so all apps share the unified entry point
3. Standardize triple-output contract across workflows and artifact exports
4. Stand up presets + experiment diffing modules and PDF reporting
5. Track progress via GitHub issues per phase

