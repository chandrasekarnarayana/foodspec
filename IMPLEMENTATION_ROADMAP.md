# FoodSpec End-to-End Design Audit: IMPLEMENTATION ROADMAP

**Status:** Comprehensive audit complete; ready for staged implementation  
**Date:** January 26, 2026  
**Key Finding:** FoodSpec has excellent individual components but lacks unified orchestration.

---

## EXECUTIVE SUMMARY

FoodSpec's goal is to provide **"Protocol-driven, trustworthy Raman/FTIR workflows for food science that are reproducible, auditable, QC-first, and capable of producing regulatory-grade reports."**

### Current State ✅ / ❌

✅ **Exists & Works Well:**
- Preprocessing pipelines (normalize, baseline, smooth)
- Feature engineering (wavelength regions, ratios, statistics)
- Modeling API with cross-validation (LOBO, LOSO, nested)
- Trust stack (calibration, conformal prediction, abstention)
- QC policy system (thresholds for data, spectral, model quality)
- Visualization (ROC, confusion matrix, distributions)
- HTML/PDF reporting infrastructure
- Experiment orchestration (partial)
- Error handling utilities

❌ **Missing / Broken:**
- **Unified orchestrator** that guarantees sequential execution
- **QC gate enforcement** (currently advisory, not blocking)
- **Regulatory mode** with mandatory trust + compliance statements
- **Exit code contract** (0=success, 2=CLI err, 3=validation, 4=protocol, 5=modeling, 6=trust, 7=QC, 8=reporting, 9=artifact)
- **Artifact contract validation** (no check that all required files exist)
- **Structured logging** (logs/run.jsonl with parseable JSON)
- **Dataset fingerprinting** (SHA256 + metadata)
- **error.json** generation on failures with remediation hints
- **CLI `run-workflow`** command (exists but not fully integrated)
- **Regulatory compliance statements**

### Why It Matters

**Without orchestration:**
- Users can run preprocessing, forget features, fit model on wrong data
- QC gates don't actually block pipeline; just print warnings
- Regulatory mode has no guarantee of trust stack application
- Artifacts can disappear silently; no validation
- Error messages don't suggest fixes
- Can't replay exact command line later

**With orchestration:**
- ✅ Pipeline guaranteed: preprocess → features → CV → trust → report
- ✅ QC gates block in regulatory mode (exit code 7)
- ✅ Regulatory mode: calibration + conformal auto-applied
- ✅ Artifact contract: validation fails if required files missing
- ✅ error.json: detailed remediation hints
- ✅ Manifest: exact reproducibility info (seed, git hash, input fingerprints)

---

## WHAT WAS AUDITED

| Component | Status | Health |
|-----------|--------|--------|
| CLI (`main.py`) | ✅ Exists | ⚠️ No exit code contract; protocol error handling weak |
| Protocol system | ✅ Exists | ⚠️ Authority rules not enforced; override policy missing |
| Data objects | ✅ Exists | ⚠️ Schema validation exists but not mandatory |
| Preprocessing | ✅ Exists | ✅ Good; modular; configurable |
| Features | ✅ Exists | ⚠️ Supports many methods; no data leakage guards |
| Modeling API | ✅ Exists | ⚠️ Powerful but not policy-aware; no model approval list |
| Trust stack | ✅ Exists | ✅ Well-built; disconnected from orchestrator |
| QC system | ✅ Exists | ❌ **Critical gap**: gates advisory, not blocking |
| Validation | ✅ Exists | ✅ Good metrics; not in critical path |
| Visualization | ✅ Exists | ✅ Good coverage; missing calibration curves |
| Reporting | ✅ Exists | ❌ **Critical gap**: no QC/trust embedding; no compliance statements |
| **Orchestration** | ❌ **MISSING** | ❌ **BLOCKER**: No single entry point guaranteeing pipeline |
| Error handling | ⚠️ Partial | ❌ **Critical gap**: No `error.json`; no exit code semantics |
| Logging | ⚠️ Partial | ❌ **Critical gap**: No structured JSON logging |
| Testing | ⚠️ Partial | ⚠️ Unit tests good; no end-to-end regulatory workflows |

---

## RECOMMENDED 3-PHASE IMPLEMENTATION

### 🎯 Phase 1: Orchestrator + Error Handling (Weeks 1-2)

**Goal:** Establish single entry point with guaranteed artifact contract

**Files to Create/Modify:**
```
NEW:
  src/foodspec/workflow/orchestrator.py      [INCOMPLETE - enhance] ~800 lines
  src/foodspec/utils/dataset_fingerprint.py  [NEW] ~150 lines
  tests/test_orchestrator_unit.py            [NEW] ~400 lines
  schemas/error.json                         [NEW] ~80 lines
  schemas/manifest.json                      [NEW] ~100 lines

MODIFY:
  src/foodspec/core/errors.py                [ADD ErrorContext, custom exceptions] +100 lines
  src/foodspec/cli/main.py                   [ADD run-workflow command] +50 lines
  src/foodspec/utils/run_artifacts.py        [ENHANCE manifest generation] +50 lines
  src/foodspec/logging_utils.py              [ADD StructuredJsonFormatter] +80 lines
```

**What Gets Built:**
- Orchestrator with `run_workflow(config) → WorkflowResult`
- Sequential stages: load → validate → preprocess → features → model → report
- Error handling: `error.json` on all failures with exit codes + hints
- Artifact contract: validation at end (required files must exist)
- Manifest: versions, seeds, git hash, protocol hash, input fingerprints
- CLI: `foodspec run-workflow --protocol ... --input ... --mode research|regulatory`

**Acceptance Criteria:**
- ✅ `foodspec run-workflow` with research mode runs successfully (exit 0)
- ✅ CSV validation failure produces `error.json` + exit code 3
- ✅ Protocol error produces `error.json` + exit code 4
- ✅ Artifact contract validation: missing files → exit code 9
- ✅ Manifest includes: foodspec version, numpy/sklearn versions, git hash, seed, protocol hash, input sha256
- ✅ Unit tests: 90%+ coverage of orchestrator logic
- ✅ Integration test (fixture dataset): end-to-end success path
- ✅ **No existing APIs broken** (backward compatibility maintained)

**Risks & Mitigations:**
- **Risk:** ProtocolRunner integration complexity
  - **Mitigation:** Create adapter layer; keep orchestrator independent
- **Risk:** Test fixture instability
  - **Mitigation:** Use deterministic synthetic data; set seed early
- **Risk:** Backward compatibility
  - **Mitigation:** Keep `run_protocol` command; add `run-workflow` as new

---

### 🎯 Phase 2: QC Gates + Regulatory Mode (Weeks 3-4)

**Goal:** Implement mandatory QC gates and regulatory compliance

**Files to Create/Modify:**
```
NEW:
  tests/test_end_to_end.py                   [NEW] ~500 lines (research + regulatory)

ENHANCE:
  src/foodspec/workflow/orchestrator.py      [ADD 3 QC gates + trust stack] +400 lines
  src/foodspec/qc/gates.py                   [NEW - refactor existing QC] ~200 lines
  src/foodspec/trust/                        [Auto-apply calibration + conformal] +150 lines
  src/foodspec/reporting/html.py             [Embed QC + trust in report] +100 lines
  src/foodspec/reporting/pdf.py              [Regulatory PDF template] +100 lines
```

**What Gets Built:**

**QC Gate #1: Data Quality** (before preprocessing)
```python
checks = [
    min_samples_per_class ≥ threshold,
    imbalance_ratio ≤ threshold,
    missing_fraction ≤ threshold,
]
→ data_qc_report.json
→ FAIL: exit code 7 (regulatory blocks), warnings (research continues)
```

**QC Gate #2: Spectral Quality** (after preprocessing)
```python
checks = [
    health_score ≥ threshold,
    spike_fraction ≤ threshold,
    saturation_fraction ≤ threshold,
    baseline_drift ≤ threshold,
]
→ spectral_qc_report.json
→ FAIL: exit code 7 (regulatory blocks)
```

**QC Gate #3: Model Performance** (after CV)
```python
checks = [
    accuracy ≥ 0.85,
    per_class_recall ≥ 0.80,
    specificity ≥ 0.90 (if binary),
]
→ model_qc_report.json
→ FAIL: exit code 7 (regulatory blocks)
```

**Regulatory Mode Mandatory Trust Stack:**
```
Calibration: Isotonic or Platt on hold-out set
  → calibration_artifact.json
Conformal: α=0.1 (90% coverage guarantee)
  → conformal_artifact.json
Abstention: optional (if protocol specifies)
```

**Regulatory Report:**
- HTML + PDF (PDF required)
- Embed all QC reports (gate 1, 2, 3)
- Embed calibration + conformal artifacts
- Compliance statement: "This model meets [standard] and is suitable for [use case]"

**Acceptance Criteria:**
- ✅ Research mode: QC gates are warnings (continue on fail)
- ✅ Regulatory mode: QC gate failures block (exit 7)
- ✅ Data QC: identifies imbalance, missing data
- ✅ Spectral QC: identifies poor quality spectra
- ✅ Model QC: identifies insufficient performance
- ✅ Regulatory mode: calibration auto-applied
- ✅ Regulatory mode: conformal prediction with 90% coverage
- ✅ Regulatory PDF: includes all QC + trust artifacts
- ✅ Integration test (fixture): research mode, all gates pass
- ✅ Integration test: regulatory mode, forced QC failure → exit 7
- ✅ Integration test: regulatory mode, all gates pass → PDF created

**Risks & Mitigations:**
- **Risk:** QC thresholds too strict (block legitimate runs)
  - **Mitigation:** Configurable via protocol; defaults from data science literature
- **Risk:** Trust stack needs validation split (not enough data for calibration)
  - **Mitigation:** Allocate 15-20% of CV folds to calibration; fail gracefully if insufficient
- **Risk:** PDF generation complexity
  - **Mitigation:** Use existing reportlab infrastructure; keep templates simple

---

### 🎯 Phase 3: Documentation + Polish (Weeks 5-6)

**Goal:** Public-facing documentation + CI/CD integration + examples

**Files to Create/Modify:**
```
NEW:
  docs/north_star_workflow.md                [NEW] ~400 lines
  docs/modes_research_vs_regulatory.md       [NEW] ~300 lines
  docs/artifact_contract.md                  [NEW] ~400 lines
  docs/error_handling.md                     [NEW] ~200 lines
  examples/protocols/research_simple.yaml    [NEW]
  examples/protocols/regulatory_strict.yaml  [NEW]

ENHANCE:
  README.md                                  [Add run-workflow quickstart]
  .github/workflows/                         [ADD artifact contract validation]
```

**What Gets Built:**
- **North Star diagram:** Visual pipeline (research + regulatory)
- **Mode guide:** Policy differences table + examples
- **Artifact contract:** Complete schema + required files list
- **Error handling:** Exit code guide + remediation hints per code
- **Example protocols:** Simple research protocol + strict regulatory protocol
- **CI/CD:** Validation step checks artifact contract for test runs
- **README:** Quick-start with `foodspec run-workflow` command

**Acceptance Criteria:**
- ✅ Docs render correctly (no broken links)
- ✅ Example protocols load without error
- ✅ CI smoke test: `foodspec run-workflow` completes successfully
- ✅ README updated with North Star section
- ✅ Help text: `foodspec run-workflow --help` is clear and useful

**Risks & Mitigations:**
- **Risk:** Documentation drift (examples fall out of sync)
  - **Mitigation:** Run docs examples in CI; fail if they break
- **Risk:** CI flakiness
  - **Mitigation:** Use minimal fixture datasets; set timeouts

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Orchestrator + Error Handling

**Week 1:**
- [ ] Enhance `orchestrator.py`:
  - [ ] Add `_stage_load_protocol()` with error handling
  - [ ] Add `_stage_load_data()` with CSV validation + fingerprinting
  - [ ] Add `_write_error_json()` with hints
  - [ ] Add `_finalize_result()` with manifest generation
  - [ ] Add `_stage_validate_artifact_contract()`

- [ ] Create `dataset_fingerprint.py`:
  - [ ] `DatasetFingerprint` dataclass
  - [ ] `compute_fingerprint()` function
  - [ ] SHA256 hashing

- [ ] Create error schemas: `schemas/error.json`, `schemas/manifest.json`

- [ ] Enhance `src/foodspec/core/errors.py`:
  - [ ] Add `ErrorContext` dataclass
  - [ ] Add custom exceptions: `ProtocolError`, `ValidationError`, etc.
  - [ ] Exit code mapping

**Week 2:**
- [ ] Add CLI command in `main.py`:
  - [ ] `foodspec run-workflow` with all flags
  - [ ] Flag validation + type checking
  - [ ] Call orchestrator + print result

- [ ] Enhance `logging_utils.py`:
  - [ ] `StructuredJsonFormatter` for JSON logs
  - [ ] `setup_structured_logging()` 
  - [ ] `log_stage_start()`, `log_stage_complete()`

- [ ] Create unit tests: `tests/test_orchestrator_unit.py`
  - [ ] Test protocol loading
  - [ ] Test CSV validation
  - [ ] Test artifact contract
  - [ ] Test error.json generation

- [ ] Create minimal integration test: fixture dataset → research mode

**Validation:**
- [ ] `foodspec run-workflow --protocol examples/protocols/test.yaml --input data/test.csv` runs
- [ ] Check `runs/{run_id}/manifest.json` contains: version, seed, git hash, protocol hash, input sha256
- [ ] Check `runs/{run_id}/error.json` exists on failure + contains "recommendations"
- [ ] Unit test coverage ≥ 90%

---

### Phase 2: QC Gates + Regulatory Mode

**Week 3:**
- [ ] Add `_stage_data_qc()` to orchestrator
  - [ ] Call `check_class_balance()`
  - [ ] Evaluate against `QCPolicy`
  - [ ] Write `data_qc_report.json`
  - [ ] Enforce gate in regulatory mode (exit 7)

- [ ] Add `_stage_spectral_qc()` to orchestrator
  - [ ] Call spectral quality checks
  - [ ] Write `spectral_qc_report.json`
  - [ ] Enforce gate

- [ ] Add `_stage_model_qc()` to orchestrator
  - [ ] Check accuracy, per-class recall
  - [ ] Write `model_qc_report.json`
  - [ ] Enforce gate

- [ ] Create `qc/gates.py` (refactor existing QC into gate functions)

**Week 4:**
- [ ] Add `_stage_trust()` to orchestrator
  - [ ] Auto-apply calibration (Isotonic) in regulatory mode
  - [ ] Auto-apply conformal (α=0.1) in regulatory mode
  - [ ] Write `trust/calibration_artifact.json`, `trust/conformal_artifact.json`
  - [ ] On error: log + fail if regulatory (exit 6)

- [ ] Enhance reporting:
  - [ ] HTML report: embed QC gate results
  - [ ] PDF report: multi-page regulatory template
  - [ ] Generate `REGULATORY_COMPLIANCE_STATEMENT.txt`

- [ ] Create integration tests: `tests/test_end_to_end.py`
  - [ ] Research mode: fixture → all gates pass → HTML report
  - [ ] Regulatory mode: fixture → all gates pass → HTML + PDF reports
  - [ ] Regulatory mode: fixture (imbalanced) → data QC fails → exit 7
  - [ ] Regulatory mode: fixture (bad quality) → spectral QC fails → exit 7
  - [ ] Regulatory mode: fixture (low accuracy) → model QC fails → exit 7

**Validation:**
- [ ] `foodspec run-workflow --mode regulatory` creates `data_qc_report.json`, `spectral_qc_report.json`, `model_qc_report.json`
- [ ] Regulatory mode PDF report contains all QC results + compliance statement
- [ ] Forced QC failure (imbalanced data) → exit code 7 + `error.json` with hint
- [ ] Integration test: fixture → regulatory → success path completes

---

### Phase 3: Documentation + Polish

**Week 5:**
- [ ] Create `docs/north_star_workflow.md`
  - [ ] ASCII pipeline diagrams (research + regulatory)
  - [ ] Module ownership table
  - [ ] Artifact directory tree

- [ ] Create `docs/modes_research_vs_regulatory.md`
  - [ ] Policy differences table
  - [ ] Example commands

- [ ] Create `docs/artifact_contract.md`
  - [ ] Required files per mode
  - [ ] Schema validation rules
  - [ ] Examples

**Week 6:**
- [ ] Create example protocols:
  - [ ] `examples/protocols/research_simple.yaml`
  - [ ] `examples/protocols/regulatory_strict.yaml`

- [ ] Update README:
  - [ ] Add "North Star" section with diagram
  - [ ] Add quick-start: `foodspec run-workflow --protocol ... --input ...`

- [ ] Enhance CI/CD:
  - [ ] Add `.github/workflows/` step: artifact contract validation
  - [ ] Smoke test: `foodspec run-workflow` with fixture

- [ ] Polish CLI help:
  - [ ] `foodspec run-workflow --help` shows exit code legend
  - [ ] Example in help text

**Validation:**
- [ ] Docs build without errors (broken links, etc.)
- [ ] Example protocols load + validate
- [ ] CI smoke test passes
- [ ] README rendered correctly on GitHub

---

## FILE-BY-FILE IMPLEMENTATION GUIDE

### 1. `src/foodspec/workflow/orchestrator.py` (ENHANCE)

**Current:** Mostly stub; basic initialization  
**Target:** Full end-to-end pipeline with all 8 stages

```python
# Pseudocode structure:

class Orchestrator:
    def run(self) -> WorkflowResult:
        # Phase 1: Core pipeline
        self._init_run()
        protocol = self._stage_load_protocol()  # Exit 4 on fail
        dataset = self._stage_load_data()       # Exit 3 on fail
        
        # Phase 2: QC gates (regulatory only)
        if mode == REGULATORY:
            self._stage_data_qc(dataset)        # Exit 7 on fail
        
        X, y, groups = self._stage_preprocess_features(dataset, protocol)
        
        if mode == REGULATORY:
            self._stage_spectral_qc(X)          # Exit 7 on fail
        
        result = self._stage_model(X, y, groups, protocol)  # Exit 5 on fail
        
        if mode == REGULATORY:
            self._stage_model_qc(result)        # Exit 7 on fail
        
        # Phase 3: Trust + reporting
        if mode == REGULATORY or enable_trust:
            self._stage_trust(result, ...)      # Exit 6 on fail
        
        if enable_figures:
            self._stage_figures(result, ...)    # Exit 8 on fail
        
        if enable_report:
            self._stage_report(result, ...)     # Exit 8 on fail
        
        # Phase 4: Validation
        self._stage_validate_artifact_contract()  # Exit 9 on fail
        
        return self._finalize_result(SUCCESS)
```

**Key Methods to Implement:**
- `_stage_load_protocol()`: Load YAML, validate schema, return ProtocolConfig
- `_stage_load_data()`: Read CSV, validate schema, fingerprint, return SpectralDataset
- `_stage_data_qc()`: Call `qc.check_class_balance()`, evaluate against policy
- `_stage_spectral_qc()`: Call spectral QC checks
- `_stage_model_qc()`: Evaluate CV metrics against policy
- `_stage_preprocess_features()`: Call preprocessing + features stages
- `_stage_model()`: Call `modeling.api.fit_predict()`
- `_stage_trust()`: Call calibration + conformal
- `_stage_figures()`: Call viz module
- `_stage_report()`: Call HtmlReportBuilder + PdfReportBuilder
- `_stage_validate_artifact_contract()`: Check required files exist
- `_write_error_json()`: Write error artifact with hints
- `_finalize_result()`: Create manifest, return WorkflowResult

---

### 2. `src/foodspec/utils/dataset_fingerprint.py` (NEW)

```python
@dataclass
class DatasetFingerprint:
    csv_path: str
    file_size_bytes: int
    sha256_hash: str
    row_count: int
    column_count: int
    column_names: list
    dtypes: Dict[str, str]
    missing_counts: Dict[str, int]
    missing_fraction: float
    numeric_stats: Dict[str, Any]
    class_distribution: Dict[str, int]

def compute_fingerprint(csv_path: str) -> DatasetFingerprint:
    # Read CSV, compute hash, extract stats
    # Return fingerprint object
```

---

### 3. `src/foodspec/core/errors.py` (ENHANCE)

```python
@dataclass
class ErrorContext:
    exit_code: int
    error_type: str
    message: str
    stage: str
    recommendations: List[str]
    details: Dict[str, Any]

class FoodSpecError(Exception):
    exit_code = 1
    stage = "unknown"
    
    def to_error_context(self) -> ErrorContext:
        # ...

class ProtocolError(FoodSpecError):
    exit_code = 4
    stage = "protocol"

# Similar: ValidationError (3), ModelingError (5), TrustError (6),
#          QCError (7), ReportingError (8), ArtifactError (9)
```

---

### 4. `src/foodspec/cli/main.py` (ADD COMMAND)

```python
@app.command("run-workflow")
def run_workflow(
    protocol: str = typer.Option(..., "--protocol", "-p"),
    input_csv: str = typer.Option(..., "--input", "-i"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o"),
    mode: str = typer.Option("research", "--mode", "-m"),
    scheme: str = typer.Option("lobo", "--scheme", "-s"),
    seed: int = typer.Option(0, "--seed"),
    enable_trust: bool = typer.Option(False, "--trust"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run complete end-to-end workflow."""
    config = WorkflowConfig(
        protocol_path=protocol,
        input_csv=input_csv,
        output_dir=output_dir,
        mode=RunMode(mode),
        scheme=ValidationScheme(scheme),
        seed=seed,
        enable_trust=enable_trust,
        verbose=verbose,
    )
    result = run_workflow(config)
    
    # Print summary
    typer.echo(f"\nWorkflow: {result.status}")
    typer.echo(f"Exit code: {result.exit_code}")
    if result.report_path:
        typer.echo(f"Report: {result.report_path}")
    if result.error_json_path:
        typer.echo(f"Error details: {result.error_json_path}")
    
    raise typer.Exit(result.exit_code)
```

---

### 5. `tests/test_orchestrator_unit.py` (NEW - Phase 1)

```python
def test_orchestrator_init():
    config = WorkflowConfig(protocol_path="...", input_csv="...")
    orch = Orchestrator(config)
    assert orch.config.seed == 0

def test_csv_validation_fails():
    config = WorkflowConfig(..., input_csv="bad.csv")
    orch = Orchestrator(config)
    result = orch.run()
    assert result.exit_code == ExitCode.VALIDATION_ERROR
    assert (orch.run_dir / "error.json").exists()

def test_artifact_contract():
    # Run workflow, check all required files exist
    ...

def test_error_json_structure():
    # Load error.json, verify schema
    ...
```

---

### 6. `tests/test_end_to_end.py` (NEW - Phase 2)

```python
@pytest.fixture
def fixture_dataset():
    X = np.random.randn(200, 1500)
    y = np.repeat([0, 1, 2], 200 // 3)[:200]
    return X, y

@pytest.fixture
def fixture_csv(tmp_path, fixture_dataset):
    # Write to CSV
    return csv_path

def test_research_mode_end_to_end():
    config = WorkflowConfig(mode=RunMode.RESEARCH, ...)
    orch = Orchestrator(config)
    result = orch.run()
    
    assert result.exit_code == ExitCode.SUCCESS
    assert (orch.run_dir / "manifest.json").exists()
    assert (orch.run_dir / "report" / "index.html").exists()

def test_regulatory_mode_qc_gates_pass():
    config = WorkflowConfig(mode=RunMode.REGULATORY, ...)
    orch = Orchestrator(config)
    result = orch.run()
    
    assert result.exit_code == ExitCode.SUCCESS
    assert (orch.run_dir / "data_qc_report.json").exists()
    assert (orch.run_dir / "spectral_qc_report.json").exists()
    assert (orch.run_dir / "model_qc_report.json").exists()
    assert (orch.run_dir / "trust" / "calibration_artifact.json").exists()
    assert (orch.run_dir / "report" / "report_regulatory.pdf").exists()

def test_regulatory_mode_qc_gate_1_fails():
    # Imbalanced dataset
    config = WorkflowConfig(mode=RunMode.REGULATORY, ...)
    orch = Orchestrator(config)
    result = orch.run()
    
    assert result.exit_code == ExitCode.QC_ERROR
    assert (orch.run_dir / "error.json").exists()
    with open(orch.run_dir / "error.json") as f:
        error_json = json.load(f)
    assert "recommendations" in error_json
```

---

## VALIDATION STRATEGY

### Unit Tests (Phase 1)
```
tests/test_orchestrator_unit.py
  ✓ Orchestrator initialization
  ✓ Protocol loading + validation
  ✓ CSV schema validation
  ✓ Error JSON generation
  ✓ Artifact contract (missing files → exit 9)
  ✓ Manifest generation
  ✓ Exit code mapping
  ✓ Override policy logging
```

### Integration Tests (Phase 2)
```
tests/test_end_to_end.py
  ✓ Research mode: full pipeline success
  ✓ Regulatory mode: full pipeline success (all gates pass)
  ✓ Regulatory mode: data QC fails → exit 7
  ✓ Regulatory mode: spectral QC fails → exit 7
  ✓ Regulatory mode: model QC fails → exit 7
  ✓ Regulatory mode: PDF report generated
  ✓ Regulatory mode: compliance statement generated
```

### CLI Smoke Tests (Phase 3)
```
.github/workflows/
  ✓ foodspec run-workflow --help (exit 0)
  ✓ foodspec run-workflow [fixture] (exit 0)
  ✓ foodspec run-workflow [bad CSV] (exit 3 + error.json)
```

---

## RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| Breaking backward compatibility | Keep `run_protocol` command; add `run-workflow` as new |
| QC thresholds too strict | Make all configurable via protocol; document defaults |
| Trust stack complexity | Use existing implementations; add adapter layer |
| Test fixture flakiness | Deterministic seed; small fixture size |
| Performance degradation | Lazy-load heavy modules; cache preprocessed data |
| Documentation drift | Run doc examples in CI; fail if they break |

---

## SUCCESS CRITERIA

**End of Phase 1:**
- ✅ `foodspec run-workflow` command exists and works
- ✅ Research mode workflow completes (exit 0)
- ✅ Error handling produces `error.json` + correct exit codes
- ✅ Manifest contains version, seed, git hash, input fingerprint
- ✅ Artifact contract validates required files
- ✅ Unit tests: 90%+ coverage
- ✅ No existing APIs broken

**End of Phase 2:**
- ✅ Regulatory mode enforces QC gates (exit 7 on fail)
- ✅ Regulatory mode auto-applies calibration + conformal
- ✅ Regulatory mode generates PDF report + compliance statement
- ✅ All QC reports embedded in HTML + PDF
- ✅ Integration tests: research + regulatory workflows
- ✅ Regulatory mode forced QC failure produces sensible remediation hints

**End of Phase 3:**
- ✅ North Star documentation published
- ✅ Mode guide published
- ✅ Artifact contract documented
- ✅ Example protocols working
- ✅ README updated
- ✅ CI/CD validates artifact contract

---

## BACKWARD COMPATIBILITY

**Keep Working:**
- ✅ `foodspec run_protocol` (existing command)
- ✅ `foodspec run_e2e` (existing command)
- ✅ `foodspec report-run` (existing command)
- ✅ Python API: `Experiment`, `ProtocolRunner`, `fit_predict`
- ✅ Protocol YAML format (no breaking changes)

**Add New:**
- ✅ `foodspec run-workflow` (new command)
- ✅ `WorkflowConfig`, `Orchestrator` classes (new API)
- ✅ Exit code contract (new semantics)
- ✅ `error.json` artifact (new output)

---

## NEXT STEPS

1. **Immediate:** Review audit findings with team
2. **Week 1:** Start Phase 1 implementation
3. **Week 2:** Prototype orchestrator with fixture dataset
4. **Week 3:** Add QC gates, regulatory mode
5. **Week 4:** Integration tests, edge cases
6. **Week 5-6:** Documentation, CI/CD, polish

---

## APPENDIX: NORTH STAR DIAGRAM

```
INPUT: CSV + Protocol YAML
  ↓
📋 Schema Validation
  ├─ CSV shape, dtypes, missing data
  ├─ Protocol YAML syntax
  └─ Fingerprint: SHA256(CSV)
  ↓
[Research Mode Only]                  [Regulatory Mode Only]
                                      ↓
                                      🚪 QC Gate #1: Data Quality
                                      ├─ min_samples_per_class
                                      ├─ imbalance_ratio
                                      ├─ missing_fraction
                                      ✅ PASS → continue
                                      ❌ FAIL → exit 7
  ↓
🔧 Preprocessing & Features
  ├─ Normalize, smooth, baseline
  ├─ Wavelength regions, statistics
  └─ X_preprocessed, X_features
  ↓
                                      [Regulatory Only]
                                      🚪 QC Gate #2: Spectral Quality
                                      ├─ health_score
                                      ├─ spike_fraction
                                      ├─ saturation
                                      ✅ PASS → continue
                                      ❌ FAIL → exit 7
  ↓
🤖 Model Training & CV
  ├─ Cross-validation (LOBO/LOSO/nested)
  ├─ Hyperparameter search
  └─ Metrics: accuracy, precision, recall
  ↓
                                      [Regulatory Only]
                                      🚪 QC Gate #3: Model Performance
                                      ├─ accuracy ≥ 0.85
                                      ├─ recall ≥ 0.80 (per class)
                                      ├─ specificity ≥ 0.90 (binary)
                                      ✅ PASS → continue
                                      ❌ FAIL → exit 7
  ↓
🔐 Trust Stack
  ├─ [Research: Optional]
  ├─ [Regulatory: MANDATORY]
  ├─ Calibration (Isotonic or Platt)
  ├─ Conformal Prediction (α=0.1, 90% coverage)
  ├─ Abstention (optional)
  └─ Trust artifacts: JSON
  ↓
📊 Visualization & Report
  ├─ Figures: ROC, confusion, distributions
  ├─ HTML report (all modes)
  ├─ PDF report (regulatory only)
  ├─ [Research: Optional compliance]
  └─ [Regulatory: MANDATORY compliance statement]
  ↓
✅ Artifact Contract Validation
  ├─ All required files exist
  ├─ Manifest complete (versions, seeds, hashes)
  └─ error.json (only if failed)
  ↓
OUTPUT: runs/{run_id}/ directory tree
✅ SUCCESS: exit code 0
❌ FAILURE: exit code 2-9 + error.json + remediation hints
```

---

**Prepared by:** Principal Engineer + Scientific Software Auditor  
**Date:** January 26, 2026  
**Status:** Ready for implementation
