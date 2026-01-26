# FoodSpec Audit: Executive Summary & Quick Reference

**Audit Date:** January 26, 2026  
**Scope:** End-to-end design correctness, regulatory readiness, policy compliance  
**Auditor:** Principal Engineer + Scientific Software Auditor  
**Status:** ✅ Complete; ready for implementation

---

## 🎯 THE VERDICT

**FoodSpec is 85% complete but 0% orchestrated.**

| Aspect | Status | Score |
|--------|--------|-------|
| Individual modules (preprocess, features, modeling, trust, reporting) | ✅ Excellent | 95/100 |
| Error handling & exit codes | ❌ Missing | 20/100 |
| QC gate enforcement | ❌ Advisory only | 30/100 |
| Regulatory workflow support | ❌ Incomplete | 40/100 |
| Artifact contract validation | ❌ Missing | 0/100 |
| End-to-end orchestration | ❌ Missing | 0/100 |
| **Overall System Readiness** | ⚠️ **Not production-ready** | **57/100** |

---

## 🔴 CRITICAL GAPS (Blockers for Regulatory Use)

| Gap | Impact | Severity | Fix Time |
|-----|--------|----------|----------|
| **No unified orchestrator** | Users can run steps out of order; no guaranteed pipeline | BLOCKER | 2 weeks |
| **QC gates not enforced** | Regulatory workflows can't guarantee compliance | BLOCKER | 1 week |
| **No exit code contract** | Users can't distinguish error types programmatically | HIGH | 3 days |
| **No error.json** | Failures have no remediation hints | HIGH | 3 days |
| **No regulatory compliance statements** | Reports can't be certified for regulatory use | HIGH | 1 week |
| **No artifact validation** | Missing files go undetected; run integrity unknown | HIGH | 2 days |

---

## ✅ WHAT WORKS WELL

- ✅ Preprocessing pipelines (normalize, baseline, smooth, etc.)
- ✅ Feature engineering (wavelength regions, ratios, statistics)
- ✅ Modeling API with flexible cross-validation (LOBO, LOSO, nested)
- ✅ Trust stack (calibration, conformal, abstention)
- ✅ QC policy system (thresholds defined; not enforced)
- ✅ Visualization (ROC, confusion, distributions)
- ✅ HTML/PDF reporting infrastructure
- ✅ Experiment class (good foundation)

---

## ❌ WHAT'S MISSING

| Component | Why It Matters | Effort |
|-----------|---------------|--------|
| Orchestrator.run_workflow() | Guarantees sequential pipeline execution | 2 weeks |
| QC gate enforcement (data, spectral, model) | Blocks regulatory workflows on policy violation | 1 week |
| error.json + exit codes | Users can parse failures; provides fix hints | 3 days |
| Artifact contract validation | Ensures all required files exist; run integrity | 2 days |
| Structured JSON logging | Enables programmatic analysis of runs | 3 days |
| Regulatory compliance statements | Certifies reports for regulatory use | 1 week |
| Dataset fingerprinting | Reproducibility audit trail (SHA256 + metadata) | 2 days |
| North Star documentation | Users understand the guaranteed pipeline | 3 days |

---

## 📋 THE 3-PHASE IMPLEMENTATION PLAN

### Phase 1: Orchestrator + Error Handling (Weeks 1-2)
**Goal:** Establish guaranteed single entry point with error handling

**Deliverables:**
- ✅ `orchestrator.py`: Sequential stages with error propagation
- ✅ Error handling: `error.json` + exit codes (0, 2, 3, 4, 5, 6, 7, 8, 9)
- ✅ Artifact contract: validation that required files exist
- ✅ Manifest: versions, seeds, git hash, input fingerprints
- ✅ CLI: `foodspec run-workflow` command
- ✅ Unit tests: 90%+ coverage

**Result:**
```bash
$ foodspec run-workflow --protocol Oils.yaml --input data.csv --mode research
✅ Workflow complete (exit 0)
✅ Report: runs/run_20260126_123456/report/index.html
```

---

### Phase 2: QC Gates + Regulatory Mode (Weeks 3-4)
**Goal:** Enforce mandatory QC gates in regulatory mode

**Deliverables:**
- ✅ QC Gate #1 (data): min samples, imbalance, missing data
- ✅ QC Gate #2 (spectral): health score, spike fraction, saturation
- ✅ QC Gate #3 (model): accuracy, per-class recall, specificity
- ✅ Mandatory trust: calibration + conformal (α=0.1)
- ✅ Regulatory PDF + compliance statement
- ✅ Integration tests: research + regulatory workflows

**Result:**
```bash
$ foodspec run-workflow --protocol Oils.yaml --input data.csv --mode regulatory
❌ Data QC FAILED: imbalance_ratio 15.2 > 10.0
❌ exit 7
✅ Error JSON: runs/run_xxx/error.json
  {
    "error": "Data quality gate failed",
    "recommendations": [
      "Collect more samples from minority classes",
      "Consider stratified sampling",
      "Adjust QC policy if thresholds are too strict"
    ]
  }
```

---

### Phase 3: Documentation + Polish (Weeks 5-6)
**Goal:** Public-facing docs + CI/CD + examples

**Deliverables:**
- ✅ `docs/north_star_workflow.md` (pipeline diagram + artifact tree)
- ✅ `docs/modes_research_vs_regulatory.md` (policy differences)
- ✅ `docs/artifact_contract.md` (required files + schemas)
- ✅ Example protocols (research_simple.yaml, regulatory_strict.yaml)
- ✅ CI/CD artifact validation
- ✅ README quickstart

**Result:**
```
README:
# Quick Start
$ foodspec run-workflow --protocol examples/protocols/Oils.yaml \
    --input data/oils.csv --mode research
```

---

## 📊 EXIT CODE CONTRACT

```
0 ✅ SUCCESS
    → manifest.json present
    → report exists
    → no error.json

2 ❌ CLI_ERROR
    → Invalid flags or arguments
    → error.json: Check CLI --help

3 ❌ VALIDATION_ERROR
    → CSV schema invalid (shape, dtypes, missing data)
    → error.json: Check CSV format

4 ❌ PROTOCOL_ERROR
    → Protocol YAML syntax/schema invalid
    → error.json: Check YAML syntax

5 ❌ MODELING_ERROR
    → Preprocessing/feature/model fitting failed
    → error.json: Check logs/run.log

6 ❌ TRUST_ERROR
    → Calibration/conformal stack failed
    → error.json: Check trust configuration

7 ❌ QC_ERROR
    → QC gate failed (regulatory mode blocks)
    → error.json: Check *_qc_report.json + recommendations

8 ❌ REPORTING_ERROR
    → HTML/PDF generation failed
    → error.json: Check report configuration

9 ❌ ARTIFACT_ERROR
    → Required files missing from run
    → error.json: List of missing artifacts
```

---

## 📁 ARTIFACT TREE

### All Modes (Mandatory)
```
runs/{run_id}/
├─ manifest.json              ← versions, seeds, git hash, input fingerprints
├─ error.json                 ← (only if failed) exit code + recommendations
├─ logs/
│  ├─ run.log                 ← human-readable
│  ├─ run.jsonl               ← structured JSON (one per line)
│  └─ debug.log               ← DEBUG level logs
└─ data/
   └─ data_summary.json       ← shape, schema, missing data, fingerprint
```

### Research Mode (Optional)
```
├─ preprocessing/
│  ├─ preprocessing_pipeline.pkl
│  └─ X_preprocessed.npy
├─ features/
│  ├─ X_features.npy
│  └─ feature_names.json
├─ model/
│  ├─ model.pkl
│  ├─ metrics.json
│  └─ confusion_matrix.json
├─ figures/
│  ├─ roc_curve.png
│  ├─ confusion_matrix.png
│  └─ metadata.json
└─ report/
   └─ index.html
```

### Regulatory Mode (Mandatory Above + Below)
```
├─ data_qc_report.json           ← Gate #1: MUST be "status": "pass"
├─ spectral_qc_report.json       ← Gate #2: MUST be "status": "pass"
├─ model_qc_report.json          ← Gate #3: MUST be "status": "pass"
├─ trust/
│  ├─ calibration_artifact.json  ← MANDATORY
│  └─ conformal_artifact.json    ← MANDATORY (coverage ≥ 90%)
├─ report/
│  ├─ index.html
│  └─ report_regulatory.pdf      ← MANDATORY (includes all QC + trust)
└─ REGULATORY_COMPLIANCE_STATEMENT.txt
```

---

## 🚀 QUICK START (After Implementation)

### Research Mode
```bash
foodspec run-workflow \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --output-dir runs/exp1 \
  --mode research \
  --seed 42

# Output: runs/exp1/{run_id}/ directory
# - manifest.json (metadata)
# - report/index.html (interactive report)
# - error.json (if failed, with fix hints)
```

### Regulatory Mode
```bash
foodspec run-workflow \
  --protocol examples/protocols/Oils.yaml \
  --input data/oils.csv \
  --output-dir runs/compliance \
  --mode regulatory \
  --seed 42 \
  --trust

# Output: runs/compliance/{run_id}/ directory
# - All research artifacts PLUS:
# - data_qc_report.json (gate #1)
# - spectral_qc_report.json (gate #2)
# - model_qc_report.json (gate #3)
# - trust/calibration_artifact.json
# - trust/conformal_artifact.json
# - report/report_regulatory.pdf (certified)
# - REGULATORY_COMPLIANCE_STATEMENT.txt
```

---

## 🎯 POLICY CONTRACT

### Protocol Authority
✅ Protocol YAML is source of truth  
✅ CLI flags override protocol IF protocol allows (via `allow_cli_override: false`)  
✅ Overrides logged in manifest  
✅ Mode override not allowed in regulatory workflows (mode is immutable)

### Mode Rules
| Aspect | Research | Regulatory |
|--------|----------|-----------|
| QC Gates | Advisory (warn) | Mandatory (block exit 7) |
| Trust Stack | Optional | Required |
| Approved Models | Any | LogisticRegression, PLS-DA, LinearSVC only |
| Report | HTML optional | HTML + PDF required |
| Compliance | Research disclaimer | Certified statement |
| Claims | "achieves X%" | "certified for [use case]" |

### Logging Requirements
```
logs/run.log           # Human readable (INFO level)
logs/run.jsonl         # Structured JSON (DEBUG level, one per line)
                       #   {"timestamp": "...", "level": "INFO", "stage": "preprocessing", "event": "..."}
logs/debug.log         # Full DEBUG level
```

### Reproducibility Requirements
✅ Seed controls all randomness (numpy, random, sklearn, torch)  
✅ Manifest captures: foodspec version, python version, sklearn version, numpy version, git hash, protocol hash, input SHA256  
✅ No hardcoded paths; all relative to run_dir  
✅ All hyperparameters logged  
✅ Environment captured (OS, Python path, etc.)

### Regulatory Safety Requirements
✅ Claims must be qualified with uncertainty (not "100% accuracy")  
✅ Limitations section required (bias risks, data scope, fairness caveats)  
✅ Calibration required (ECE metric before/after)  
✅ Conformal prediction required (coverage ≥ 90% with guarantee)  
✅ QC gates must all pass (status: "pass" in all *_qc_report.json)  
✅ Audit trail must be complete (manifest + logs + fingerprints)

---

## 📚 DOCUMENTATION STRUCTURE (Phase 3)

After Phase 3, these docs will exist:

| Doc | Purpose | Audience |
|-----|---------|----------|
| `north_star_workflow.md` | Pipeline architecture + module ownership | Developers + researchers |
| `modes_research_vs_regulatory.md` | Policy differences + examples | Compliance + researchers |
| `artifact_contract.md` | Required files + schemas + validation rules | System integrators |
| `error_handling.md` | Exit code meanings + remediation per code | End users + automation |
| Example protocols | Working research + regulatory protocols | New users |
| README quickstart | 2-minute getting started | Everyone |

---

## 🔍 HOW TO VERIFY IMPLEMENTATION

### Phase 1 Tests
```bash
# Should succeed
pytest tests/test_orchestrator_unit.py -v
pytest tests/test_end_to_end.py::test_research_mode_end_to_end -v

# Should exit 3
foodspec run-workflow --protocol test.yaml --input bad.csv
# Check: exit code 3 + error.json with "CSV schema invalid"

# Should exit 4
foodspec run-workflow --protocol bad.yaml --input data.csv
# Check: exit code 4 + error.json with "YAML syntax error"

# Check artifacts
ls runs/{run_id}/manifest.json
cat runs/{run_id}/manifest.json | jq '.foodspec_version, .seed, .git_hash, .input_hashes'
```

### Phase 2 Tests
```bash
# Regulatory mode success
foodspec run-workflow --protocol oils.yaml --input oils.csv --mode regulatory
# Check: exit 0 + data_qc_report.json, spectral_qc_report.json, model_qc_report.json all "pass"
# Check: report_regulatory.pdf exists
# Check: REGULATORY_COMPLIANCE_STATEMENT.txt exists

# Regulatory mode QC failure
foodspec run-workflow --protocol oils.yaml --input imbalanced.csv --mode regulatory
# Check: exit 7 + error.json with "recommendations": ["Collect more data", ...]
# Check: data_qc_report.json has "status": "fail"
```

### Phase 3 Tests
```bash
# Docs build
cd docs && mkdocs build
# Check: no broken links, examples render

# Example protocols
foodspec run-workflow --protocol examples/protocols/research_simple.yaml --input data.csv
# Check: exit 0

foodspec run-workflow --protocol examples/protocols/regulatory_strict.yaml --input data.csv --mode regulatory
# Check: exit 0 (assuming data passes QC)
```

---

## 💡 KEY INSIGHTS FROM AUDIT

1. **Design is good, integration is missing.** Each module is well-built; they just don't talk to each other in a guaranteed order.

2. **QC is defined but not enforced.** `QCPolicy` exists; gates exist; but nothing blocks the pipeline on failure.

3. **Trust stack is powerful but optional.** Calibration + conformal work great; just not tied to regulatory mode.

4. **Error handling is scattered.** 50+ try-catch blocks across codebase; no unified error.json output.

5. **Manifest is incomplete.** Some metadata captured; missing: protocol hash, input fingerprints, CLI overrides.

6. **Regulatory workflows are impossible today.** No way to guarantee QC gates, trust stack, compliance statements all together.

7. **Exit codes are undefined.** Current: 0 or 1. Needed: 2-9 for specific error types.

8. **Documentation is module-focused.** Good API docs; missing: end-to-end workflow guide (North Star).

---

## 🎯 SUCCESS METRICS

### Phase 1 Complete ✅
- [ ] `foodspec run-workflow` command exists + works
- [ ] Exit code contract implemented (0, 2, 3, 4, 5, 6, 7, 8, 9)
- [ ] `error.json` generated on all failures with remediation hints
- [ ] Artifact contract validated (required files must exist)
- [ ] Manifest includes versions, seed, git hash, input fingerprints
- [ ] Unit tests: 90%+ coverage
- [ ] No existing APIs broken
- [ ] Fixture dataset end-to-end success (research mode)

### Phase 2 Complete ✅
- [ ] 3 QC gates implemented + enforced (regulatory mode blocks, research mode warns)
- [ ] Regulatory mode auto-applies calibration + conformal
- [ ] Regulatory PDF report generated with all QC + trust results
- [ ] Compliance statement generated + included in report
- [ ] Integration tests: research + regulatory success paths
- [ ] Integration test: QC gate failure → exit 7 + error.json with fix hints

### Phase 3 Complete ✅
- [ ] `north_star_workflow.md`: published + tested
- [ ] `modes_research_vs_regulatory.md`: published with examples
- [ ] `artifact_contract.md`: schema + required files documented
- [ ] Example protocols (research_simple.yaml, regulatory_strict.yaml): tested
- [ ] README: quickstart with `foodspec run-workflow` command
- [ ] CI/CD: smoke test passes; artifact validation in place

---

## 📞 KEY QUESTIONS FOR TEAM

1. **Timeline:** What's the deadline for regulatory compliance?
2. **Model approval:** Which models are approved for regulatory use? (Currently: LogisticRegression, PLS-DA, LinearSVC suggested)
3. **QC thresholds:** Are the suggested defaults reasonable? (min_health_score=0.7, max_imbalance_ratio=10, etc.)
4. **Trust stack:** Is calibration + conformal mandatory, or should abstention be optional in regulatory mode?
5. **Data sensitivity:** Should dataset fingerprints be included in manifest? (For audit trail)
6. **Backward compat:** Is it OK to keep `run_protocol` as legacy while adding `run-workflow` as new?

---

## 📎 ATTACHMENTS

### Full Audit Documents
1. ✅ `AUDIT_END_TO_END_DESIGN.md` (Parts A-I, comprehensive)
2. ✅ `IMPLEMENTATION_ROADMAP.md` (3-phase plan with checklists)
3. ✅ This quick reference

### Code Snippets
- Orchestrator pseudocode (sections ready to implement)
- Error handling examples (exit codes + JSON structure)
- Test fixtures (research + regulatory)
- CLI integration (typer commands)

---

## 🎬 NEXT ACTIONS

**Immediate (This week):**
1. ✅ Review audit findings with team
2. ✅ Clarify model approval list + QC thresholds
3. ✅ Confirm timeline (weeks vs. months)

**Week 1-2 (Phase 1):**
1. ✅ Enhance `orchestrator.py` with all stages
2. ✅ Add error handling + exit codes
3. ✅ Write unit tests
4. ✅ CLI: add `run-workflow` command

**Week 3-4 (Phase 2):**
1. ✅ Implement 3 QC gates
2. ✅ Enforce regulatory mode
3. ✅ Integration tests
4. ✅ PDF report + compliance statements

**Week 5-6 (Phase 3):**
1. ✅ Documentation (north_star, modes, artifact_contract)
2. ✅ Example protocols
3. ✅ CI/CD integration
4. ✅ Polish + release

---

**Prepared by:** Principal Engineer + Scientific Software Auditor  
**Audit Date:** January 26, 2026  
**Repository:** github.com/chandrasekarnarayana/foodspec  
**Branch:** main  
**Status:** 🟢 Ready for implementation
