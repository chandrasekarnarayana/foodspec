# Minimal Integration Test Protocol

This protocol is used by CI to verify end-to-end execution works after refactoring.

**Target**: Complete in <30 seconds, verify all stages execute, produce required artifacts.

```yaml
metadata:
  name: "Architecture E2E Test"
  version: "1.0"
  description: "Minimal end-to-end test for CI/refactoring verification"
  author: "Architecture Team"
  created: "2026-01-25"

data:
  source: "test_data.csv"  # Small dataset provided by test
  modality: "raman"
  description: "Minimal spectroscopic data"

preprocess:
  enabled: true
  recipe: "raman_standard"  # Built-in recipe
  steps:
    - name: "baseline"
      method: "polynomial"
      params:
        order: 2
    - name: "normalize"
      method: "vector"

qc:
  enabled: true
  checks:
    - name: "data_quality"
      enabled: true
    - name: "drift_detection"
      enabled: false  # Skip slow checks
    - name: "governance"
      enabled: true

features:
  type: "chemometrics"
  enabled: true
  components:
    - type: "pca"
      name: "pca"
      params:
        n_components: 3

task:
  type: "classification"
  target: "class"
  classes: ["A", "B"]

model:
  type: "plsda"
  params:
    n_components: 2

validation:
  cv:
    type: "stratified_kfold"
    n_splits: 3
    random_state: 42
  metrics:
    - "accuracy"
    - "f1_macro"
    - "auc_ovo"

trust:
  enabled: true
  conformal:
    alpha: 0.1
    method: "split"
  calibration:
    method: "platt"
  abstention:
    enabled: false

visualization:
  enabled: false  # Skip slow visualization in CI

reporting:
  enabled: false  # Skip report generation in CI
```

**Test Data** (examples/fixtures/test_minimal.csv):
```csv
wavelength,intensity1,intensity2,intensity3,class
400,10,20,15,A
405,12,22,17,A
410,14,24,19,B
415,16,26,21,B
420,18,28,23,A
425,20,30,25,B
430,22,32,27,A
435,24,34,29,B
440,26,36,31,A
445,28,38,33,B
```

**Expected Outputs**:
```
run_output/
├── manifest.json          # ✓ Must exist
├── metrics.json           # ✓ Must exist
├── predictions.json       # ✓ Must exist
├── cv_results.json        # ✓ Must exist (from CV)
├── qc_results.json        # ✓ Should exist
├── trust/
│   ├── conformal.json     # ✓ Should exist
│   └── calibration.json   # ✓ Should exist
└── test_minimal.log       # ✓ Execution log
```

**Expected Manifest Contents** (manifest.json):
```json
{
  "metadata": {
    "timestamp": "2026-01-25T...",
    "foodspec_version": "1.1.0",
    "python_version": "3.10.x",
    "protocol_name": "Architecture E2E Test",
    "protocol_version": "1.0"
  },
  "artifacts": {
    "metrics": "metrics.json",
    "predictions": "predictions.json",
    "cv_results": "cv_results.json"
  },
  "checksums": {
    "metrics": "sha256:...",
    "predictions": "sha256:..."
  }
}
```

**Verification Criteria**:
- [ ] All 7 stages execute without error (data, preprocess, qc, features, model, evaluate, trust)
- [ ] manifest.json created with required fields
- [ ] metrics.json contains: accuracy, f1_macro, auc_ovo
- [ ] predictions.json contains: predictions, confidence, conformal_set
- [ ] Execution completes in <30 seconds
- [ ] No import errors
- [ ] No missing artifact files
