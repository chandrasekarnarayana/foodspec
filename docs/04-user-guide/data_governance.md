# Data Governance & Dataset Intelligence

FoodSpec provides dataset-level intelligence to prevent silent failures in spectroscopy ML workflows. These tools summarize dataset health, diagnose class balance issues, assess replicate consistency, detect leakage, and compute a composite readiness score to gate deployment.

---

## Overview

- **Dataset Summary** — Class distribution, spectral quality (SNR, NaN/inf, negative rate), metadata completeness
- **Class Balance** — Imbalance ratio, undersized classes, recommendations
- **Replicate Consistency** — CV (%) per replicate group; flags high technical variability
- **Leakage Detection** — Batch–label correlation (Cramér's V), replicate leakage risk/detection
- **Readiness Score (0–100)** — Weighted composite across size, balance, replicates, metadata, spectral quality, leakage

---

## Key Assumptions

⚠️ Make these explicit in your study design:
- `label_column` is categorical and complete; `batch_column` and `replicate_column` exist when applicable
- Technical replicates should not be split across train/test; use GroupKFold or GroupShuffleSplit
- Severe batch–label correlation indicates confounding; correct or stratify by batch
- Recommended thresholds: min 20 samples/class, imbalance ≤10:1, technical CV ≤10%

---

## Python API Usage

```python
from foodspec import FoodSpec

fs = FoodSpec("data.csv", modality="raman")

# 1) Dataset summary
summary = fs.summarize_dataset(label_column="oil_type")

# 2) Class balance
balance = fs.check_class_balance(label_column="oil_type")

# 3) Replicate consistency
consistency = fs.assess_replicate_consistency(replicate_column="sample_id")

# 4) Leakage detection
leakage = fs.detect_leakage(
    label_column="oil_type",
    batch_column="batch",
    replicate_column="sample_id",
)

# 5) Readiness score
readiness = fs.compute_readiness_score(
    label_column="oil_type",
    batch_column="batch",
    replicate_column="sample_id",
)
```

---

## Outputs

Metrics recorded in `OutputBundle`:
- `dataset_summary`
- `class_balance`
- `replicate_consistency`
- `leakage_detection`
- `readiness_score`

Artifacts: included in `.foodspec` export via `fs.export(...)`.

---

## Best Practices

- **Stratified splits:** Use `StratifiedKFold` when class imbalance exists
- **Group-aware splits:** Use `GroupKFold`/`GroupShuffleSplit` with `replicate_column`
- **Batch-aware CV:** Use batch-stratified CV or include batch covariates; consider batch correction
- **Minimum viable data:** Target ≥20 samples/class; avoid training on severely imbalanced data without mitigation
- **Monitor SNR:** Low SNR or high negative intensity rates suggest preprocessing issues

---

## Example End-to-End Demo

Run the demo to see all features:

```bash
python examples/governance_demo.py
```

The demo prints:
- Class distribution and spectral quality stats
- Replicate CVs and high-variability flags
- Batch–label Cramér's V and leakage risk
- Readiness score with passed/failed criteria
- Exports metrics to `protocol_runs_test/`

---

## References

- Cramér’s V association measure for categorical variables
- GroupKFold (sklearn) for preventing replicate leakage
- Community defaults for spectroscopy ML readiness thresholds
