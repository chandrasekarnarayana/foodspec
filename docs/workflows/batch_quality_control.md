# Workflow: Batch Quality Control / Novelty Detection

## 📋 Standard Header

**Purpose:** Detect off-spec batches or novelty samples by comparing new spectra to a reference library using one-class classification.

**When to Use:**
- Screen incoming raw material batches for authenticity
- Monitor production lots for drift from specifications
- Identify contaminated or adulterated samples without labeled training data
- Flag unusual samples for further lab testing
- Validate supplier consistency across deliveries

**Inputs:**
- Format: HDF5 spectral library (reference) + new samples (HDF5 or CSV)
- Required metadata: `group` column ("auth_ref" vs "evaluation" OR similar)
- Optional metadata: `batch`, `supplier`, `date`
- Wavenumber range: Same as reference library (typically 600–1800 cm⁻¹)
- Min samples: 50+ reference spectra (authentic), any number of evaluation samples

**Outputs:**
- qc_scores.csv — Novelty scores for each evaluation sample
- qc_labels.csv — Predicted labels ("authentic" vs "suspect") based on threshold
- score_distribution.png — Histogram of scores with threshold line
- pca_scores.png — (Optional) PCA showing reference vs evaluation separation
- report.md — Summary with specificity/sensitivity (if labels available)

**Assumptions:**
- Reference library is representative (covers expected variability)
- Preprocessing identical for reference and evaluation samples
- Threshold chosen based on acceptable false positive rate
- One-class model appropriate (novelty = outlier from reference distribution)

---

## 🔬 Minimal Reproducible Example (MRE)

```python
import numpy as np
import matplotlib.pyplot as plt
from foodspec.apps.qc import run_qc_workflow
from foodspec.viz.qc import plot_score_distribution
from examples.qc_quickstart import _synthetic_qc

# Generate synthetic QC dataset (reference + evaluation samples)
fs = _synthetic_qc()
print(f"Total samples: {fs.x.shape[0]}")
print(f"Reference: {(fs.metadata['group'] == 'auth_ref').sum()}")
print(f"Evaluation: {(fs.metadata['group'] == 'evaluation').sum()}")

# Define train mask (reference library)
train_mask = fs.metadata["group"] == "auth_ref"

# Run QC workflow (one-class SVM)
result = run_qc_workflow(
    fs,
    train_mask=train_mask,
    model_type="oneclass_svm",  # or "isolation_forest"
    nu=0.05  # Expected outlier fraction
)

# Display results
print(f"\nQC Results:")
print(f"  Threshold: {result.threshold:.3f}")
print(f"  Predictions: {result.labels_pred.value_counts().to_dict()}")
if 'true_label' in fs.metadata.columns:
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(fs.metadata['true_label'], result.labels_pred))

# Plot score distribution
fig, ax = plt.subplots(figsize=(8, 6))
plot_score_distribution(
    result.scores,
    threshold=result.threshold,
    labels=result.labels_pred,
    ax=ax
)
ax.set_title("QC Score Distribution")
plt.tight_layout()
plt.savefig("qc_score_distribution.png", dpi=150, bbox_inches='tight')
print("Saved: qc_score_distribution.png")
```

**Expected Output:**
```mermaid
Total samples: 150
Reference: 100
Evaluation: 50

QC Results:
  Threshold: -0.325
  Predictions: {'authentic': 45, 'suspect': 5}

Saved: qc_score_distribution.png
```

---

## ✅ Validation & Sanity Checks

### Success Indicators

**Score Distribution:**
- ✅ Reference samples have high scores (> threshold)
- ✅ Clear separation between reference and known outliers
- ✅ Evaluation samples fall into two distinct groups (authentic vs suspect)

**Metrics (if labels available):**
- ✅ Specificity > 90% (few false positives = low false alarm rate)
- ✅ Sensitivity > 80% (catches most true outliers)
- ✅ Balanced performance (not all predictions "authentic" or all "suspect")

**PCA Visualization:**
- ✅ Reference samples cluster tightly
- ✅ Suspect samples fall outside reference cluster
- ✅ No strong batch effects within reference library

### Failure Indicators

**⚠️ Warning Signs:**

1. **All evaluation samples labeled "authentic" (no suspects detected)**
   - Problem: Threshold too lenient; model not sensitive enough
   - Fix: Lower threshold (increase nu parameter); check if evaluation truly contains outliers

2. **All evaluation samples labeled "suspect" (no authentics)**
   - Problem: Threshold too strict; systematic difference between reference and evaluation
   - Fix: Raise threshold; check preprocessing consistency; verify reference library representative

3. **Reference samples score below threshold (self-rejection)**
   - Problem: Model overfitting; threshold miscalibrated
   - Fix: Increase nu; simplify model (reduce gamma in OC-SVM); check for outliers in reference

4. **Score distribution unimodal (no separation)**
   - Problem: Model not discriminating; evaluation too similar to reference
   - Fix: Try alternative model (IsolationForest vs OC-SVM); check if spectral differences exist

### Quality Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|--------|
| Specificity (if labels) | 85% | 92% | 98% |
| Sensitivity (if labels) | 70% | 85% | 95% |
| Reference Self-Acceptance | 90% | 95% | 99% |
| Score Separation (suspect-authentic) | 0.2 | 0.5 | 1.0 |

---

## ⚙️ Parameters You Must Justify

### Critical Parameters

**1. Model Type**
- **Parameter:** `model_type` ("oneclass_svm" or "isolation_forest")
- **Default:** "oneclass_svm"
- **When to adjust:** Use IsolationForest if reference very large (>1000 samples) or high-dimensional
- **Justification:** "One-class SVM (RBF kernel) was used to model the reference distribution, as it handles nonlinear boundaries and is robust to small outliers."

**2. Threshold (nu parameter)**
- **Parameter:** `nu` (expected outlier fraction in reference)
- **Default:** 0.05 (5% outliers expected)
- **When to adjust:** Increase (0.10) if reference noisy; decrease (0.01) if very clean
- **Justification:** "nu=0.05 was chosen to allow 5% of reference samples as support vectors, balancing sensitivity to true outliers vs false alarms."

**3. Preprocessing Consistency**
- **Parameter:** Same baseline, normalization, cropping for reference and evaluation
- **Critical:** Must be identical
- **Justification:** "Reference and evaluation samples were preprocessed identically (ALS baseline, L2 normalization) to ensure scores comparable."

---

```mermaid
flowchart LR
  subgraph Data
    A[Reference library] --> D[New samples]
  end
  subgraph Preprocess
    B[Baseline + norm + crop]
  end
  subgraph Model/Stats
    C[OC-SVM / IsolationForest]
    F[Scores + threshold + optional PCA]
  end
  subgraph Report
    G[Plots (scores, PCA) + report.md]
  end
  A --> B --> C --> F --> G
  D --> B
  B --> F
```

## What? / Why? / When? / Where?
- **What:** One-class QC workflow (preprocess, train OC-SVM/IsolationForest on references, score evaluation samples, threshold into authentic/suspect).  
- **Why:** Detect drift/off-spec batches before release; supplement chemical QC.  
- **When:** Reference library available; evaluation batches incoming; labels may be absent. Limitations: threshold choice, small reference sets, imbalance.  
- **Where:** Upstream preprocessing identical for ref/eval; downstream metrics (specificity/sensitivity if labels), ratio tests, reporting.

## 1. Problem and dataset
- **Inputs:** Reference spectra (authentic) in HDF5; evaluation samples to score. Labels optional.  
- **Typical size:** Dozens–hundreds of references; evaluation count varies.

## 2. Pipeline (default)
- **Preprocessing:** Same stack as authentication (baseline, smoothing, normalization, crop).  
- **Model:** One-class SVM (RBF) or IsolationForest trained on references only.  
- **Threshold:** Default uses decision scores (median/quantile); can be tightened/loosened.  
- **Outputs:** Scores, predicted labels (authentic/suspect), threshold.

## 3. Python example (synthetic)
```python
from examples.qc_quickstart import _synthetic_qc
from foodspec.apps.qc import run_qc_workflow

fs = _synthetic_qc()
train_mask = fs.metadata["group"] == "auth_ref"
res = run_qc_workflow(fs, train_mask=train_mask, model_type="oneclass_svm")
print(res.labels_pred.value_counts())
print("Threshold:", res.threshold)
```

## 4. CLI example (with config)
Create `examples/configs/qc_quickstart.yml`:
```yaml
input_hdf5: libraries/qc_ref.h5
label_column: ""   # optional, if you want to inspect labels
model_type: oneclass_svm
```
Run:
```bash
foodspec qc --config examples/configs/qc_quickstart.yml --output-dir runs/qc_demo
```
Outputs: `qc_scores.csv` with scores and predicted labels, summary.json.

## 5. Interpretation
- Scores near/below threshold → suspect; above → authentic (for the default high-score-normal convention).  
- Investigate suspects with additional tests (chemical assays, microscopy).  
- Main reporting: counts of authentic/suspect; optional histograms of scores; parameters/thresholds.

### Qualitative & quantitative interpretation
- **Qualitative:** Score histograms show separation between reference and new batches; PCA scores (optional) can highlight outliers.  
- **Quantitative:** If labels exist, compute specificity/sensitivity and a confusion matrix. Silhouette on PCA scores (if used) can quantify structure; tests on key ratios (t-test/ANOVA/Games–Howell) can support suspicion (link to [Hypothesis testing](../methods/statistics/hypothesis_testing_in_food_spectroscopy.md)).  
- **Reviewer phrasing:** “Most evaluation samples score above the QC threshold; suspects (n=…) are supported by lower ratio values (t-test p < …) and lower PCA silhouette.”

## Summary
- Train a one-class model on authentic references; score new batches with identical preprocessing.  
- Tune threshold to balance sensitivity/specificity; document settings for audits.

## Statistical analysis
- **Why:** Complement QC scores with comparisons of key ratios or PCs between reference and suspect sets.  
- **Example (two-sample t-test on a ratio):**
```python
import pandas as pd
from foodspec.stats import run_ttest
from foodspec.apps.qc import run_qc_workflow
from examples.qc_quickstart import _synthetic_qc

fs = _synthetic_qc()
train_mask = fs.metadata["group"] == "auth_ref"
qc_res = run_qc_workflow(fs, train_mask=train_mask, model_type="oneclass_svm")
# Suppose we computed a ratio per sample (not shown here); fake example:
df = pd.DataFrame({"ratio": [1.0,1.1,1.0,1.2,1.8,1.9], "group": ["ref","ref","ref","ref","eval","eval"]})
res = run_ttest(df[df["group"]=="ref"]["ratio"], df[df["group"]=="eval"]["ratio"])
print(res.summary)
```
- **Interpretation:** If t-test shows a significant shift in ratio between reference and evaluation, it supports the QC suspicion; otherwise the spectral difference may be minor.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for batch QC workflow:**

1. **Reference spectra collected on different day/instrument than evaluation batch**
   - Drift or calibration differences can exceed batch differences
   - Impossible to know if detected difference is real or instrumental
   - **Fix:** Collect reference and evaluation on same instrument/conditions; include instrumental blanks and controls

2. **QC decision boundary chosen post-hoc to match batch labels (tuning threshold after seeing results)**
   - Data-dependent thresholds overfit; new batches won't match boundary
   - Reproducibility requires pre-set criteria
   - **Fix:** Define QC limits (control chart bounds, ratio thresholds) before batch evaluation; document basis

3. **No positive/negative controls in QC run (no known good/bad sample for comparison)**
   - Without internal controls, drift or contamination goes undetected
   - Can't distinguish batch failure from instrumental failure
   - **Fix:** Include positive control (pass standard), negative control (fail standard), and blank in each QC run

4. **Single metric used for QC without redundancy (only peak ratio A/B, ignore others)**
   - Single metric can be confounded (e.g., peak A sensitive to pH, peak B to temperature)
   - Multiple metrics provide robustness
   - **Fix:** Use multiple orthogonal metrics (e.g., 2–3 independent ratios); flag if metrics disagree

5. **Batch-to-batch variation in spectra (different operators, times, prep) not quantified**
   - Unknown normal variation; can't distinguish batch issues from daily noise
   - QC limits set too tight (false failures) or too loose (miss real problems)
   - **Fix:** Quantify batch variability over time; set QC control limits based on baseline distribution

6. **QC workflow applied without periodic revalidation (model trained in 2023, used in 2024 without retesting)**
   - Instrument drift, aging, or calibration changes model assumptions
   - Old QC limits may become inappropriate
   - **Fix:** Periodically revalidate QC criteria; plot control charts; retrain if drift detected

7. **Failed batches discarded without investigation (batch fails QC, gets tossed, no root cause analysis)**
   - Miss opportunities to understand failure modes
   - Same problem may recur
   - **Fix:** Document failure reason for every failed batch; track trends; address root causes

8. **QC decision time too tight (decide batch pass/fail in minutes based on single measurement)**
   - Insufficient time for replication, controls, or troubleshooting
   - Pressure to pass can lead to overlooking problems
   - **Fix:** Build in time for replication (≥3 repeats); require agreement before batch acceptance

## Further reading
- [Normalization & smoothing](../methods/preprocessing/normalization_smoothing.md)  
- [Classification & regression](../methods/chemometrics/classification_regression.md)  
- [Model evaluation](../methods/chemometrics/model_evaluation_and_validation.md)  
- [Hyperspectral mapping](spatial/hyperspectral_mapping.md)
