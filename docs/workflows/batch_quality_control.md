# Workflow: Batch Quality Control / Novelty Detection

> New to workflow design? See [Designing & reporting workflows](workflow_design_and_reporting.md).  
> For model/evaluation choices, see [ML & DL models](../ml/models_and_best_practices.md) and [Metrics & evaluation](../metrics/metrics_and_evaluation/).

QC/novelty detection answers “Does this batch look like my reference library?” It is useful for screening incoming materials, production lots, or detecting drift.

Suggested visuals: score histograms, boxplots of key ratios, confusion matrix if labels exist, correlation scatter/heatmap of QC metrics vs batch attributes. See [Plots guidance](workflow_design_and_reporting.md#plots-visualizations).
For troubleshooting (class imbalance, outliers), see [Common problems & solutions](../troubleshooting/common_problems_and_solutions.md).

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
- **Quantitative:** If labels exist, compute specificity/sensitivity and a confusion matrix. Silhouette on PCA scores (if used) can quantify structure; tests on key ratios (t-test/ANOVA/Games–Howell) can support suspicion (link to [Hypothesis testing](../stats/hypothesis_testing_in_food_spectroscopy.md)).  
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
- [Normalization & smoothing](../preprocessing/normalization_smoothing.md)  
- [Classification & regression](../ml/classification_regression.md)  
- [Model evaluation](../ml/model_evaluation_and_validation.md)  
- [Hyperspectral mapping](hyperspectral_mapping.md)
