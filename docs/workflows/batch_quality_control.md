# Workflow: Batch Quality Control / Novelty Detection

> New to workflow design? See [Designing & reporting workflows](workflow_design_and_reporting.md).
> For model/evaluation choices, see [ML & DL models](../ml/models_and_best_practices.md) and [Metrics & evaluation](../metrics/metrics_and_evaluation.md).

QC/novelty detection answers “Does this batch look like my reference library?” It is useful for screening incoming materials, production lots, or detecting drift.

Suggested visuals: score histograms, boxplots of key ratios, confusion matrix if labels exist, correlation scatter/heatmap of QC metrics vs batch attributes. See [Plots guidance](workflow_design_and_reporting.md#plots-visualizations).
For troubleshooting (class imbalance, outliers), see [Common problems & solutions](../troubleshooting/common_problems_and_solutions.md).

```mermaid
flowchart LR
  A[Reference library (authentic)] --> B[Preprocessing (baseline, norm, crop)]
  B --> C[Train one-class model (OC-SVM/IsolationForest)]
  D[New samples] --> E[Preprocess with same pipeline]
  C --> F[Score new samples]
  E --> F
  F --> G[Threshold → authentic / suspect + report]
```

## 1. Problem and dataset
- **Why labs care:** Early detection of off-spec batches, supplier changes, or process drift.
- **Inputs:** Reference spectra (authentic) in HDF5; evaluation samples to score. Labels optional for inspection.
- **Typical size:** Dozens to hundreds of reference spectra; unknown number of evaluation samples.

## 2. Pipeline (default)
- **Preprocessing:** Same stack as authentication (baseline, smoothing, normalization, crop). Apply identical steps to reference and evaluation.
- **Model:** One-class SVM (RBF) or IsolationForest trained on reference spectra only.
- **Threshold:** Default uses decision scores (median or quantile); can be set manually for stricter/looser QC.
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
- Investigate suspects with additional tests (chemical assays, microscopy) to confirm.
- Main reporting: counts of authentic/suspect; optional histograms of scores. Supplement: parameters/thresholds.

### Qualitative & quantitative interpretation
- **Qualitative:** Score histograms show separation between reference and new batches; PCA scores (optional) can highlight outliers.
- **Quantitative:** Report counts of authentic/suspect; if labels exist, compute specificity/sensitivity and a confusion matrix. Silhouette on PCA scores (if used) can quantify structure; two-sample tests on key ratios can support suspicion (link to [Hypothesis testing](../stats/hypothesis_testing_in_food_spectroscopy.md)).
- **Reviewer phrasing:** “Most evaluation samples score above the QC threshold; suspects (n=…) are supported by lower ratio values (t-test p < …) and lower PCA silhouette.”

## Summary
- Train a one-class model on authentic references; score new batches with identical preprocessing.
- Tune threshold to balance sensitivity/specificity; document settings for audits.

## Statistical analysis
- **Why:** Complement QC scores with simple comparisons of key ratios or PCs between reference and suspect sets.
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

## Further reading
- [Normalization & smoothing](../preprocessing/normalization_smoothing.md)
- [Classification & regression](../ml/classification_regression.md)
- [Model evaluation](../ml/model_evaluation_and_validation.md)
- [Hyperspectral mapping](hyperspectral_mapping.md)
