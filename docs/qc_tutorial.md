# Quality control / novelty detection

Questions this page answers
- Is a new batch in-distribution compared to a reference library?
- How do I train/apply a one-class model (Python and CLI)?
- How do I interpret QC flags?
- How should I report QC findings?

## Use case
Train on reference (authentic) spectra and score new samples to flag suspects (outliers/adulterated/off-spec).

## Running the workflow
CLI:
```bash
foodspec qc libraries/oils.h5 \
  --model-type oneclass_svm \
  --output-dir runs/qc_demo
```
Outputs: qc_scores.csv (scores + labels), summary.json/report.md.

Python:
```python
from foodspec.data import load_library
from foodspec.apps.qc import train_qc_model, apply_qc_model
fs = load_library("libraries/oils.h5")
model = train_qc_model(fs, model_type="oneclass_svm")
res = apply_qc_model(fs, model=model)
print(res.labels_pred.value_counts())
```

## Interpretation
- Scores above threshold → “authentic”; below → “suspect”.  
- Check distribution of scores; adjust threshold if needed (domain knowledge).  
- A QC failure may indicate adulteration, contamination, instrument drift, or out-of-spec batch; verify with orthogonal tests.

## Reporting
- Main: histogram of scores with threshold; counts of authentic vs suspect.  
- Supplementary: per-batch scores, thresholds used, preprocessing steps, model parameters.  
- Document reference set composition and any label filters used for training.

See also
- `csv_to_library.md`
- `metrics_interpretation.md`
- `reporting_guidelines.md`
