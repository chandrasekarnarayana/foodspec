# Chemometrics Validation: Oils

This page describes a typical oil authentication pipeline and how to run the protocol benchmarks.

## Oil authentication workflow

The `run_oil_authentication_workflow` function implements a typical pipeline:

1. Load a spectral library or public dataset.  
2. Apply baseline correction, smoothing, and normalization.  
3. Extract peak/band ratios.  
4. Train and evaluate a classifier (e.g., Random Forest).

Example (Python API):

```python
from foodspec.data.libraries import load_library
from foodspec.apps.oils import run_oil_authentication_workflow

fs = load_library("libraries/oils_raman.h5")
result = run_oil_authentication_workflow(fs, label_column="oil_type", classifier_name="rf", cv_splits=3)
print(result.cv_metrics)
print("Confusion matrix:\n", result.confusion_matrix)
```

You can also run the example script:

```bash
python examples/validation_chemometrics_oils.py
```

This produces PCA and confusion matrix plots in the working directory.

## Protocol benchmarks (CLI)

The `foodspec protocol-benchmarks` command runs a standardized benchmark suite using public datasets (if available) or example data:

```bash
foodspec protocol-benchmarks --output-dir ./protocol_benchmarks
```

Each run produces a timestamped directory containing:

- `classification_metrics.json` – oil classification metrics
- `classification_confusion_matrix.csv`
- `mixture_metrics.json` – EVOO–sunflower mixture regression metrics
- `run_metadata.json` – environment and version information
- `report.md` – human-readable summary

If datasets are missing, the command reports clear errors; download public datasets into the expected folders (see `docs/libraries.md`).
