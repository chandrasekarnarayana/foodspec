# MethodsX protocol reproduction

This page explains how to reproduce the core analyses described in the MethodsX
protocol article using foodspec.

## Command

```bash
foodspec reproduce-methodsx --output-dir runs/methodsx_protocol
```

This command:

- loads public edible-oil datasets (or synthetic stand-ins, depending on the environment and configuration),
- runs:
  - an oil-type classification analysis,
  - an EVOO–sunflower mixture analysis,
  - PCA visualization of preprocessed spectra,
- and writes:
  - `metrics.json` (classification and mixture metrics),
  - `run_metadata.json` (environment and version info),
  - confusion matrix and PCA plots,
  - a summary `report.md`.

## Mapping to the MethodsX article

The outputs from `foodspec reproduce-methodsx` correspond to:

- Figure X – PCA score plot → `pca_scores.png` in the run directory.
- Figure Y – Confusion matrix for oil-type classification → `confusion_matrix.png`.
- Table 1 – Classification metrics (accuracy, F1, etc.) → `metrics.json` under the "classification" section.
- Table 2 – Mixture-analysis metrics (R², RMSE) → `metrics.json` under the "mixture" section.

(Replace X and Y with actual figure numbers once the manuscript is finalized.)

## Datasets

The reproduction workflow uses:

- a public edible-oil dataset for classification,
- a public EVOO–sunflower mixture dataset for regression,

accessed via the `foodspec.data.public` loaders. See [Libraries](libraries.md)
for instructions on downloading and organizing these datasets.
