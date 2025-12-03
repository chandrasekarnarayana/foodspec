# MethodsX protocol reproduction

This page maps the MethodsX protocol paper directly to FoodSpec commands and outputs so that all figures and tables can be reproduced from public datasets.

> If you are a reviewer or reader checking reproducibility, run the command below, then compare the produced metrics/plots with the figures/tables described here. See the Reproducibility checklist and Reporting guidelines for what to archive.

## Command

```bash
foodspec reproduce-methodsx --output-dir runs/methodsx_protocol
```

Produces a timestamped run directory with metrics.json, run_metadata.json, confusion_matrix/PCA plots, and report.md.

## Datasets (public)

1. **Mendeley edible oils (Raman/FTIR)** – multi-class classification.  
2. **EVOO–sunflower Raman mixtures (data.gouv.fr, DOI 10.57745/DOGT0E)** – fraction regression/mixture analysis.  
3. **Groundnut adulteration ATR-MIR (Kaggle)** – optional robustness/adulteration check.  

Convert each to HDF5 using `foodspec csv-to-library` or the public loaders; see Libraries for folder structure.

## Mapping figures/tables to commands

- **PCA scores (Figure)**: generated from the classification dataset; output `oil_pca_scores.png`.  
  - Command: `foodspec reproduce-methodsx` (internal PCA).  
  - Dataset: public oils library (Raman/FTIR).
- **Confusion matrix (Figure)**: `oil_confusion_matrix.png`.  
  - Command: `foodspec reproduce-methodsx` (oil classification step).  
  - Metrics: accuracy, F1 in `metrics.json` (classification section).
- **Classification metrics (Table)**: accuracy/F1 summary in `metrics.json`.  
  - Use overall values for main text; per-class precision/recall can be supplementary if available.
- **Mixture analysis metrics (Table/Plot)**: `mixture_r2`, `mixture_rmse` in `metrics.json`; add predicted vs true fraction plot if desired.  
  - Dataset: EVOO–sunflower library.

## Reproduction checklist

1. Download public datasets and convert to HDF5 libraries (see Libraries and CSV→HDF5 pages).  
2. Run `foodspec reproduce-methodsx --output-dir runs/methodsx_protocol`.  
3. Verify artifacts in the run directory: metrics.json, report.md, PCA/confusion matrix plots.  
4. Align outputs with paper figures/tables (rename or re-caption as needed).  
5. For publication:  
   - **Main**: PCA scores figure, confusion matrix, classification metrics (accuracy/F1), mixture metrics (R²/RMSE).  
   - **Supplementary**: additional per-class metrics, residual plots, spectra examples, run_metadata.json for reproducibility.

See also
- Keyword index: [keyword_index.md](../keyword_index.md)
- Reproducibility: [reproducibility_checklist.md](reproducibility_checklist.md), [reporting_guidelines.md](../reporting_guidelines.md), [benchmarking_framework.md](benchmarking_framework.md)
- Related workflows: [workflows/oil_authentication.md](../workflows/oil_authentication.md), [workflows/mixture_analysis.md](../workflows/mixture_analysis.md)
