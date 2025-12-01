# Command-line reference

Major commands (see `foodspec about` for full list):

- **about**: print version, Python info, optional extras status.
- **csv-to-library**: convert CSV (wide/long) to HDF5 spectral library.
- **preprocess**: run default preprocessing on a folder of spectra + metadata, output HDF5.
- **oil-auth**: oil authentication workflow (preprocess → features → classifier → report).
- **heating**: heating degradation ratios vs time, optional ANOVA, report folder.
- **qc**: novelty detection (one-class SVM / IsolationForest) with scores/labels report.
- **domains**: dairy/meat/microbial templates; same pattern as oil-auth.
- **mixture**: NNLS or MCR-ALS decomposition of mixtures against pure spectra.
- **hyperspectral**: build cube from per-pixel spectra and plot intensity map.
- **protocol-benchmarks**: standardized benchmark suite on public datasets.
- **reproduce-methodsx**: run MethodsX protocol reproduction (classification + mixture + PCA).
- **model-info**: inspect saved model metadata (from model registry).

Example:
```bash
foodspec oil-auth libraries/oils_demo.h5 --label-column oil_type --output-dir runs/oil_demo
```
Outputs: metrics.json/CSV, confusion_matrix.png, report.md in a timestamped folder.
