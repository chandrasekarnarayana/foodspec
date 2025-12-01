# Architecture & design

FoodSpec is a headless, reproducible toolkit aligned with FAIR principles and MethodsX-style protocols. It offers both CLI and Python APIs over the same core components.

## Module layers
- **core**: `FoodSpectrumSet`, `HyperSpectralCube` — canonical data structures.
- **preprocess**: baseline/smoothing/normalization/cropping, FTIR/Raman helpers — sklearn-style transformers.
- **features & chemometrics**: peaks/bands/ratios, PCA/PLS, classifier factory, mixture analysis — reusable building blocks.
- **io & data**: load/save libraries (HDF5), CSV import, public/example loaders.
- **apps**: workflows (oil, heating, QC, domains, protocol benchmarks, MethodsX reproduction) that orchestrate preprocess + features + models.
- **viz**: plotting utilities (spectra, PCA, classification, heating, hyperspectral).
- **reporting/logging/config**: standardized run folders, JSON/Markdown reports, logging metadata, config loading.

## Flow (textual/ASCII)
```
Raw data (CSV/TXT/HDF5) --> io/data loaders --> core(FoodSpectrumSet/HyperSpectralCube)
      |                                 |
      v                                 v
preprocess transformers (baseline, smoothing, norm, crop, FTIR/Raman helpers)
      |
      v
features/chemometrics (peaks/ratios/PCA/PLS/mixture/models)
      |
      v
apps (oil, heating, QC, domains, protocol, MethodsX)
      |
      v
viz + reporting + logging (plots, metrics.json, report.md, run_metadata.json)
```

Design principles:
- **Headless & reproducible**: deterministic pipelines, run artifacts, config-driven CLI.
- **FAIR**: keep spectra + metadata together; rely on public datasets when possible; clear provenance.
- **Dual interface**: CLI for turnkey use; Python API for customization.
- **Composability**: sklearn-style transformers and pipelines for preprocessing and features.
