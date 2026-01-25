# FoodSpec

<p align="left">
  <img src="docs/assets/foodspec_logo.png" alt="FoodSpec logo" width="200">
</p>

[![Tests](https://github.com/chandrasekarnarayana/foodspec/actions/workflows/ci.yml/badge.svg)](https://github.com/chandrasekarnarayana/foodspec/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

FoodSpec is a protocol-driven spectroscopy framework for food matrices (Raman, FTIR, NIR). It prioritizes reproducibility, quality control, and trustworthy outputs for scientific workflows.

## Goals
- Protocol-driven spectroscopy workflows
- Reproducibility by default
- Trust and uncertainty as first-class outputs
- QC is mandatory, not optional
- Designed for food matrices with complex backgrounds

## Non-goals
- Not a general deep learning framework
- Not a vendor replacement tool
- Not claiming clinical or regulatory approval

## Features (mindmap-aligned)
- **Data objects**: `Spectrum`, `SpectraSet`, `SpectralDataset` for consistent data and metadata handling.
- **Data extraction**: CSV, JCAMP, SPC, OPUS loaders and vendor adapters.
- **Programming engine**: preprocessing steps (baseline, smoothing, normalization) with reusable pipelines.
- **QC system**: spectral QC, dataset QC, leakage detection, drift monitoring.
- **Feature engineering**: peak ratios, chemometrics (PCA/PLS), minimal marker panels.
- **Modeling & validation**: classical ML, nested CV, confidence/metrics reporting.
- **Trust & uncertainty**: calibration, conformal prediction, reliability metrics.
- **Visualization & reporting**: HTML/PDF reports, figures, reproducibility packs.

## Quickstart (Python)

```python
from foodspec.data_objects import SpectraSet
from foodspec.engine.preprocessing import baseline_als, smooth_savgol, normalize_vector
import numpy as np

# Example spectra
spectra = np.random.rand(5, 100)
wn = np.linspace(400, 1800, 100)

fs = SpectraSet(x=spectra, wavenumbers=wn, metadata=None, modality="raman")

# Preprocess
spectra = baseline_als(spectra)
spectra = smooth_savgol(spectra, window_length=9, polyorder=3)
spectra = normalize_vector(spectra, norm="l2")
```

## Quickstart (CLI)

```bash
# Validate input
foodspec io validate data/oils.csv

# Spectral QC
foodspec qc spectral data/oils.csv --run-dir runs/qc

# Run protocol preprocessing
foodspec preprocess run --protocol examples/configs/oil_auth_quickstart.yml --input data/oils.csv
```

## Documentation
- Design philosophy: `docs/concepts/design_philosophy.md`
- Full docs: https://chandrasekarnarayana.github.io/foodspec/

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src/ tests/
mkdocs build
```

## Citation

See `CITATION.cff` for citation metadata.

