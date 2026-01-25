# Design Philosophy

FoodSpec is a protocol-driven spectroscopy framework for food matrices. The
architecture mirrors the mindmap and makes goals and non-goals explicit.

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

## Feature map (mindmap → module)

| Mindmap node | Module path | Public API | CLI command | Artifacts |
| --- | --- | --- | --- | --- |
| Data Objects | `foodspec.data_objects` | `Spectrum`, `SpectraSet`, `SpectralDataset` | `foodspec io validate` | `protocol.yaml`, `run_summary.json` |
| Data Extraction | `foodspec.io` | `read_spectra`, `detect_format` | `foodspec io validate` | ingest logs |
| Programming Engine | `foodspec.engine` | preprocessing pipeline | `foodspec preprocess run` | preprocessing logs |
| Quality Control | `foodspec.qc` | QC engine, dataset checks | `foodspec qc spectral|dataset` | `qc_results.json` |
| Feature Engineering | `foodspec.features` | peaks, ratios, chemometrics | `foodspec features extract` | feature tables |
| Modeling & Validation | `foodspec.modeling` | model factories, validation | `foodspec train` | `metrics.json` |
| Trust & Uncertainty | `foodspec.trust` | calibration, conformal | `foodspec report` | `uncertainty_metrics.json` |
| Visualization & Reporting | `foodspec.viz`, `foodspec.reporting` | plots, reports | `foodspec report` | HTML/PDF reports |

