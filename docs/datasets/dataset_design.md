# Dataset Design & Best Practices

Questions this page answers:
- How should I design spectral datasets for FoodSpec workflows?
- What metadata, replicates, and balance are needed?
- How do acquisition and instrument choices affect downstream analysis?

## Goals of good dataset design
- Align data collection with the scientific question (authentication, adulteration detection, calibration, QC).
- Ensure reproducibility: clear metadata, consistent wavenumber axes, documented preprocessing.
- Support robust modeling: enough samples per class/group, balanced design when possible, capture variability (batches, instruments).

## Core principles
- **Metadata completeness:** sample ID, class/target label, instrument, acquisition date, integration/laser settings, replicates, batch/lot.
- **Replicates & balance:** aim for multiple replicates per class; avoid extreme imbalance for classifiers; for rare events, collect targeted positives.
- **Wavenumber integrity:** ascending cm⁻¹ axis; document spectral range/resolution; align across instruments where possible.
- **Split-aware collection:** plan train/validation/test splits up front; keep batches and instruments represented across splits to test robustness.

## Recommended fields
- Required: `sample_id`, `label` (or target), `instrument`, `modality`, `acquisition_date`.
- Optional but useful: `batch`, `matrix`, `temperature/time`, `preparation`, `operator`.
- For calibration: reference measurements (e.g., peroxide value, moisture) with units and methods.

## Designing for tasks
- **Authentication / multi-class:** balanced classes; capture intra-class variability (different origins/lots); include known confounders.
- **Adulteration / rare positives:** oversample positives; record dilution levels; use PR-oriented metrics; consider anomaly/QC models.
- **Calibration/regression:** cover the full concentration range evenly; avoid only extremes; include replicates to estimate noise.
- **QC/novelty:** collect authentic reference sets across batches/instruments; include known outliers for stress testing.

## Acquisition guidance
- Instrument settings: record laser power, integration time, number of accumulations, spectral resolution, temperature.
- Sample prep: record protocols (mixing, filtration, ATR contact); aim for consistency.
- Instrument variation: if multiple instruments, track IDs; consider alignment/cropping; evaluate cross-instrument robustness.

## Practical checklist
- [ ] Wavenumbers ascending and consistent across all spectra.
- [ ] Metadata columns present and validated (`check_missing_metadata`).
- [ ] Class balance summarized (`summarize_class_balance`).
- [ ] Replicates per class/group > 3 where feasible.
- [ ] Reference targets recorded with units (regression).
- [ ] Planned splits documented; no leakage between train/test.

## Data formats and flows (visual)
Describe the ingestion pipeline so readers know how files become libraries and workflows:
- Raw files (CSV/TXT/vendor) → `read_spectra` → `FoodSpectrumSet`
- `FoodSpectrumSet` → `create_library` → HDF5
- HDF5 → Workflows (preprocess → features → models/stats → reports)
> Reproducible figure: run  
> ```bash
> python docs/examples/visualization/generate_dataset_flow_diagram.py
> ```  
> to save `docs/assets/dataset_flow.png`, a simple schematic of the above pipeline (raw → read_spectra → FoodSpectrumSet → create_library → workflows). Ensure axes (cm⁻¹) and metadata fields are annotated.

## See also
- Instrument formats & loading: [instrument_file_formats](../user_guide/instrument_file_formats.md)
- Workflow design: [workflow_design_and_reporting](../workflows/workflow_design_and_reporting.md)
- Metrics & evaluation: [metrics_and_evaluation](../metrics/metrics_and_evaluation/)
- Troubleshooting dataset issues: [common_problems_and_solutions](../troubleshooting/common_problems_and_solutions.md)
