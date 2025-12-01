# Troubleshooting & FAQs

## Common issues
- **Missing label column**: ensure metadata includes the column passed to `--label-column` or used in Python (`fs.metadata`).
- **Non-monotonic wavenumbers**: sort axes before creating a library; `validate_spectrum_set` will fail otherwise.
- **HDF5 load errors**: confirm the file was created by foodspec (`create_library` or `foodspec preprocess/csv-to-library`).
- **Small class sizes**: cross-validation may fail if each class has fewer than 2 samples; add more data or reduce `cv_splits`.
- **NaNs in data**: impute or filter; many models do not accept NaNs.

## FAQs
- **Can foodspec handle non-food spectra?** Yes; it is domain-agnostic but tuned for food spectroscopy defaults.
- **What accuracy is “good”?** Depends on task and dataset; use protocol benchmarks as a reference and report F1/CM plots.
- **How do I choose preprocessing?** Start with ALS + Savitzky-Golay + Vector/MSC; consult `preprocessing_guide.md`.
- **Where are reports written?** CLI commands create timestamped folders under `--output-dir` with metrics, plots, and markdown summaries.
- **Can I customize models?** Yes; use the Python API to build your own pipelines or swap classifiers via CLI flags.
