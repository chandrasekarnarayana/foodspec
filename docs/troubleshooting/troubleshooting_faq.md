# Troubleshooting & FAQs

**Purpose:** Quickly diagnose and fix common issues in FoodSpec workflows.  
**Audience:** Users troubleshooting preprocessing, modeling, or I/O errors.  
**Time to read:** 5–10 minutes.  
**Prerequisites:** Completed a FoodSpec run; familiar with workflows.

---

## Quick Diagnostic Script

```python
from foodspec.io import load_csv_spectra
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.validation import validate_spectrum_set

# Load and check for common issues
spectra = load_csv_spectra("your_data.csv", id_column="id", wavelength_column="wavenumber")

# Check 1: Non-monotonic wavenumbers
try:
    validate_spectrum_set(spectra)
    print("✓ Spectra are valid.")
except Exception as e:
    print(f"✗ Validation failed: {e}")

# Check 2: NaN values
nan_count = spectra.data.isna().sum().sum()
print(f"NaN count: {nan_count}")

# Check 3: Baseline issues
baseline_corrected = baseline_als(spectra, lam=1e5)
print(f"Baseline correction applied. Range before: {spectra.data.min():.2f}–{spectra.data.max():.2f}")
print(f"  Range after: {baseline_corrected.data.min():.2f}–{baseline_corrected.data.max():.2f}")
```

---

## Common issues
- **Missing label column**: ensure metadata includes the column passed to `--label-column` or used in Python (`fs.metadata`).
- **Non-monotonic wavenumbers**: sort axes before creating a library; `validate_spectrum_set` will fail otherwise.
- **HDF5 load errors**: confirm the file was created by foodspec (`create_library` or `foodspec preprocess/csv-to-library`).
- **Small class sizes**: cross-validation may fail if each class has fewer than 2 samples; add more data or reduce `cv_splits`.
- **NaNs in data**: impute or filter; many models do not accept NaNs.

## FAQs
- **Can foodspec handle non-food spectra?** Yes; it is domain-agnostic but tuned for food spectroscopy defaults.
- **What accuracy is “good”?** Depends on task and dataset; use protocol benchmarks as a reference and report F1/CM plots.
- **How do I choose preprocessing?** Start with ALS + Savitzky-Golay + Vector/MSC; see `ftir_raman_preprocessing.md`.
- **Where are reports written?** CLI commands create timestamped folders under `--output-dir` with metrics, plots, and markdown summaries.
- **Can I customize models?** Yes; use the Python API to build your own pipelines or swap classifiers via CLI flags.
---

## Next Steps

- [Common Problems & Solutions](common_problems_and_solutions.md) — Detailed diagnosis and fixes for each issue type.
- [Reporting Guidelines](reporting_guidelines.md) — How to document your troubleshooting and validation steps.
- [CSV to Library](../user-guide/csv_to_library.md) — Prevent issues upstream with correct data handling.