# Baseline Correction Validation

This example script (`examples/validation_preprocessing_baseline.py`) generates synthetic Raman-like spectra with a known polynomial baseline and Gaussian peaks, then compares:

- ALSBaseline
- RubberbandBaseline
- PolynomialBaseline

It computes:
- Non-peak region mean before/after correction
- RMSE between estimated and true baseline

Run:

```bash
python examples/validation_preprocessing_baseline.py
```

This produces `validation_baseline.png` and prints a summary table in the console.
> **Status:** Archived  
> This page describes an earlier iteration of baseline validation. See the current preprocessing guides in [ftir_raman_preprocessing.md](ftir_raman_preprocessing.md) and workflow-specific validation in [oil_auth_tutorial.md](oil_auth_tutorial.md).
