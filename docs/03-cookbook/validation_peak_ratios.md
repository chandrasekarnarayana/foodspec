# Peak Ratio Validation

This example script (`examples/validation_peak_ratios.py`) generates synthetic spectra with two Gaussian peaks at 1655 and 1742 cm^-1 with varying height ratios.

It uses:
- `PeakFeatureExtractor`
- `RatioFeatureGenerator`

Outputs:
- Scatter plot of true ratio vs measured ratio.
- Correlation and RMSE between true and measured ratios.

Run:

```bash
python examples/validation_peak_ratios.py
```

This produces `validation_peak_ratios.png` and prints summary statistics.
> **Status:** Archived  
> This page reflects older peak-ratio validation examples. Refer to the current workflows in [../workflows/oil_authentication.md](../workflows/oil_authentication.md) and metrics guidance in [../metrics/metrics_and_evaluation.md](../metrics/metrics_and_evaluation.md).
