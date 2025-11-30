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
