# Harmonization: Automated Calibration Curves & Multi-Instrument Alignment

## Overview
FoodSpec provides automated workflows for aligning spectra across multiple instruments via calibration curve estimation and multi-instrument harmonization.

## Calibration Curve Estimation

### Automated Curve Generation
For instruments without pre-computed calibration curves, use automatic curve estimation:

```python
from foodspec import SpectralDataset
from foodspec.harmonization import estimate_calibration_curve, generate_calibration_curves

# Load reference and target instrument datasets
reference_ds = SpectralDataset.from_hdf5("reference_instrument.h5")
target_ds = SpectralDataset.from_hdf5("target_instrument.h5")

# Estimate one curve (reference → target)
curve, diagnostics = estimate_calibration_curve(reference_ds, target_ds)
print(f"Shift: {diagnostics['shift_cm']} cm^-1")
print(f"Correlation: {diagnostics['corr_coeff']:.4f}")
```

### Bulk Calibration Curve Generation
For multiple instruments, generate curves relative to a reference instrument:

```python
from pathlib import Path

# Load all datasets
datasets = [
    SpectralDataset.from_hdf5(f)
    for f in Path("data").glob("*.h5")
]

# Generate curves for all vs. reference instrument
curves, diagnostics = generate_calibration_curves(
    datasets,
    reference_instrument_id="InstrumentA",
    max_shift_points=25  # search range
)

for inst_id, curve in curves.items():
    print(f"{inst_id}: shift = {diagnostics[inst_id]['shift_cm']:.2f} cm^-1")
```

## Multi-Instrument Harmonization Workflow

### Full Harmonization Pipeline
```python
from foodspec.harmonization import harmonize_datasets_advanced

# Apply calibration curves + intensity normalization + common grid alignment
harmonized_datasets, diag = harmonize_datasets_advanced(
    datasets,
    calibration_curves=curves,
    intensity_meta_key="laser_power_mw"  # or None if not available
)

print(f"Target grid length: {len(harmonized_datasets[0].wavenumbers)}")
print(f"Residual variation: {diag['residual_std_mean']:.4f}")
```

### Intensity Normalization
Correct for laser power variations:

```python
from foodspec.harmonization import intensity_normalize_by_power

for ds in datasets:
    power_mw = ds.instrument_meta.get("laser_power_mw")
    ds_norm = intensity_normalize_by_power(ds, power_mw)
```

## Mathematical Assumptions

1. **Calibration Curve Linearity:** Wavenumber drift is modeled as a linear shift (suitable for stable instruments).
2. **Paired Standards:** The reference and target instruments measure the same standard samples (required for curve fitting).
3. **Intensity Additivity:** Laser power affects intensity multiplicatively: $I_{\text{corrected}} = I_{\text{observed}} / P_{\text{mW}}$.
4. **Common Grid Alignment:** Spectra are interpolated to a shared wavenumber grid (assumes smooth spectral features).

## Failure Modes & Diagnostics

### Poor Correlation During Curve Estimation
**Symptom:** Low `corr_coeff` (< 0.8) in diagnostics.  
**Cause:** Spectra too different, or instruments not measuring the same samples.  
**Solution:** Inspect the raw spectra; ensure standards are authentic and representative.

### Residual Variation High
**Symptom:** `residual_std_mean` > expected noise level.  
**Cause:** Incomplete harmonization; possible instrument drift or scale mismatch.  
**Solution:** Increase `max_shift_points` range; verify intensity metadata.

### Misaligned Peaks Post-Harmonization
**Symptom:** Peaks don't align after harmonization.  
**Cause:** Nonlinear wavenumber drift not captured by linear model.  
**Solution:** Use manual calibration points or export all data to vendor software for advanced correction.

## Advanced: Manual Calibration Curves

If automated estimation fails, create curves manually:

```python
from foodspec.harmonization import CalibrationCurve

# Define manually (e.g., from vendor calibration)
curve = CalibrationCurve(
    instrument_id="InstrumentB",
    wn_source=np.array([1000, 1100, 1200, ...]),  # expected wavenumbers
    wn_target=np.array([1001, 1102, 1199, ...])   # observed wavenumbers
)

# Apply to dataset
ds_corrected = apply_calibration(ds, curve)
```

## Outputs & Reproducibility

All harmonization operations are logged in `SpectralDataset.logs` and `SpectralDataset.history`:

```python
print("\n".join(harmonized_datasets[0].logs))
# Output:
# harmonized_to_grid len=3601
# advanced_harmonized_to_grid len=3601
# ...

print(harmonized_datasets[0].history)
# [{'step': 'advanced_harmonize', 'len_grid': 3601}, ...]
```

## See Also
- [Multi-Instrument HSI Workflows](hsi_and_harmonization.md)
- [Calibration Transfer](calibration_transfer.md)
- [Data Governance & Quality](data_governance.md)
