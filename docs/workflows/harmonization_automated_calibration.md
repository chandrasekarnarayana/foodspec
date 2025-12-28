# Harmonization: Automated Calibration Curves & Multi-Instrument Alignment

## 📋 Standard Header

**Purpose:** Align spectra from multiple instruments to enable cross-instrument model transfer and harmonized datasets.

**When to Use:**
- Combine spectral data from different instruments into one unified dataset
- Transfer trained model from reference instrument to new instruments
- Correct for wavenumber shifts between instruments
- Normalize intensity differences due to laser power or detector sensitivity
- Build multi-site collaborative datasets with instrument harmonization

**Inputs:**
- Format: Multiple HDF5 files (one per instrument) with overlapping samples
- Required metadata: `instrument_id` (unique identifier per instrument)
- Optional metadata: `laser_power_mw`, `integration_time_ms` (for intensity normalization)
- Wavenumber range: Overlapping range across all instruments (typically 600–1800 cm⁻¹)
- Min samples: 20+ shared samples per instrument pair for calibration curve estimation

**Outputs:**
- calibration_curves.pkl — Wavenumber shift corrections for each instrument
- harmonized_datasets/ — Aligned HDF5 files on common wavenumber grid
- diagnostics.csv — Shift (cm⁻¹), correlation, RMSE for each instrument pair
- alignment_plot.png — Spectra before/after alignment for visual verification
- report.md — Summary of harmonization quality and recommended corrections

**Assumptions:**
- Instruments measure same samples (overlapping subset required for calibration)
- Wavenumber shifts are linear and consistent (global offset, not sample-dependent)
- Intensity differences due to instrument factors, not chemical differences
- Reference instrument spectra are high-quality baseline for alignment

---

## 🔬 Minimal Reproducible Example (MRE)

```python
import numpy as np
import matplotlib.pyplot as plt
from foodspec import SpectralDataset
from foodspec.harmonization import estimate_calibration_curve, harmonize_datasets_advanced
from foodspec.viz.harmonization import plot_harmonization_comparison

# Generate synthetic multi-instrument data
def generate_synthetic_multi_instrument(n_samples=50, shift_cm=3.0, intensity_scale=1.2, random_state=42):
    """Create reference and target instrument datasets with known shift."""
    np.random.seed(random_state)
    wavenumbers_ref = np.linspace(600, 1800, 400)
    wavenumbers_target = wavenumbers_ref + shift_cm  # Simulated wavenumber shift
    
    spectra_ref = []
    spectra_target = []
    
    for i in range(n_samples):
        # Base spectrum
        spectrum = 1.5 * np.exp(-((wavenumbers_ref - 1655) ** 2) / 2000) + \
                   1.2 * np.exp(-((wavenumbers_ref - 1450) ** 2) / 1500) + \
                   np.random.normal(0, 0.05, len(wavenumbers_ref))
        
        spectra_ref.append(spectrum)
        
        # Target spectrum (shifted + intensity scaled)
        spectrum_target = intensity_scale * spectrum + np.random.normal(0, 0.03, len(wavenumbers_ref))
        spectra_target.append(spectrum_target)
    
    import pandas as pd
    
    # Create reference dataset
    df_ref = pd.DataFrame(
        np.array(spectra_ref),
        columns=[f"{w:.1f}" for w in wavenumbers_ref]
    )
    df_ref.insert(0, 'sample_id', [f"S{i:03d}" for i in range(n_samples)])
    df_ref.insert(1, 'instrument_id', 'InstrumentA')
    
    fs_ref = SpectralDataset.from_dataframe(
        df_ref,
        metadata_columns=['sample_id', 'instrument_id'],
        intensity_columns=[f"{w:.1f}" for w in wavenumbers_ref],
        wavenumber=wavenumbers_ref
    )
    
    # Create target dataset (with shift)
    df_target = pd.DataFrame(
        np.array(spectra_target),
        columns=[f"{w:.1f}" for w in wavenumbers_target]
    )
    df_target.insert(0, 'sample_id', [f"S{i:03d}" for i in range(n_samples)])
    df_target.insert(1, 'instrument_id', 'InstrumentB')
    
    # Interpolate to common grid for SpectralDataset
    spectra_target_interp = []
    for spectrum in spectra_target:
        spectrum_interp = np.interp(wavenumbers_ref, wavenumbers_target, spectrum)
        spectra_target_interp.append(spectrum_interp)
    
    df_target_interp = pd.DataFrame(
        np.array(spectra_target_interp),
        columns=[f"{w:.1f}" for w in wavenumbers_ref]
    )
    df_target_interp.insert(0, 'sample_id', [f"S{i:03d}" for i in range(n_samples)])
    df_target_interp.insert(1, 'instrument_id', 'InstrumentB')
    
    fs_target = SpectralDataset.from_dataframe(
        df_target_interp,
        metadata_columns=['sample_id', 'instrument_id'],
        intensity_columns=[f"{w:.1f}" for w in wavenumbers_ref],
        wavenumber=wavenumbers_ref
    )
    
    return fs_ref, fs_target, shift_cm, intensity_scale

# Generate data
fs_ref, fs_target, true_shift, true_scale = generate_synthetic_multi_instrument(
    n_samples=50,
    shift_cm=3.0,
    intensity_scale=1.2
)
print(f"Reference instrument: {fs_ref.metadata['instrument_id'].iloc[0]}")
print(f"Target instrument: {fs_target.metadata['instrument_id'].iloc[0]}")
print(f"True wavenumber shift: {true_shift:.1f} cm⁻¹")
print(f"True intensity scale: {true_scale:.2f}")

# Estimate calibration curve
curve, diagnostics = estimate_calibration_curve(fs_ref, fs_target)
print(f"\nEstimated shift: {diagnostics['shift_cm']:.2f} cm⁻¹")
print(f"Correlation: {diagnostics['corr_coeff']:.4f}")
print(f"RMSE: {diagnostics['rmse']:.4f}")

# Harmonize datasets
harmonized = harmonize_datasets_advanced(
    [fs_ref, fs_target],
    calibration_curves={'InstrumentB': curve},
    intensity_meta_key=None  # or 'laser_power_mw' if available
)

print(f"\nHarmonization complete:")
for inst_id, ds in harmonized.items():
    print(f"  {inst_id}: {ds.x.shape[0]} spectra on common grid")

# Plot before/after comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before harmonization
axes[0].plot(fs_ref.wavenumber, fs_ref.x[0], label='Reference', alpha=0.7)
axes[0].plot(fs_target.wavenumber, fs_target.x[0], label='Target (before)', alpha=0.7)
axes[0].set_xlabel('Wavenumber (cm⁻¹)')
axes[0].set_ylabel('Intensity')
axes[0].set_title('Before Harmonization')
axes[0].legend()
axes[0].grid(alpha=0.3)

# After harmonization
axes[1].plot(harmonized['InstrumentA'].wavenumber, harmonized['InstrumentA'].x[0], label='Reference', alpha=0.7)
axes[1].plot(harmonized['InstrumentB'].wavenumber, harmonized['InstrumentB'].x[0], label='Target (after)', alpha=0.7)
axes[1].set_xlabel('Wavenumber (cm⁻¹)')
axes[1].set_ylabel('Intensity')
axes[1].set_title('After Harmonization')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("harmonization_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: harmonization_comparison.png")
```

**Expected Output:**
```yaml
Reference instrument: InstrumentA
Target instrument: InstrumentB
True wavenumber shift: 3.0 cm⁻¹
True intensity scale: 1.20

Estimated shift: 2.98 cm⁻¹
Correlation: 0.9856
RMSE: 0.0234

Harmonization complete:
  InstrumentA: 50 spectra on common grid
  InstrumentB: 50 spectra on common grid

Saved: harmonization_comparison.png
```

---

## ✅ Validation & Sanity Checks

### Success Indicators

**Diagnostics:**
- ✅ Estimated shift close to known/expected shift (if calibration standard used)
- ✅ Correlation > 0.95 between reference and aligned target
- ✅ RMSE < 0.05 (normalized intensity units)

**Visual Comparison:**
- ✅ Before: Spectra misaligned (peaks shifted)
- ✅ After: Spectra overlapping (peaks aligned)
- ✅ Peak positions match across instruments

**Model Transfer:**
- ✅ Model trained on reference instrument performs well on harmonized target data
- ✅ Cross-instrument CV accuracy similar to within-instrument CV

### Failure Indicators

**⚠️ Warning Signs:**

1. **Correlation < 0.85 after harmonization**
   - Problem: Non-linear shifts; different spectral features; sample mismatch
   - Fix: Verify same samples measured; check for different baseline/normalization; use more calibration samples

2. **Estimated shift > 10 cm⁻¹**
   - Problem: Instruments very poorly calibrated; using different spectral ranges
   - Fix: Recalibrate instruments; verify wavenumber axes correct; check if Raman vs FTIR

3. **Harmonized spectra still visually misaligned**
   - Problem: Linear shift insufficient; baseline differences; matrix effects
   - Fix: Apply baseline correction before harmonization; use piecewise linear or polynomial calibration curves

4. **Model transfer accuracy drops > 20% on harmonized data**
   - Problem: Intensity normalization inadequate; features not transferable; batch effects
   - Fix: Include intensity metadata (laser power); use robust features (ratios); train on multi-instrument data

### Quality Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|--------|
| Correlation (after) | 0.85 | 0.95 | 0.99 |
| RMSE (normalized) | < 0.10 | < 0.05 | < 0.02 |
| Shift Estimation Error | < 2 cm⁻¹ | < 1 cm⁻¹ | < 0.5 cm⁻¹ |
| Model Transfer Accuracy Drop | < 20% | < 10% | < 5% |

---

## ⚙️ Parameters You Must Justify

### Critical Parameters

**1. Reference Instrument**
- **Parameter:** `reference_instrument_id`
- **No default:** Must specify
- **Justification:** "InstrumentA chosen as reference due to highest SNR and largest calibration dataset."

**2. Max Shift Search Range**
- **Parameter:** `max_shift_points` (search window for wavenumber shift)
- **Default:** 25 points (~6 cm⁻¹ at 0.25 cm⁻¹ resolution)
- **When to adjust:** Increase (50) if instruments very misaligned; decrease (10) if well-calibrated
- **Justification:** "Max shift of ±6 cm⁻¹ searched to accommodate typical Raman calibration drift."

**3. Intensity Normalization Key**
- **Parameter:** `intensity_meta_key` (metadata column for intensity correction)
- **Default:** None (no intensity normalization)
- **When to adjust:** Use 'laser_power_mw' if available and variable across instruments
- **Justification:** "Intensity normalized by laser power (metadata: laser_power_mw) to correct for instrument-specific detector sensitivity."

**4. Common Wavenumber Grid**
- **Parameter:** Interpolation target grid (start, stop, resolution)
- **Default:** Intersection of all instrument ranges at finest resolution
- **Justification:** "All spectra interpolated to common 600–1800 cm⁻¹ grid at 1 cm⁻¹ resolution for harmonization."

---

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
- [Multi-Instrument HSI Workflows](../theory/harmonization_theory.md)
- [Calibration Transfer](quantification/calibration_regression_example.md)
- [Data Governance & Quality](../user-guide/data_governance.md)
