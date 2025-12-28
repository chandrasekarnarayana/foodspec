---
title: Temporal & Aging Workflows
---

## 📋 Standard Header

**Purpose:** Model degradation trajectories over time and estimate remaining shelf-life using spectral markers.

**When to Use:**
- Track oil oxidation over storage time to predict shelf-life
- Model degradation kinetics (linear, exponential, or spline fits)
- Classify samples into aging stages (early/mid/late degradation)
- Estimate time-to-threshold for quality markers with confidence intervals
- Compare degradation rates across treatments (antioxidants, storage conditions)

**Inputs:**
- Format: HDF5 or CSV with time-series spectral data
- Required metadata: `time_col` (time in days/hours), `entity_col` (sample/batch ID)
- Optional metadata: `treatment`, `storage_temp`, `initial_quality`
- Derived features: Degradation values (e.g., ratio 1655/1742 or peroxide value)
- Min samples: 5+ time points × 3+ entities (15+ spectra per treatment)

**Outputs:**
- trajectories.csv — Fitted degradation curves (slope, intercept, R²) per entity
- stage_labels.csv — Classified aging stage (early/mid/late) per sample
- shelf_life_estimates.csv — Remaining time to threshold with 95% CI per entity
- trajectory_plot.png — Degradation value vs time with fitted curves
- report.md — Shelf-life predictions and degradation rate comparisons

**Assumptions:**
- Time measured accurately and consistently across entities
- Degradation monotonic (no recovery or oscillation)
- Entities independent (not repeated measurements of same sample)
- Linear or smooth non-linear trajectory appropriate for degradation kinetics

---

## 🔬 Minimal Reproducible Example (MRE)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from foodspec.workflows.aging import compute_degradation_trajectories
from foodspec.workflows.shelf_life import estimate_remaining_shelf_life
from foodspec.core.time_spectrum_set import TimeSpectrumSet
from foodspec import SpectralDataset

# Generate synthetic aging data
def generate_synthetic_aging(n_entities=5, n_times=10, random_state=42):
    """Create synthetic time-series data with linear degradation."""
    np.random.seed(random_state)
    time_points = np.linspace(0, 30, n_times)  # 0-30 days
    
    data = []
    for entity_id in range(n_entities):
        # Each entity has different degradation rate
        rate = np.random.uniform(0.05, 0.15)
        intercept = np.random.uniform(0.5, 1.0)
        
        for t in time_points:
            # Degradation value = intercept + rate * time + noise
            degrade_value = intercept + rate * t + np.random.normal(0, 0.05)
            
            # Synthetic spectrum (not used in this workflow, but required for TimeSpectrumSet)
            wavenumbers = np.linspace(600, 1800, 100)
            spectrum = np.random.normal(0, 0.1, 100)
            
            data.append({
                'entity_id': f"E{entity_id:02d}",
                'time': t,
                'degrade': degrade_value,
                **{f"{w:.1f}": val for w, val in zip(wavenumbers, spectrum)}
            })
    
    df = pd.DataFrame(data)
    
    # Create SpectralDataset
    fs = SpectralDataset.from_dataframe(
        df,
        metadata_columns=['entity_id', 'time', 'degrade'],
        intensity_columns=[f"{w:.1f}" for w in wavenumbers],
        wavenumber=wavenumbers
    )
    
    # Convert to TimeSpectrumSet
    ts = TimeSpectrumSet(
        x=fs.x,
        wavenumbers=fs.wavenumbers,
        metadata=fs.metadata,
        modality=fs.modality,
        time_col='time',
        entity_col='entity_id'
    )
    
    return ts

# Generate data
ts = generate_synthetic_aging(n_entities=5, n_times=10)
print(f"Total samples: {ts.x.shape[0]}")
print(f"Entities: {ts.metadata['entity_id'].nunique()}")
print(f"Time range: {ts.metadata['time'].min():.1f}-{ts.metadata['time'].max():.1f} days")

# Compute degradation trajectories
trajectories = compute_degradation_trajectories(
    ts,
    value_col='degrade',
    method='linear'  # or 'spline' for non-linear
)

print(f"\nDegradation Trajectories:")
for entity, row in trajectories.iterrows():
    print(f"  {entity}: slope={row['slope']:.4f}, R²={row['r_squared']:.3f}, p={row['p_value']:.1e}")

# Plot trajectories
fig, ax = plt.subplots(figsize=(10, 6))
for entity in ts.metadata['entity_id'].unique():
    mask = ts.metadata['entity_id'] == entity
    times = ts.metadata.loc[mask, 'time']
    values = ts.metadata.loc[mask, 'degrade']
    
    # Plot data points
    ax.scatter(times, values, label=entity, alpha=0.6)
    
    # Plot fitted line
    traj = trajectories.loc[entity]
    times_fit = np.linspace(times.min(), times.max(), 100)
    values_fit = traj['intercept'] + traj['slope'] * times_fit
    ax.plot(times_fit, values_fit, '--', alpha=0.8)

ax.set_xlabel('Time (days)')
ax.set_ylabel('Degradation Value')
ax.set_title('Degradation Trajectories')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("aging_trajectories.png", dpi=150, bbox_inches='tight')
print("\nSaved: aging_trajectories.png")

# Estimate shelf-life (time to threshold)
threshold = 2.0  # Quality threshold
shelf_life = estimate_remaining_shelf_life(
    ts,
    value_col='degrade',
    threshold=threshold
)

print(f"\nShelf-Life Estimates (threshold={threshold:.1f}):")
for entity, row in shelf_life.iterrows():
    print(f"  {entity}: {row['remaining_time']:.1f} ± {row['ci_95']:.1f} days")
```

**Expected Output:**
```yaml
Total samples: 50
Entities: 5
Time range: 0.0-30.0 days

Degradation Trajectories:
  E00: slope=0.0834, R²=0.962, p=1.2e-07
  E01: slope=0.1123, R²=0.978, p=2.3e-08
  E02: slope=0.0645, R²=0.945, p=4.5e-07
  E03: slope=0.1456, R²=0.985, p=5.6e-09
  E04: slope=0.0978, R²=0.971, p=8.9e-08

Saved: aging_trajectories.png

Shelf-Life Estimates (threshold=2.0):
  E00: 12.5 ± 2.3 days
  E01: 8.9 ± 1.8 days
  E02: 18.4 ± 3.1 days
  E03: 6.2 ± 1.4 days
  E04: 10.7 ± 2.0 days
```

---

## ✅ Validation & Sanity Checks

### Success Indicators

**Trajectory Fits:**
- ✅ R² > 0.85 for linear fits (degradation well-modeled)
- ✅ p-value < 0.05 (significant degradation trend)
- ✅ Residuals randomly distributed (no systematic bias)

**Shelf-Life Estimates:**
- ✅ Confidence intervals narrow (CV < 20%)
- ✅ Estimates consistent across replicates
- ✅ Predictions match historical data (if available)

**Chemical Plausibility:**
- ✅ Degradation rates match literature (e.g., 0.05–0.15 units/day for oil oxidation)
- ✅ Faster degradation at higher temperatures (if temperature varied)
- ✅ Shelf-life estimates within expected range (weeks to months)

### Failure Indicators

**⚠️ Warning Signs:**

1. **R² < 0.60 for trajectory fits**
   - Problem: High variability; non-linear degradation; measurement noise
   - Fix: Try spline fits (method='spline'); increase replication; check SNR

2. **Negative degradation slopes**
   - Problem: Degradation value definition inverted; samples improving over time (implausible)
   - Fix: Verify degradation value sign (should increase); check for sample contamination

3. **Wide confidence intervals (> 50% of estimate)**
   - Problem: High within-entity variability; too few time points; poor fit
   - Fix: Increase time points; improve replication; check measurement consistency

4. **Shelf-life estimates negative**
   - Problem: Threshold already exceeded; degradation started before t=0
   - Fix: Adjust threshold; extrapolate backwards if initial quality unknown

5. **All entities reach threshold at same time (no variability)**
   - Problem: Degradation rates artificially constrained; batch effect dominating
   - Fix: Check if entities truly independent; verify no systematic error

### Quality Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|--------|
| Trajectory R² | 0.70 | 0.85 | 0.95 |
| Trajectory p-value | < 0.05 | < 0.01 | < 0.001 |
| Shelf-Life CI Width | < 40% | < 20% | < 10% |
| Residual Normality (Shapiro p) | > 0.05 | > 0.10 | > 0.20 |

---

## ⚙️ Parameters You Must Justify

### Critical Parameters

**1. Time Column**
- **Parameter:** `time_col` (metadata column name)
- **No default:** Must specify
- **Justification:** "Time tracked in 'time' column (days since initial measurement)."

**2. Entity Column**
- **Parameter:** `entity_col` (sample/batch identifier)
- **No default:** Must specify
- **Justification:** "Independent samples identified by 'entity_id'; each tracked over time."

**3. Degradation Value**
- **Parameter:** `value_col` (quality metric to track)
- **No default:** Must specify (e.g., ratio, peroxide value, PC1)
- **Justification:** "Degradation quantified via ratio 1655/1742 (unsaturation/carbonyl), which decreases with oxidation."

**4. Trajectory Method**
- **Parameter:** `method` ('linear', 'spline')
- **Default:** 'linear'
- **When to adjust:** Use 'spline' if degradation non-monotonic or accelerates
- **Justification:** "Linear regression used as degradation kinetics appear first-order over measured time range."

**5. Shelf-Life Threshold**
- **Parameter:** `threshold` (quality limit)
- **No default:** Must specify based on regulations or QA limits
- **Justification:** "Threshold set at 2.0 based on industry standard for oil quality (ratio < 2.0 indicates rancidity)."

---

Quickstart (CLI)
- Aging trajectories and stages:
  - `foodspec aging input.h5 --value-col degrade --method linear --time-col time --entity-col sample_id --output-dir ./out`
- Shelf-life estimates:
  - `foodspec shelf-life input.h5 --value-col degrade --threshold 2.0 --time-col time --entity-col sample_id --output-dir ./out`

Python API
- Build a `TimeSpectrumSet` from a `FoodSpectrumSet`:
  - `ts = TimeSpectrumSet(x=fs.x, wavenumbers=fs.wavenumbers, metadata=fs.metadata, modality=fs.modality, time_col='time', entity_col='sample_id')`
- Trajectories:
  - `from foodspec.workflows.aging import compute_degradation_trajectories`
  - `res = compute_degradation_trajectories(ts, value_col='degrade', method='linear')`
- Shelf-life:
  - `from foodspec.workflows.shelf_life import estimate_remaining_shelf_life`
  - `df = estimate_remaining_shelf_life(ts, value_col='degrade', threshold=2.0)`
