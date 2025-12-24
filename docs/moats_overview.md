# FoodSpec Moats: Differentiating Capabilities

This document describes FoodSpec's differentiating "moat" capabilities that provide unique value in food spectroscopy:

1. **Matrix Correction** — Handle matrix effects (chips vs. pure oil)
2. **Heating/Oxidation Trajectory Analysis** — Time-series degradation modeling
3. **Calibration Transfer Toolkit** — Transfer models between instruments
4. **Data Governance & Dataset Intelligence** — Prevent silent dataset failures (imbalance, leakage) and gate ML readiness

---

#### Drift Detection & Adaptation

```python
from foodspec.calibration_transfer import detect_drift, adapt_calibration_incremental
```

---

## 4. Data Governance & Dataset Intelligence

**Problem:** Most ML failures in food science are dataset failures (imbalance, leakage, batch confounding), not algorithm failures.

**Solution:** Tools to summarize dataset health, diagnose class balance, assess replicate consistency, detect leakage, and compute a readiness score (0–100) for deployment gatekeeping.

### Features

- Dataset Summary — class distribution, SNR, NaN/inf, negative intensity rate, metadata completeness
- Class Balance — imbalance ratio, undersized classes, recommendations
- Replicate Consistency — CV (%) per replicate group; flags high technical variability
- Leakage Detection — batch–label correlation (Cramér's V), replicate leakage risk/detection
- Readiness Score (0–100) — weighted composite across size, balance, replicates, metadata, spectral quality, leakage

### Key Assumptions

⚠️ Users must be aware:
- Labels/batches/replicates defined in metadata; replicates should not be split across train/test
- Severe batch–label correlation indicates confounding; use batch-aware CV or correction
- Thresholds: min 20 samples/class, imbalance ≤10:1, technical CV ≤10%

### Usage (Python API)

```python
from foodspec import FoodSpec

fs = FoodSpec("data.csv", modality="raman")

summary = fs.summarize_dataset(label_column="oil_type")
balance = fs.check_class_balance(label_column="oil_type")
consistency = fs.assess_replicate_consistency(replicate_column="sample_id")
leakage = fs.detect_leakage(label_column="oil_type", batch_column="batch", replicate_column="sample_id")
readiness = fs.compute_readiness_score(label_column="oil_type", batch_column="batch", replicate_column="sample_id")
```

### Outputs

Saved to `OutputBundle`:
- `dataset_summary`, `class_balance`, `replicate_consistency`, `leakage_detection`, `readiness_score`

See detailed guide in [docs/data_governance.md](docs/data_governance.md).
  base_dir: ./results
```

Run via CLI:
```bash
foodspec run-exp config.yml --artifact-path model.foodspec
```

### Outputs

All metrics/diagnostics saved to `OutputBundle` and `.foodspec` artifact:

- `matrix_correction_background_subtraction`: baseline metrics
- `matrix_correction_robust_scaling`: per-matrix scaling stats
- `matrix_correction_domain_adaptation_*`: alignment shift per target matrix
- `matrix_correction_matrix_effect_magnitude`: total correction magnitude + per-matrix breakdown

---

## 2. Heating/Oxidation Trajectory Analysis

**Problem:** Oil degradation and heating effects evolve over time; need to classify degradation stage, predict shelf life, and track key oxidation indices.

**Solution:** Time-series modeling on spectral indices with trajectory fitting, stage classification, and shelf-life estimation with confidence intervals.

### Features

- **Index Extraction**
  - Peroxide Index (PI): ratio ~840 / ~1080 cm⁻¹
  - Total Fatty Chain (TFC): intensity ~1440 cm⁻¹
  - OIT Proxy: ratio ~1660 / ~1440 cm⁻¹
  - C=C stretch, CH₂ bending

- **Trajectory Modeling**
  - Linear, exponential, sigmoidal fits
  - R², RMSE, trend direction reporting

- **Degradation Stage Classifier**
  - RandomForest classifier on index features
  - Cross-validated accuracy, feature importance
  - Confidence scores per sample

- **Shelf-Life Estimation**
  - Regression-based time-to-threshold
  - Confidence intervals (t-distribution)
  - Extrapolation warnings

### Key Assumptions

⚠️ **Users must be aware:**
- **Time column** exists in metadata and is **numeric** (hours, days, timestamps)
- Samples are **measured repeatedly over time** (longitudinal data)
- Degradation is **monotonic or follows known patterns** (linear/exponential/sigmoidal)
- **≥5 time points** per sample/group for reliable regression
- **No major batch effects** confounding time trends

### Usage

#### Python API

```python
from foodspec import FoodSpec

fs = FoodSpec("heating_study.csv", modality="raman")
results = fs.analyze_heating_trajectory(
    time_column="time_hours",
    indices=["pi", "tfc", "oit_proxy"],
    classify_stages=True,
    stage_column="degradation_stage",    # "fresh", "early", "advanced", "spoiled"
    estimate_shelf_life=True,
    shelf_life_threshold=2.0,            # PI threshold for spoilage
    shelf_life_index="pi"
)

print(results["shelf_life"]["shelf_life_estimate"])  # Time to threshold
print(results["shelf_life"]["confidence_interval"])  # (lower, upper)
```

#### exp.yml Configuration

```yaml
dataset:
  path: heating_data.csv
  modality: raman
  schema:
    time_column: time_hours
    stage_column: degradation_stage

heating_trajectory:
  time_column: time_hours
  indices: [pi, tfc, oit_proxy]
  classify_stages: true
  stage_column: degradation_stage
  estimate_shelf_life: true
  shelf_life_threshold: 2.0
  shelf_life_index: pi
```

### Outputs

Metrics saved to `OutputBundle`:

- `heating_trajectory`: trajectory fit metrics per index (R², RMSE, trend)
- `stage_classification`: CV accuracy, feature importance, stage distribution
- `shelf_life`: estimate, confidence interval, extrapolation warning, fit quality

---

## 3. Calibration Transfer Toolkit

**Problem:** Models trained on one instrument often fail when deployed to another due to instrument-specific biases (drift, temperature, optics).

**Solution:** Direct Standardization (DS) and Piecewise DS (PDS) v2 with robust regression, drift adaptation, and transfer success metrics dashboard.

### Features

- **Direct Standardization (DS)**
  - Global linear transformation: target → source domain
  - Ridge regularization for stability
  - Fast, effective for consistent instrument differences

- **Piecewise Direct Standardization (PDS) v2**
  - Local transformations per wavenumber window
  - Robust to localized instrument variations
  - Automated window selection

- **Drift Adaptation Pipeline**
  - Detect drift: mean shift + variance ratio
  - Incremental reference updates (exponential weighting)
  - Trigger recalibration when drift exceeds threshold

- **Transfer Success Metrics Dashboard**
  - Pre/post RMSE, R², MAE
  - Improvement ratios
  - Leverage/outlier counts
  - Residual statistics

### Key Assumptions

⚠️ **Users must be aware:**
- **Source (reference) and target (slave) instruments** measure **the same samples** (paired standards)
- **Standard samples span the calibration range**
- **Spectral alignment** (wavelength registration) done **separately before transfer**
- **Drift is gradual** and can be modeled incrementally
- **Transfer samples are representative** of production variability

### Usage

#### Python API

```python
from foodspec import FoodSpec
import numpy as np

# Load source reference spectra and target standards
source_standards = np.load("source_std.npy")  # (n_standards, n_wavenumbers)
target_standards = np.load("target_std.npy")  # Same samples on target instrument

fs = FoodSpec("target_production.csv", modality="raman")
fs.apply_calibration_transfer(
    source_standards=source_standards,
    target_standards=target_standards,
    method="pds",              # Piecewise DS
    pds_window_size=11,        # Local window size
    alpha=1.0                   # Ridge regularization
)
# Target spectra now aligned to source domain
```

#### Drift Detection & Adaptation

```python
from foodspec.calibration_transfer import detect_drift, adapt_calibration_incremental

# Check if new batch has drifted from reference
drift_detected, drift_metrics = detect_drift(
    X_reference=reference_spectra,
    X_current=new_batch_spectra,
    threshold=0.1
)

if drift_detected:
    # Update reference with new standards
    X_ref_updated, update_metrics = adapt_calibration_incremental(
        X_reference=reference_spectra,
        X_new_standards=new_standards,
        weight_decay=0.9  # Exponential weighting
    )
```

#### exp.yml Configuration

```yaml
dataset:
  path: target_production.csv
  modality: raman

calibration_transfer:
  method: pds
  pds_window_size: 11
  alpha: 1.0
  source_standards: source_std.csv
  target_standards: target_std.csv
```

### Outputs

Metrics saved to `OutputBundle`:

- `calibration_transfer_transfer`: reconstruction RMSE, transformation condition number, n_standards
- `calibration_transfer_success_dashboard` (if validation provided):
  - Pre/post RMSE, R², MAE
  - Improvement ratios
  - Leverage/outlier counts
  - Residual mean/std

---

## Integration with FoodSpec Workflow

All moats are:
- **Chainable** via `FoodSpec` API
- **Configurable** via `exp.yml`
- **Reproducible** via `OutputBundle` + `.foodspec` artifacts
- **CLI-friendly** via `foodspec run-exp`

### Full Workflow Example

```python
from foodspec import FoodSpec

# Load data
fs = FoodSpec("chips_heating_study.csv", modality="raman")

# Step 1: Matrix correction
fs.apply_matrix_correction(
    method="adaptive_baseline",
    scaling="median_mad",
    domain_adapt=True,
    matrix_column="matrix_type"
)

# Step 2: Preprocess
fs.preprocess(preset="standard")

# Step 3: Analyze heating trajectory
trajectory = fs.analyze_heating_trajectory(
    time_column="time_hours",
    indices=["pi", "tfc"],
    estimate_shelf_life=True,
    shelf_life_threshold=2.0
)

# Step 4: Export artifacts
fs.export(path="./results/", formats=["json", "csv", "joblib"])

print(f"Shelf life: {trajectory['shelf_life']['shelf_life_estimate']} hours")
print(f"Matrix correction magnitude: {fs.bundle.metrics['matrix_correction_matrix_effect_magnitude']['total_correction_magnitude']}")
```

---

## Performance Considerations

- **Matrix Correction**: O(n_samples × n_wavenumbers) for baseline + scaling; O(n_samples² × n_components) for domain adaptation
- **Heating Trajectory**: O(n_samples × n_wavenumbers) for index extraction; O(n_samples log n_samples) for trajectory fitting
- **Calibration Transfer**: O(n_standards × n_wavenumbers²) for DS; O(n_standards × n_wavenumbers × window_size) for PDS

Typical runtimes:
- 1000 samples × 1000 wavenumbers: <5 seconds for matrix correction + trajectory analysis
- DS/PDS transfer with 50 standards: <2 seconds

---

## Validation & Testing

All moats have:
- Unit tests for core algorithms
- Integration tests via `run-exp` CLI
- Synthetic data fixtures for edge cases
- Real-world validation on benchmark datasets

See `tests/test_matrix_correction.py`, `tests/test_heating_trajectory.py`, `tests/test_calibration_transfer.py` for coverage.

---

## References

**Matrix Correction:**
- Eilers & Boelens (2005), "Baseline Correction with Asymmetric Least Squares Smoothing"
- Fernando et al. (2013), "Unsupervised Visual Domain Adaptation Using Subspace Alignment"

**Heating Trajectory:**
- Guillén-Casla et al. (2011), "Monitoring oil degradation by Raman spectroscopy"
- ASTM D6186: "Standard Test Method for Oxidation Induction Time of Lubricating Oils by Pressure Differential Scanning Calorimetry"

**Calibration Transfer:**
- Wang et al. (1991), "Multivariate Instrument Standardization"
- Bouveresse et al. (1996), "Standardization of NIR spectra in diffuse reflectance mode"

---

## Contributing

To extend the moats or add new ones, follow the pattern:

1. Create module in `src/foodspec/<moat_name>.py` with:
   - Comprehensive module docstring (assumptions, usage)
   - Function-level docstrings (inputs/outputs/logic)
   - Type hints and parameter validation
   - Metric dictionaries for `OutputBundle`

2. Add method to `FoodSpec` class in `core/api.py`:
   - Chainable (return `self`)
   - Record metrics to `bundle`
   - Add step to `run_record`

3. Export from `__init__.py`

4. Add tests in `tests/test_<moat_name>.py`

5. Document in this file + `docs/quickstart_python.md`

---

**For questions or support:** Open an issue on GitHub with moat-specific tag.
