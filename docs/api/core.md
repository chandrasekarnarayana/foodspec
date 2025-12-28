# Core API Reference

!!! info "Module Purpose"
    Core data structures and high-level workflows for spectral analysis. This is the foundation of all FoodSpec operations.

---

## Quick Navigation

| Component | Purpose | Common Use |
|-----------|---------|------------|
| [`FoodSpectrumSet`](#foodspectrumset) | Primary data container | Load, store, manipulate spectra |
| [`FoodSpec`](#foodspec-high-level-api) | High-level workflow API | Quick preprocessing and analysis |
| [`HyperSpectralCube`](#hyperspectralcube) | HSI data structure | Spatial spectral mapping |
| [`MultiModalDataset`](#multimodaldataset) | Multi-modality fusion | Raman + FTIR combinations |
| [`OutputBundle`](#outputbundle) | Result packaging | Save plots, tables, metadata |
| [`RunRecord`](#runrecord) | Provenance tracking | Reproducibility and audit trails |

---

## Common Patterns

### Pattern 1: Load → Preprocess → Analyze

```python
from foodspec import FoodSpectrumSet
from foodspec.io import load_folder
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import train_classifier
import numpy as np

# Load data
fs = load_folder(
    "data/oils/",
    pattern="*.txt",
    modality="raman",
    metadata_csv="data/oils/metadata.csv"
)
print(f"Loaded: {len(fs)} spectra, {len(fs.wavenumbers)} wavenumbers")

# Preprocess
fs_clean = baseline_als(fs, lam=1e4, p=0.01)
fs_clean = normalize_snv(fs_clean)

# Analyze
model, metrics = train_classifier(
    fs_clean,
    label_column="oil_type",
    method="random_forest"
)
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

### Pattern 2: Subsetting and Filtering

```python
# Filter by metadata
olive_only = fs[fs.metadata['oil_type'] == 'Olive']
print(f"Olive samples: {len(olive_only)}")

# Spectral cropping
from foodspec.preprocess import crop_spectrum
fs_cropped = crop_spectrum(fs, wn_min=600, wn_max=1800)

# Train/test split
train_fs, test_fs = fs.train_test_split(
    test_size=0.2,
    stratify_by="oil_type",
    random_state=42
)
```

### Pattern 3: Batch Processing with Provenance

```python
from foodspec.core import RunRecord, OutputBundle
from pathlib import Path
import datetime

# Create run record
run = RunRecord(
    run_id=f"oil_auth_{datetime.datetime.now():%Y%m%d_%H%M%S}",
    dataset_hash=fs.hash(),
    parameters={
        "baseline": "ALS",
        "lam": 1e4,
        "normalize": "SNV",
        "classifier": "RF"
    }
)

# Run analysis with outputs
output = OutputBundle(output_dir=f"runs/{run.run_id}")
output.save_figure(fig_confusion, "confusion_matrix.png")
output.save_table(metrics_df, "metrics.csv")
output.save_metadata(run.to_dict(), "run_record.json")
print(f"Results saved to: {output.output_dir}")
```

---

## FoodSpectrumSet

Primary data container for spectral data (Raman, FTIR, NIR).

::: foodspec.core.dataset.FoodSpectrumSet
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - __len__
        - __getitem__
        - validate
        - copy
        - train_test_split
        - hash
        - labels
        - groups
        - batch_ids

### Key Methods

#### `train_test_split()`

```python
# Stratified split preserving class balance
train_fs, test_fs = fs.train_test_split(
    test_size=0.25,
    stratify_by="oil_type",
    random_state=42
)
print(f"Train: {len(train_fs)}, Test: {len(test_fs)}")
print(f"Train classes: {train_fs.metadata['oil_type'].value_counts()}")
```

#### `hash()`

```python
# Dataset fingerprint for reproducibility
dataset_hash = fs.hash()
print(f"Dataset hash: {dataset_hash}")

# Verify data hasn't changed
fs_reloaded = load_folder("data/oils/", pattern="*.txt")
assert fs.hash() == fs_reloaded.hash(), "Dataset changed!"
```

#### Indexing and Slicing

```python
# Single spectrum
spectrum_0 = fs[0]
print(f"Shape: {spectrum_0.x.shape}")  # (1, n_wavenumbers)

# Slice
first_10 = fs[:10]
print(f"Subset: {len(first_10)} spectra")

# Boolean mask
high_quality = fs[fs.metadata['snr'] > 20]
print(f"High SNR samples: {len(high_quality)}")
```

**See Also:** [`load_folder()`](io.md#load_folder), [`baseline_als()`](preprocessing.md#baseline-correction)

---

## FoodSpec High-Level API

Convenience wrapper for quick analysis workflows.

::: foodspec.core.api.FoodSpec
    options:
      show_source: false
      heading_level: 3

### Example: Interactive Analysis

```python
from foodspec.core import FoodSpec

# Create high-level interface
fs_api = FoodSpec(fs)

# Chain preprocessing
fs_api.preprocess(
    baseline='rubberband',
    normalize='vector',
    smooth_window=21
)

# PCA visualization
fs_api.pca(n_components=3)
fs_api.plot_pca_scores(color_by='oil_type', save_path='pca_scores.png')

# Classification
results = fs_api.classify(
    label_column='oil_type',
    method='svm',
    cv_folds=5
)
print(f"CV Accuracy: {results['cv_accuracy']:.1%}")
```

**See Also:** [`preprocess()`](preprocessing.md), [`run_pca()`](chemometrics.md#pca-analysis)

---

## HyperSpectralCube

Hyperspectral imaging data structure for spatial analysis.

::: foodspec.core.hyperspectral.HyperSpectralCube
    options:
      show_source: false
      heading_level: 3

### Example: Spatial Mapping

```python
from foodspec.core.hyperspectral import HyperSpectralCube
from foodspec.viz.hyperspectral import plot_intensity_map

# Reconstruct cube from flattened spectra
cube = HyperSpectralCube.from_spectrum_set(
    fs_pixels,
    image_shape=(30, 30)
)
print(f"Cube shape: {cube.data.shape}")  # (height, width, n_wavenumbers)

# Extract intensity map at specific wavenumber
intensity_map = cube.get_intensity_map(target_wavenumber=1655, window=5)
plot_intensity_map(intensity_map, title="C=C Stretch (1655 cm⁻¹)")

# Calculate ratio map
ratio_map = cube.get_ratio_map(
    numerator_wn=1655,
    denominator_wn=1742,
    window=5
)
plot_intensity_map(ratio_map, title="Unsaturation/Carbonyl Ratio")
```

**See Also:** [Hyperspectral Mapping Workflow](../workflows/spatial/hyperspectral_mapping.md)

---

## MultiModalDataset

Combined spectral modalities (e.g., Raman + FTIR fusion).

::: foodspec.core.multimodal.MultiModalDataset
    options:
      show_source: false
      heading_level: 3

### Example: Multi-Modal Fusion

```python
from foodspec.core.multimodal import MultiModalDataset
from foodspec.ml.fusion import late_fusion_concat

# Combine Raman and FTIR datasets
fs_raman = load_folder("data/raman/", modality="raman")
fs_ftir = load_folder("data/ftir/", modality="ftir")

mm_dataset = MultiModalDataset(
    modalities={'raman': fs_raman, 'ftir': fs_ftir},
    align_by='sample_id'  # Match samples across modalities
)
print(f"Aligned samples: {len(mm_dataset)}")

# Feature-level fusion
X_fused = late_fusion_concat(
    mm_dataset.modalities['raman'].x,
    mm_dataset.modalities['ftir'].x
)
print(f"Fused feature shape: {X_fused.shape}")

# Train on fused features
model, metrics = train_classifier(
    X_fused,
    y=mm_dataset.modalities['raman'].labels,
    method='random_forest'
)
print(f"Fusion accuracy: {metrics['accuracy']:.1%}")
```

**See Also:** [`late_fusion_concat()`](ml.md#late_fusion_concat), [Multimodal Workflows](../05-advanced-topics/multimodal_workflows.md)

---

## OutputBundle

Reproducible output management for analysis results.

::: foodspec.core.output_bundle.OutputBundle
    options:
      show_source: false
      heading_level: 3

### Example: Structured Output

```python
from foodspec.core import OutputBundle
import matplotlib.pyplot as plt
import pandas as pd

# Create output directory
output = OutputBundle(output_dir="runs/oil_auth_20250128")

# Save figures
fig, ax = plt.subplots()
ax.plot(fs.wavenumbers, fs.x[0])
output.save_figure(fig, "spectrum_example.png", dpi=300)

# Save tables
metrics_df = pd.DataFrame({
    'metric': ['accuracy', 'f1_macro'],
    'value': [0.92, 0.89]
})
output.save_table(metrics_df, "metrics.csv")

# Save metadata
output.save_metadata(
    {'run_id': 'oil_auth_20250128', 'model': 'RandomForest'},
    "run_config.json"
)

print(f"Outputs saved to: {output.output_dir}")
print(f"Files: {list(output.output_dir.glob('*'))}")
```

**See Also:** [`RunRecord`](#runrecord), [Workflow Design & Reporting](../workflows/workflow_design_and_reporting.md)

---

## RunRecord

Experiment tracking and reproducibility.

::: foodspec.core.run_record.RunRecord
    options:
      show_source: false
      heading_level: 3

### Example: Provenance Tracking

```python
from foodspec.core import RunRecord
import datetime
import hashlib

# Create run record
run = RunRecord(
    run_id=f"exp_{datetime.datetime.now():%Y%m%d_%H%M%S}",
    dataset_hash=fs.hash(),
    parameters={
        'baseline_method': 'ALS',
        'baseline_lam': 1e4,
        'normalize': 'SNV',
        'classifier': 'RandomForest',
        'n_estimators': 100,
        'cv_folds': 5
    },
    metrics={
        'accuracy': 0.923,
        'f1_macro': 0.915,
        'balanced_accuracy': 0.920
    },
    timestamp=datetime.datetime.now()
)

# Save run record
run.save("runs/exp_20250128_143022/run_record.json")

# Load and verify
run_loaded = RunRecord.load("runs/exp_20250128_143022/run_record.json")
assert run_loaded.run_id == run.run_id
print(f"Run ID: {run_loaded.run_id}")
print(f"Parameters: {run_loaded.parameters}")
print(f"Metrics: {run_loaded.metrics}")
```

**See Also:** [`OutputBundle`](#outputbundle), [Model Lifecycle](../05-advanced-topics/model_lifecycle.md)

---

## Cross-References

**Related Modules:**
- [IO & Data Loading](io.md) - Load data into `FoodSpectrumSet`
- [Preprocessing](preprocessing.md) - Transform `FoodSpectrumSet` data
- [ML & Modeling](ml.md) - Train models on `FoodSpectrumSet`
- [Chemometrics](chemometrics.md) - PCA, PLS-DA on `FoodSpectrumSet`

**Related Workflows:**
- [Oil Authentication](../workflows/authentication/oil_authentication.md)
- [Hyperspectral Mapping](../workflows/spatial/hyperspectral_mapping.md)
- [Multimodal Fusion](../05-advanced-topics/multimodal_workflows.md)

