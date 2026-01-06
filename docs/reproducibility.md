# Reproducibility Guide for FoodSpec

**Purpose:** Best practices for capturing provenance, enforcing data integrity, and making analyses reproducible and auditable.

**Audience:** Food scientists, researchers, QA/validation teams.

**Time:** 30–45 minutes to read; implement in your workflow incrementally.

**Prerequisites:** FoodSpec installed; familiarity with protocols, CSV/HDF5 data, and basic Python.

---

## Statement of Need

Food spectroscopy analyses are often difficult to reproduce because:

1. **Preprocessing parameters** are scattered across notebooks, vendor software, and manual scripts
2. **Data leakage** (preprocessing before splitting, batch effects ignored) is silent—hard to detect in review
3. **Metadata** (versions, parameters, environmental conditions) are not consistently captured

This guide shows how to use FoodSpec's protocol system, provenance logging, and validation framework to make your analyses reproducible, auditable, and defensible.

---

## Working Rules (Short Version)

1. **Determinism first.** Set seeds everywhere. Avoid nondeterministic algorithms unless you document them and run enough trials.
2. **Preprocess inside validation.** Baseline/normalize/smooth within each fold to avoid leakage. Never fit preprocessing on the full dataset if you report CV performance.
3. **Batch-aware splits.** Keep replicates and batches together. Stratify on label + batch to avoid optimistic estimates.
4. **Document rationale, not just choices.** Defaults are not self-justifying. Note why you chose ALS vs. rubberband, or vector vs. SNV.
5. **Save artifacts, not just metrics.** Models, pipelines, configs, figures, hashes, and environment info belong together.
6. **Prefer protocols over prose.** YAML > loose text. Version control your protocol files.
7. **Make reruns cheap.** One command (CLI or Python) should recreate the full run.

---

## Minimum Metadata to Capture

Keep these four blocks with every run (JSON/YAML is fine):

**1) Dataset**
- Source path/URI, format, hash (SHA-256)
- Instrument, laser/wavelength, resolution, acquisition dates
- Sample IDs, labels, batch IDs, replicate groups

**2) Preprocessing**
- Baseline method + parameters (e.g., ALS λ, p)
- Normalization method
- Smoothing/filters and parameters
- Any corrections (ATR, cosmic ray removal)

**3) Modeling & Validation**
- Features used (peaks/ratios/PCA components)
- Algorithm + hyperparameters
- Split strategy (train/test, CV, nested CV) and seeds
- Metrics with uncertainty (mean ± std or CI)

**4) Execution Environment**
- FoodSpec version, Python version
- Dependency versions (numpy, scipy, sklearn, etc.)
- Git commit hash (if available)
- Timestamps and user/host (optional but helpful)

If you capture nothing else, capture these four blocks.

---

## Pipeline Recipe (Do This Every Time)

### 1. Make it deterministic

```python
import numpy as np

SEED = 42
np.random.seed(SEED)

# Example: scikit-learn
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Example: Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1)
```

Set seeds for numpy, CV splitters, models, and any stochastic preprocessing.

### 2. Preprocess inside validation

```python
from foodspec.preprocessing import BaselineCorrector, Normalizer, Smoother, Pipeline

preproc = Pipeline([
    ('baseline', BaselineCorrector(method='als', lambda_=1e5, p=0.01)),
    ('normalize', Normalizer(method='vector')),
    ('smooth', Smoother(method='savitzky_golay', window=9, poly_order=3))
])

# In cross-validation: fit_transform on train fold, transform on test fold
```

Do **not** fit preprocessing on the full dataset if you are estimating generalization performance.

### 3. Use batch-aware splitting

```python
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

labels = np.array(ds.metadata['labels'])
batches = np.array(ds.metadata['batch'])
strat_key = np.array([f"{l}_{b}" for l, b in zip(labels, batches)])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(split.split(labels, strat_key))
```

Keep replicates and batches intact. For CV, use combined label+batch keys.

### 4. Record the run

Save these in a run folder (timestamped):

- `preprocessing_pipeline.pkl`
- `model.pkl`
- `features.csv` (or `.npy`)
- `predictions.csv`
- `metrics.json` (include std/CI if CV)
- `metadata.json` (the four blocks above)
- `protocol.yaml` (exact steps)
- `figures/` (PNG/PDF)
- `reproducibility_log.txt` (plain text summary)

FoodSpec helpers (examples):

```python
from foodspec.reporting import MethodsTextGenerator
methods_gen = MethodsTextGenerator(
    title="Oil authentication (Raman)",
    preprocessing_config={...},
    model_name="Random Forest",
    n_samples=len(ds),
    cv_folds=5
)
methods_text = methods_gen.generate()
open(f"{run_dir}/methods.txt", "w").write(methods_text)
```

---

## FAIR in Practice (Short, Applied)

- **Findable:** Name runs with IDs and include hashes. Example: `FSP-OILAUTH-20240106-120301_sha256abcd/`.
- **Accessible:** Use open formats (CSV, HDF5, JSON, YAML). Note license and access level in metadata.
- **Interoperable:** Stick to standard columns: `wavenumber`, `intensities`, `sample_id`, `label`, `batch`. Provide schema notes in metadata.
- **Reusable:** Keep parameters, versions, and rationale together. Export a methods text and a protocol file with every run.

You do not need lengthy FAIR essays—focus on the practical pieces above.

---

## Pitfalls and How to Avoid Them

| Pitfall | Why it hurts | Fix |
|---------|-------------|-----|
| Preprocessing before split | Leakage → inflated metrics | Fit preprocessing inside CV folds/train split only |
| Replicates split across folds | Over-optimistic scores | Group by replicate/batch; stratify on label+batch |
| Unseeded randomness | Irreproducible results | Set seeds for numpy, CV, models, any stochastic step |
| Default metrics only | Miss bias/imbalance | Report balanced accuracy, ROC-AUC, precision/recall, with CI |
| Missing parameter logs | Cannot rerun | Save preprocessing + model params and protocol |
| Hidden environment drift | Works only on author machine | Record package versions and Python version |
| No data hashes | Wrong file, silent drift | Store SHA-256 for inputs and key outputs |

---

## Validation Playbook

1) **Choose the split strategy**
- Small, balanced data: stratified k-fold
- Batches or days: batch-aware stratification
- Hyperparameter search: nested CV (inner tune, outer evaluate)

2) **Metrics to report**
- Classification: balanced accuracy, ROC-AUC, precision, recall, F1; add calibration if probabilities matter
- Regression: RMSE/MAE + confidence intervals; include plots of residuals vs. wavenumber or batches

3) **Uncertainty**
- Use CV means ± std or bootstrapped CI
- Report sample counts per class in each fold (or min/max across folds)

4) **Leakage checks**
- Ensure preprocessing is inside the CV pipeline
- Keep related samples together (replicates, batches, time series)

---

## Lightweight Protocol Template

Save as `protocols/<name>.yaml` and version-control it.

```yaml
name: "Oil authentication: VOO vs ROO"
version: "1.0"
created: "2026-01-06"
seed: 42

data:
  path: "data/oils_raw.csv"
  wavenumber_col: "wavenumber"
  label_col: "oil_type"
  batch_col: "batch"

preprocessing:
  - baseline: {method: als, lambda: 1e5, p: 0.01}
  - normalize: {method: vector}
  - smooth: {method: savitzky_golay, window: 9, poly_order: 3}

features:
  type: peak_ratios
  regions:
    C=C_1650: [1630, 1670]
    CH2_1440: [1420, 1450]
    C_O_1275: [1260, 1290]
  ratios:
    - [C=C_1650, CH2_1440]
    - [C_O_1275, C=C_1650]

model:
  type: RandomForestClassifier
  params: {n_estimators: 200, max_depth: 8, random_state: 42}

validation:
  strategy: nested_cv
  outer_folds: 5
  inner_folds: 3
  stratify_on: [label, batch]

outputs:
  run_dir: "runs/oil_auth_20260106_120301"
  save: [model, preprocessing_pipeline, features, predictions, metrics, methods_text]
```

Run it via Python or CLI wrappers, but always commit the protocol.

---

## Run Artifact Layout (Example)

```
runs/oil_auth_20260106_120301/
  protocol.yaml
  preprocessing_pipeline.pkl
  model.pkl
  features.csv
  predictions.csv
  metrics.json
  metadata.json
  methods.txt
  figures/
    01_raw.png
    02_baseline.png
    03_importance.png
  reproducibility_log.txt
```

This bundle is what you share with collaborators or attach as supplementary material.

---

## Reproducibility Log (Fill-and-Run Template)

Save this as `reproducibility_log.txt` in each run folder and fill values programmatically.

```
FoodSpec Reproducibility Log
============================

Timestamp: <ISO8601>
Analyst: <name/email>
Project: <short title>

Data
-----
Input file: <path>
SHA-256: <hash>
Samples: <n>
Classes: <counts per class>
Batch/replicates: <description>

Preprocessing
-------------
Baseline: <method + params>
Normalization: <method>
Smoothing/filters: <method + params>
Other corrections: <if any>

Features
--------
Type: <peaks/ratios/PCA/etc.>
List: <names>

Model
-----
Algorithm: <name>
Hyperparameters: <dict>

Validation
----------
Strategy: <train/test, k-fold, nested>
Stratification: <label/batch/etc.>
Seed(s): <list>

Metrics
-------
Balanced accuracy: <mean ± std>
ROC-AUC (if used): <mean ± std>
Other metrics: <list>

Environment
-----------
FoodSpec: <version>
Python: <version>
Key deps: numpy=<>, scipy=<>, sklearn=<>, pandas=<>
Git commit: <hash or N/A>

Artifacts
---------
Model: model.pkl
Preprocessing: preprocessing_pipeline.pkl
Features: features.csv
Predictions: predictions.csv
Figures: figures/
Protocol: protocol.yaml

Notes
-----
<observations, anomalies, decisions>
```

---

## Quick Checklist (Use Before You Share Results)

- [ ] Seeds set for numpy, CV, and models
- [ ] Preprocessing fit inside CV/train folds (no leakage)
- [ ] Batch/replicate grouping enforced
- [ ] Metrics include uncertainty (CV std/CI)
- [ ] Data hash recorded
- [ ] Environment versions recorded
- [ ] Protocol file saved and versioned
- [ ] Artifacts (model, pipeline, features, predictions, figures) saved
- [ ] Reproducibility log written
- [ ] Rationale captured (why each method/parameter was chosen)

If all boxes are checked, someone else can rerun your analysis—now or two years from now.

---

## FAQ

**Q: Do I need nested cross-validation?**  
Use it when hyperparameters are tuned and data is limited. For quick checks or large data, a train/validation/test split with leakage controls may be enough—just state what you did.

**Q: How do I handle tiny datasets (n<40)?**  
Use leave-one-group-out with batch/replicate grouping. Report uncertainty generously. Avoid heavy hyperparameter searches.

**Q: What about hyperspectral cubes?**  
Keep per-pixel pipelines but aggregate at ROI/patch levels for validation. Record the spatial resolution, masks, and any downsampling.

**Q: How do I share data with restrictions?**  
Share hashes, schemas, and synthetic examples; describe access conditions. Keep the protocol and code fully open so others can rerun once they have data access.

---

## Closing Note

Reproducibility is easier when it is the default. FoodSpec’s guardrails (leakage checks, batch-aware splits, protocol-driven runs, and artifact bundling) exist because I kept tripping over the same issues. Use this guide as your minimum bar; adjust upward for regulatory or high-stakes studies.

*Last updated: January 2026*# Reproducibility and FAIR Data in FoodSpec

**Audience:** Researchers, data managers, reviewers  
**Scope:** Design principles, implementation guidance, compliance with FAIR principles  
**Goals:** Enable transparent, auditable, and reproducible food spectroscopy research

---

## Why Reproducibility Matters in Food Spectroscopy

The "reproducibility crisis" in science affects food authentication and quality control research:

- **Undocumented preprocessing:** Baseline correction parameters, normalization methods, and smoothing windows are often omitted from publications
- **Manual workflows:** Ad hoc scripts and vendor software create analysis chains that are difficult to replicate
- **Implicit assumptions:** Sample handling, instrument calibration, and validation strategies are rarely fully specified
- **Model opacity:** Trained classifiers are published without hyperparameters, training data descriptions, or uncertainty quantification

**Consequence:** Peer reviewers and replicators cannot verify published results, leading to wasted research effort and irreproducible claims in regulatory contexts.

FoodSpec addresses this through **deterministic pipelines**, **comprehensive metadata**, and **FAIR-aligned data practices**.

---

## Core Principles

### Principle 1: Deterministic Pipelines

A reproducible analysis produces *identical results* given the same input data and parameters.

**FoodSpec ensures determinism through:**

1. **Fixed random seeds:** All stochastic operations (train/test splitting, cross-validation, random forest initialization) use explicit random seeds
2. **Versioned algorithms:** Preprocessing, feature extraction, and modeling steps are reproducible across Python versions
3. **Parameter transparency:** All hyperparameters are logged and accessible via configuration files or function arguments

**Example:**
```python
from foodspec import SpectralDataset
from foodspec.preprocessing import BaselineCorrector

# Identical preprocessing with same parameters → identical results
corrector = BaselineCorrector(method='als', lambda_=1e5, p=0.01, random_state=42)
ds_corrected = corrector.fit_transform(ds)

# Same result every time (deterministic)
corrector2 = BaselineCorrector(method='als', lambda_=1e5, p=0.01, random_state=42)
ds_corrected2 = corrector2.fit_transform(ds)

assert (ds_corrected.intensities == ds_corrected2.intensities).all()
```

**Non-determinism risks:**
- ❌ Using numpy without seeding: `train_idx = np.random.choice(n_samples, size=n_train)`
- ❌ Relying on default random states: `RandomForestClassifier(n_estimators=100)`
- ✅ Seeding all randomness: `RandomForestClassifier(n_estimators=100, random_state=42)`

---

### Principle 2: Complete Metadata Capture

Reproducibility requires knowing *how* an analysis was performed, not just *that* it was performed.

**FoodSpec captures metadata at multiple levels:**

**Level 1: Dataset Metadata**
```python
ds.metadata = {
    'acquisition_date': '2024-01-15',
    'instrument': 'Bruker SENTERRA II',
    'laser_wavelength': '785 nm',
    'sample_count': 60,
    'batches': ['day1', 'day2'],
    'labels': ['VOO', 'ROO', ...],
    'temporal_order': [1, 2, 3, ...]  # Measurement sequence for drift detection
}
```

**Level 2: Preprocessing Metadata**
```python
preprocessing_config = {
    'baseline': {
        'method': 'als',
        'lambda': 1e5,
        'p': 0.01,
        'applied_date': '2024-01-16',
        'applied_by': 'user@institution.org'
    },
    'normalize': {
        'method': 'vector',
        'fitting_samples': 60  # All samples or subset?
    },
    'smooth': {
        'method': 'savitzky_golay',
        'window': 9,
        'poly_order': 3
    }
}
```

**Level 3: Model Training Metadata**
```python
model_metadata = {
    'model_type': 'RandomForestClassifier',
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42,
        'n_jobs': -1
    },
    'training_data': {
        'n_samples': 42,
        'class_distribution': {'VOO': 21, 'ROO': 21},
        'feature_count': 5,
        'feature_names': ['C=C_ratio', 'C_O_ratio', ...]
    },
    'validation_strategy': {
        'method': 'nested_cross_validation',
        'outer_folds': 5,
        'inner_folds': 3,
        'stratification': 'label_and_batch'
    }
}
```

**Level 4: Execution Metadata**
```python
execution_log = {
    'timestamp': '2024-01-16T14:32:15Z',
    'analyst': 'researcher@institution.org',
    'foodspec_version': '1.0.0',
    'python_version': '3.10.8',
    'dependencies': {
        'numpy': '1.24.1',
        'pandas': '2.0.0',
        'scikit-learn': '1.3.1'
    },
    'git_commit': 'a3f8b2c7e9d1f5a6',
    'execution_time_seconds': 23.5
}
```

---

### Principle 3: Protocol-Driven Analysis

Rather than ad hoc scripts, FoodSpec supports declarative analysis via YAML configuration files.

**Benefits:**
- Separation of logic (what to do) from implementation (how to do it)
- Human-readable documentation
- Validation against schema
- Version control-friendly

**Example protocol file:**
```yaml
# protocols/oil_authentication_v1.0.yaml
name: "Oil Authentication: VOO vs. ROO"
version: "1.0.0"
created: "2024-01-15"
doi: "10.5281/zenodo.1234567"

description: |
  Classification of virgin olive oil (VOO) from refined olive oil (ROO)
  using Raman spectroscopy and Random Forest classification.
  
  References:
    - Cepeda et al. (2019). Food Control 103:283-289
    - Galtier et al. (2007). Analytica Chimica Acta 581:227-234

input:
  data_source: "s3://institutional-repository/oils_dataset_v2.h5"
  data_sha256: "a3f8b2c7e9d1f5a6b9c0d3e4f5a6b7c8"
  format: "hdf5"
  metadata:
    acquisition_date: "2024-01-15"
    instrument: "Bruker SENTERRA II"
    n_samples: 60
    class_labels: ["VOO", "ROO"]

preprocessing:
  steps:
    - name: "baseline_correction"
      method: "als"
      parameters:
        lambda: 1e5
        p: 0.01
      rationale: "Remove instrument baseline drift without specifying manually"
    
    - name: "vector_normalization"
      method: "l2"
      rationale: "Account for sample thickness and probe-to-sample distance variation"
    
    - name: "smoothing"
      method: "savitzky_golay"
      parameters:
        window: 9
        poly_order: 3
      rationale: "Reduce high-frequency noise while preserving peak structure"

feature_extraction:
  method: "peak_ratios"
  regions:
    - name: "C=C_stretch"
      range: [1630, 1670]
      units: "wavenumber_cm_inv"
    - name: "CH2_bend"
      range: [1420, 1450]
    - name: "C_O_stretch"
      range: [1260, 1290]
  ratios:
    - ["C=C_stretch", "CH2_bend"]
    - ["C_O_stretch", "C=C_stretch"]

modeling:
  algorithm: "RandomForestClassifier"
  hyperparameters:
    n_estimators: 100
    max_depth: 5
    random_state: 42
    n_jobs: -1
  
  training:
    method: "stratified_split"
    test_size: 0.3
    stratify_by: ["label", "batch"]
    random_state: 42

validation:
  method: "nested_cross_validation"
  outer_cv:
    n_splits: 5
    strategy: "stratified_k_fold"
  inner_cv:
    n_splits: 3
    strategy: "stratified_k_fold"
  metrics:
    - "balanced_accuracy"
    - "roc_auc"
    - "precision"
    - "recall"
    - "f1_score"

reporting:
  figures:
    - name: "raw_spectra"
      type: "matplotlib"
      format: "png"
    - name: "pca_scores"
      type: "matplotlib"
    - name: "confusion_matrix"
      type: "seaborn"
  
  tables:
    - name: "feature_importance"
      format: "csv"
    - name: "cross_validation_metrics"
      format: "csv"
  
  methods_text: true

output:
  directory: "runs/oil_auth_20240116_143215"
  save_artifacts:
    - "model.pkl"
    - "preprocessor.pkl"
    - "feature_extractor.pkl"
    - "predictions.csv"
    - "metrics.json"
    - "figures/"
```

**Execute protocol:**
```python
from foodspec.protocols import load_and_execute

result = load_and_execute('protocols/oil_authentication_v1.0.yaml')

print(f"Balanced accuracy: {result.metrics['balanced_accuracy']:.3f}")
print(f"Figures saved to: {result.output_dir}/figures/")
print(f"Reproducibility log: {result.output_dir}/reproducibility_log.txt")
```

---

## FAIR Principles in FoodSpec

The FAIR principles (Findable, Accessible, Interoperable, Reusable) guide modern data management. FoodSpec implements FAIR practices:

### Findable: Unique Identifiers and Metadata

**What FoodSpec does:**
- Assign unique identifiers to datasets and runs
- Store rich metadata (Dublin Core elements)
- Generate descriptive file names

```python
# Auto-generated run ID
run_id = "FSP-OILAUTH-20240116-143215-a3f8b2c7"

# Dataset with persistent identifier
ds.metadata['doi'] = "10.5281/zenodo.1234567"
ds.metadata['uuid'] = "f47ac10b-58cc-4372-a567-0e02b2c3d479"

# Save with metadata-rich filename
filename = f"oils_raman_1200samples_VOO-ROO_20240115_{ds.metadata['uuid']}.h5"
ds.save_hdf5(filename)
```

### Accessible: Open Formats and Access Control

**What FoodSpec does:**
- Use open formats (HDF5, CSV, JSON, YAML)
- Support data export for external tools
- Document access permissions and licensing

```python
# Export to open formats
ds.to_csv('output/spectra.csv')  # Open, tool-independent
ds.save_hdf5('output/spectra.h5')  # Open HDF5 standard
ds.to_json('output/metadata.json')  # Structured metadata

# License declaration
ds.metadata['license'] = 'CC-BY-4.0'
ds.metadata['access_level'] = 'public'
ds.metadata['embargo_until'] = None
```

### Interoperable: Standard Data Models

**What FoodSpec does:**
- Use standard data structures (SpectralDataset, HyperspectralCube)
- Support cross-tool compatibility
- Document data schema

```python
# Standard structure: wavenumber + intensities + metadata
ds.wavenumber  # 1D array
ds.intensities  # 2D array (n_samples × n_wavenumbers)
ds.metadata  # Structured dict with standard keys

# Export to external tools
ds.to_netcdf('output/spectra.nc')  # Compatible with R, MATLAB, CDO
ds.to_feather('output/spectra.feather')  # Arrow format, language-agnostic
```

### Reusable: Comprehensive Documentation and Licensing

**What FoodSpec does:**
- Generate machine-readable methods
- Archive full analysis provenance
- Specify data usage licenses

```python
# Auto-generated methods for publication
methods_text = result.generate_methods_section()
print(methods_text)

# Reproducibility bundle
result.save_reproducibility_bundle(
    output_dir='runs/oil_auth_20240116/',
    include=[
        'preprocessing_config',
        'model_weights',
        'feature_importance',
        'cross_validation_scores',
        'execution_log',
        'methods_text'
    ]
)
```

---

## Implementation Guide

### Step 1: Configure Deterministic Analysis

```python
import numpy as np
import pandas as pd
from foodspec import SpectralDataset
from foodspec.preprocessing import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Set global random seed
np.random.seed(42)

# Load data
ds = SpectralDataset.from_csv('data/oils.csv', wavenumber_col='wavenumber')
ds.metadata['random_seed'] = 42

# Preprocessing with deterministic parameters
pipeline = Pipeline([
    ('baseline', {'method': 'als', 'lambda': 1e5, 'p': 0.01}),
    ('normalize', {'method': 'vector'}),
    ('smooth', {'method': 'savitzky_golay', 'window': 9, 'poly_order': 3})
])

ds_processed = pipeline.fit_transform(ds)

# Modeling with explicit random state
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,  # Critical for reproducibility
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Step 2: Capture Complete Metadata

```python
import json
from datetime import datetime
import hashlib

# Dataset metadata
dataset_metadata = {
    'acquisition': {
        'date': '2024-01-15',
        'instrument': 'Bruker SENTERRA II',
        'operator': 'researcher@institution.org',
        'conditions': {
            'temperature': 25.0,
            'humidity': 45.0
        }
    },
    'processing': {
        'preprocessing_pipeline': str(pipeline),
        'feature_extraction': 'peak_ratios',
        'feature_count': 5
    },
    'validation': {
        'cv_method': 'nested_stratified_k_fold',
        'outer_folds': 5,
        'inner_folds': 3
    },
    'environment': {
        'python_version': '3.10.8',
        'foodspec_version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
        'git_commit': 'a3f8b2c7e9d1f5a6'
    }
}

# Save metadata
with open('metadata.json', 'w') as f:
    json.dump(dataset_metadata, f, indent=2)

# Compute data checksums for integrity verification
with open('data/oils.csv', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
    dataset_metadata['data_integrity'] = {
        'filename': 'oils.csv',
        'sha256': file_hash
    }
```

### Step 3: Use Protocols for Reproducibility

```yaml
# Save as: protocols/analysis_v1.yaml
name: "Oil Authentication"
version: "1.0"
description: "VOO vs. ROO classification using Raman spectroscopy"

preprocessing:
  - baseline: {method: als, lambda: 1e5, p: 0.01}
  - normalize: {method: vector}
  - smooth: {method: savitzky_golay, window: 9, poly_order: 3}

model: RandomForestClassifier
hyperparameters: {n_estimators: 100, max_depth: 5, random_state: 42}

validation:
  method: nested_cross_validation
  outer_folds: 5
  inner_folds: 3
  stratify_by: label_and_batch
```

```python
from foodspec.protocols import Protocol

# Load and execute protocol
protocol = Protocol.from_yaml('protocols/analysis_v1.yaml')
result = protocol.execute(ds)

# Results are fully reproducible
print(f"Balanced accuracy: {result.metrics['balanced_accuracy']:.3f}")
print(f"Full provenance: {result.execution_log}")
```

### Step 4: Archive Artifacts for Reproducibility

```python
import os
import shutil

# Create timestamped output directory
from datetime import datetime
run_dir = f"runs/oil_auth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(run_dir, exist_ok=True)

# Save all artifacts
artifacts = {
    'model': clf,
    'preprocessor': pipeline,
    'dataset_metadata': dataset_metadata,
    'predictions': predictions_df,
    'metrics': metrics_dict,
    'protocol': 'protocols/analysis_v1.yaml'
}

# Serialize
import pickle
with open(f'{run_dir}/model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open(f'{run_dir}/preprocessor.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open(f'{run_dir}/metadata.json', 'w') as f:
    json.dump(dataset_metadata, f, indent=2)

predictions_df.to_csv(f'{run_dir}/predictions.csv', index=False)

# Copy protocol
shutil.copy('protocols/analysis_v1.yaml', f'{run_dir}/protocol.yaml')

print(f"Analysis artifacts saved to: {run_dir}")
```

---

## Reproducibility Checklist

Use this checklist before finalizing an analysis:

**Code and Data**
- [ ] All random seeds explicitly set (numpy, scikit-learn, etc.)
- [ ] Input data filename and hash documented
- [ ] Data preprocessing applied before train/test splitting
- [ ] No manual parameter tuning on test set

**Metadata**
- [ ] Dataset metadata captured (acquisition date, instrument, operator)
- [ ] Preprocessing configuration saved
- [ ] Model hyperparameters documented
- [ ] Execution environment recorded (Python version, package versions, timestamp)
- [ ] Code version (git commit hash) recorded

**Validation**
- [ ] Stratified cross-validation used
- [ ] Batch effects controlled for (if applicable)
- [ ] Nested CV for hyperparameter tuning
- [ ] Multiple evaluation metrics reported
- [ ] Uncertainty quantified (confidence intervals, standard errors)

**Artifacts**
- [ ] Trained model saved
- [ ] Preprocessing pipeline saved
- [ ] Predictions and metrics saved
- [ ] Reproducibility log generated
- [ ] Methods text auto-generated

**Documentation**
- [ ] Protocol YAML file created and version-controlled
- [ ] README with execution instructions
- [ ] Data dictionary or schema
- [ ] Licensing and access permissions declared
- [ ] DOI or persistent identifier assigned

**Publication**
- [ ] Supplementary methods text generated
- [ ] Data available (open repository, upon request, or embargoed with clear timeline)
- [ ] Code available (GitHub or institutional repository)
- [ ] Sufficient detail for independent replication

---

## Common Pitfalls to Avoid

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **No random seed** | Different results on re-run | Set `random_state=42` in all stochastic operations |
| **Preprocessing after splitting** | Data leakage; inflated performance | Apply preprocessing to *full* dataset before splitting |
| **Manual hyperparameter tuning** | Overfitting to test set | Use nested CV with internal hyperparameter search |
| **Undocumented parameters** | Cannot reproduce preprocessing | Save preprocessing config to JSON/YAML |
| **Inconsistent metric reporting** | Unclear performance | Report balanced accuracy, ROC-AUC, and confidence intervals |
| **Lost intermediate steps** | Cannot debug failures | Save preprocessed spectra, features, and predictions |
| **Version mismatch** | Code works on author's machine but not elsewhere | Pin dependency versions (requirements.txt or pyproject.toml) |
| **Implicit assumptions** | Reader cannot understand decision logic | Use protocols to explicitly state processing order |

---

## Integration with Research Workflow

### Version Control (Git)

```bash
# Repository structure
.
├── data/
│   └── oils_raw.csv
├── protocols/
│   ├── oil_authentication_v1.0.yaml
│   └── oil_authentication_v1.1.yaml
├── scripts/
│   └── analysis.py
├── runs/
│   ├── oil_auth_20240116_143215/
│   │   ├── model.pkl
│   │   ├── predictions.csv
│   │   ├── metadata.json
│   │   └── reproducibility_log.txt
│   └── oil_auth_20240117_092045/
├── .gitignore
└── README.md

# .gitignore
*.pkl
runs/
__pycache__/
*.pyc
.DS_Store

# Track protocols but not large artifacts
protocols/ ✓
data/ ✓ (if < 100 MB)
runs/ ✗ (use data repository instead)
```

### Data Repository (Zenodo, OSF, Institutional)

```python
# Upload processed data with metadata
import requests
import json

zenodo_metadata = {
    'title': 'Raman spectra of virgin and refined olive oils',
    'description': 'Raw and preprocessed Raman spectra for oil authentication study',
    'creators': [{'name': 'Smith, Jane', 'affiliation': 'University X'}],
    'keywords': ['food authentication', 'Raman spectroscopy', 'olive oil'],
    'license': 'CC-BY-4.0',
    'access_right': 'open',
    'upload_type': 'dataset'
}

# Upload via Zenodo API
response = requests.post(
    'https://zenodo.org/api/deposit/depositions',
    json={'metadata': zenodo_metadata},
    headers={'Authorization': f'Bearer {ZENODO_TOKEN}'}
)

dataset_doi = response.json()['doi']
print(f"Dataset deposited at: {dataset_doi}")
```

### Publication Integration

```python
# Auto-generate Methods section for manuscript
methods_section = f"""
**Methods**

*Data and Preprocessing.* Raw Raman spectra (n={len(ds)} samples) were acquired 
on a Bruker SENTERRA II spectrometer with {ds.metadata['laser_wavelength']} 
excitation. Preprocessing consisted of: (1) asymmetric least squares baseline 
correction (λ={pipeline['baseline']['lambda']}, p={pipeline['baseline']['p']}); 
(2) vector normalization; (3) Savitzky-Golay smoothing (window=9, poly_order=3). 
All preprocessing was applied to the complete dataset prior to train/test splitting 
to avoid data leakage.

*Feature Extraction.* Five features were extracted from spectral regions of 
interest: peak areas at 1651 cm⁻¹ (C=C stretch), 1438 cm⁻¹ (CH₂ bending), 
and 1275 cm⁻¹ (C-O stretching); and ratios C=C/CH₂ and C-O/C=C.

*Modeling and Validation.* A Random Forest classifier (100 trees, max_depth=5) 
was trained on 70% of samples and evaluated on the remaining 30% using nested 
5-fold cross-validation (inner 3-fold for hyperparameter tuning). Stratification 
controlled for batch effects and class imbalance. Performance was assessed using 
balanced accuracy, ROC-AUC, precision, recall, and F1 score with 95% confidence 
intervals via bootstrap.

*Reproducibility.* The complete analysis pipeline, including preprocessing 
configuration, model weights, and predictions, is available at {GITHUB_URL} 
(commit {GIT_COMMIT}). Raw and preprocessed spectra are deposited at Zenodo 
({ZENODO_DOI}) with full metadata and code to reproduce all results.
"""

# Save to file for pasting into manuscript
with open('methods_section.txt', 'w') as f:
    f.write(methods_section)
```

---

## Compliance with Journal Requirements

**Nature, Science, Cell, and similar journals now require:**

- [ ] **Code availability:** GitHub, Zenodo, or institutional repository
- [ ] **Data availability:** Raw and processed data in open repository
- [ ] **Methods detail:** Sufficient for independent replication
- [ ] **Statistical reporting:** Sample sizes, effect sizes, p-values, confidence intervals
- [ ] **Conflict of interest:** Funding and affiliations
- [ ] **Reproducibility statement:** "Code and data are available at [URL]"

**FoodSpec supports all requirements through:**
- Auto-generated methods text (`result.generate_methods_section()`)
- Artifact archiving (`result.save_reproducibility_bundle()`)
- Metadata capture (`ds.metadata`)
- Protocol version control (`.yaml` files in Git)

---

## Further Reading

- **FAIR Principles:** [Wilkinson et al. (2016)](https://www.nature.com/articles/sdata201618) — Scientific Data
- **Reproducibility Crisis:** [Open Science Collaboration (2015)](https://www.science.org/doi/10.1126/science.aac4716) — Science
- **Best Practices:** [Perkel (2021)](https://www.nature.com/articles/d41586-021-01174-8) — Tools and techniques for reproducible research

---

## Related Documentation

- [End-to-End Pipeline](workflows/end_to_end_pipeline.md) — Complete worked example with reproducibility
- [Cross-Validation and Data Leakage](methods/validation/cross_validation_and_leakage.md) — Prevent overfitting
- [Protocols and YAML Configuration](user-guide/protocols_and_yaml.md) — Protocol-driven workflows
- [Reproducibility Checklist](protocols/reproducibility_checklist.md) — Pre-publication checklist
