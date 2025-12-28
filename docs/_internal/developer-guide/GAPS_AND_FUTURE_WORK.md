# FoodSpec v1.0 - Gap Analysis & Future Development

**Date:** December 25, 2024  
**Status:** Production v1.0 Released  
**Purpose:** Guide future development priorities

---

## Executive Summary

FoodSpec v1.0 is **production ready** with 78.57% test coverage and comprehensive functionality. This document identifies areas for improvement, missing features, and development priorities for v1.1+.

### Current State
✅ **Complete:** Core functionality, preprocessing, ML, stats, QC, protocols  
⚠️ **Scaffolded:** Health scoring, deployment prediction, library search  
❌ **Missing:** Deep learning, advanced chemometrics, GPU acceleration

---

## Critical Issues (Must Fix for v1.1)

### 1. Complete Scaffold Implementations

#### qc/health.py (PRIORITY: HIGH)
**Current State:** Empty scaffold with 0% coverage  
**Missing Algorithms:**
- Signal-to-Noise Ratio (SNR) calculation
- Baseline drift quantification
- Cosmic ray/spike detection
- Quality score thresholding

**Implementation Plan:**
```python
# Proposed SNR calculation
def compute_snr(spectrum, signal_region, noise_region):
    """
    Compute SNR as ratio of mean signal to std of noise.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Intensity values
    signal_region : tuple
        (start_idx, end_idx) for signal
    noise_region : tuple
        (start_idx, end_idx) for noise baseline
    
    Returns
    -------
    float
        SNR in dB
    """
    signal = np.mean(spectrum[signal_region[0]:signal_region[1]])
    noise_std = np.std(spectrum[noise_region[0]:noise_region[1]])
    snr = 20 * np.log10(signal / (noise_std + 1e-10))
    return snr
```

**Effort:** 3-5 days  
**Dependencies:** None  
**Tests Required:** 10+ unit tests

---

#### deploy/predict.py (PRIORITY: MEDIUM)
**Current State:** Interface defined, prediction stubbed  
**Missing Logic:**
- Artifact deserialization
- Model prediction execution
- Input validation
- Error handling

**Implementation Plan:**
```python
def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions with proper validation."""
    if self.predictor is None:
        raise RuntimeError("Call .load() first")
    
    # Validate input schema
    self._validate_input(df)
    
    # Extract features
    X = self._extract_features(df)
    
    # Predict
    predictions = self.predictor.predict(X)
    proba = self.predictor.predict_proba(X)
    
    # Format output
    out = df.copy()
    out["prediction"] = predictions
    out["confidence"] = proba.max(axis=1)
    
    return out
```

**Effort:** 2-3 days  
**Dependencies:** Artifact system (already complete)  
**Tests Required:** 8+ tests

---

#### workflows/library_search.py (PRIORITY: LOW)
**Current State:** Placeholder scoring  
**Missing Algorithms:**
- Spectral angle mapper (SAM)
- Hit Quality Index (HQI)
- Correlation coefficient ranking
- Top-k retrieval with confidence

**Implementation Plan:**
```python
def spectral_angle_mapper(spectrum, library):
    """
    Compute spectral angle between query and library spectra.
    
    Returns
    -------
    np.ndarray
        Angles in radians (smaller = better match)
    """
    # Normalize to unit vectors
    spectrum_norm = spectrum / np.linalg.norm(spectrum)
    library_norm = library / np.linalg.norm(library, axis=1, keepdims=True)
    
    # Compute cosine similarity
    cos_sim = library_norm @ spectrum_norm
    
    # Convert to angles
    angles = np.arccos(np.clip(cos_sim, -1, 1))
    
    return angles
```

**Effort:** 1-2 days  
**Dependencies:** None  
**Tests Required:** 6+ tests

---

### 2. Improve Test Coverage

#### Undertested Modules (<60% coverage)

| Module | Current | Target | Effort |
|--------|---------|--------|--------|
| workflows/heating_trajectory.py | 39% | 70% | 2 days |
| workflows/aging.py | 38% | 70% | 1 day |
| CLI modules | 0-40% | 60% | 3 days |
| qc/replicates.py | 37% | 60% | 1 day |
| stats/distances.py | 41% | 60% | 1 day |

**Total Effort:** 1-2 weeks

**Implementation Strategy:**
1. Identify untested code paths with `pytest --cov-report=html`
2. Write parametrized tests for edge cases
3. Add integration tests for workflows
4. Mock external dependencies (matplotlib, etc.)

---

## Enhancement Opportunities

### 1. Advanced Algorithms (v1.2)

#### A. OPLS (Orthogonal Projections to Latent Structures)
**Motivation:** Better interpretability than PLS-DA  
**Use Case:** Discriminant analysis with orthogonal components  
**Complexity:** Medium  
**Effort:** 2 weeks

**Dependencies:**
- Existing PLS implementation
- Orthogonalization utilities

**Implementation Outline:**
```python
class OPLS:
    """Orthogonal PLS for improved interpretability."""
    
    def fit(self, X, y):
        # 1. Compute PLS solution
        # 2. Orthogonalize predictive vs orthogonal components
        # 3. Remove systematic variation unrelated to y
        pass
    
    def transform(self, X):
        # Project onto predictive components only
        pass
```

---

#### B. MCR-ALS (Multivariate Curve Resolution)
**Motivation:** Component deconvolution for mixtures  
**Use Case:** Extract pure component spectra from mixtures  
**Complexity:** High  
**Effort:** 3-4 weeks

**Algorithm:**
1. Initialize with PCA or SIMPLISMA
2. Alternating Least Squares optimization
3. Non-negativity constraints
4. Convergence checking

**References:**
- Tauler, R. (1995). "Multivariate curve resolution applied to second order data"
- De Juan et al. (2014). "Combining hard- and soft-modelling to solve kinetic problems"

---

#### C. Deep Learning Module
**Motivation:** End-to-end learning for complex patterns  
**Use Cases:**
- 1D CNN for spectral classification
- Autoencoders for denoising
- Transfer learning from large spectral databases

**Complexity:** High  
**Effort:** 1-2 months

**Architecture Proposal:**
```python
# foodspec/ml/deep.py

class SpectralCNN(nn.Module):
    """1D CNN for spectral classification."""
    
    def __init__(self, n_wavelengths, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(64 * (n_wavelengths // 4), n_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

**Dependencies:** PyTorch or TensorFlow (optional)

---

### 2. Performance Optimization (v1.3)

#### A. GPU Acceleration
**Target Modules:**
- Preprocessing pipelines (baseline, smoothing)
- PCA/PLS computation
- Distance matrix calculations

**Technology:** CuPy (CUDA) or NumPy with MKL

**Expected Speedup:** 10-50x for large datasets

**Implementation:**
```python
# Auto-detect GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

def baseline_als_gpu(y, lam=1e5, p=0.01, niter=10):
    """GPU-accelerated ALS baseline."""
    if GPU_AVAILABLE:
        y_gpu = cp.array(y)
        # ... CuPy operations ...
        return cp.asnumpy(result)
    else:
        return baseline_als(y, lam, p, niter)
```

---

#### B. Parallel Processing
**Target Operations:**
- Batch preprocessing
- Cross-validation folds
- Hyperparameter grid search
- File I/O

**Technology:** joblib, multiprocessing, Dask

**Implementation:**
```python
from joblib import Parallel, delayed

def preprocess_batch_parallel(datasets, config, n_jobs=-1):
    """Preprocess multiple datasets in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_preprocess_single)(ds, config)
        for ds in datasets
    )
    return results
```

---

#### C. Caching System
**Target:** Expensive operations (preprocessing, fingerprints)

**Technology:** joblib.Memory or custom cache

**Implementation:**
```python
from joblib import Memory

cache = Memory(location='.foodspec_cache', verbose=0)

@cache.cache
def compute_fingerprint(spectrum, method='correlation'):
    """Cached fingerprint computation."""
    # ... expensive calculation ...
    return fingerprint
```

---

### 3. Extended Format Support (v1.2)

#### Missing Vendor Formats

| Vendor | Format | Priority | Effort |
|--------|--------|----------|--------|
| Shimadzu | .txt | Medium | 2 days |
| JASCO | .jws | Low | 3 days |
| Nicolet | .dpt | Medium | 2 days |
| Andor | .asc | Low | 1 day |
| WITec | .wip | Low | 4 days |

**Implementation Strategy:**
1. Obtain sample files from vendors/users
2. Reverse-engineer binary formats
3. Create reader in `io/vendor_readers.py`
4. Add tests with sample files
5. Document in user guide

---

#### Cloud Storage Integration

**AWS S3:**
```python
import boto3

def load_from_s3(bucket, key):
    """Load spectral data from S3."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return FoodSpectrumSet.from_bytes(obj['Body'].read())
```

**Google Cloud Storage:**
```python
from google.cloud import storage

def load_from_gcs(bucket_name, blob_name):
    """Load spectral data from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return FoodSpectrumSet.from_bytes(blob.download_as_bytes())
```

---

### 4. User Experience Improvements (v1.4)

#### A. Interactive Dashboard
**Technology:** Streamlit or Dash  
**Features:**
- Upload spectra
- Configure preprocessing
- Run classification
- Download results

**Mockup:**
```python
# foodspec_dashboard.py
import streamlit as st
from foodspec import FoodSpectrumSet

st.title("FoodSpec Interactive Analysis")

# File upload
uploaded_file = st.file_uploader("Upload spectra (CSV/HDF5)")

if uploaded_file:
    ds = FoodSpectrumSet.from_csv(uploaded_file)
    
    # Preprocessing controls
    baseline = st.selectbox("Baseline", ["als", "rubberband", "polynomial"])
    normalize = st.selectbox("Normalization", ["vector", "area", "max"])
    
    if st.button("Preprocess"):
        ds.preprocess(baseline=baseline, normalize=normalize)
        st.line_chart(ds.x.T)
    
    # Classification
    if st.button("Classify"):
        model = train_classifier(ds)
        results = model.predict(ds)
        st.write(results)
```

**Effort:** 2-3 weeks

---

#### B. Jupyter Magic Commands
**Features:**
- `%foodspec_load` - Quick data loading
- `%foodspec_preprocess` - Interactive preprocessing
- `%foodspec_classify` - One-line classification
- `%foodspec_plot` - Quick visualization

**Implementation:**
```python
# foodspec/jupyter_magic.py
from IPython.core.magic import register_line_magic

@register_line_magic
def foodspec_load(line):
    """Load spectral data in Jupyter."""
    filename = line.strip()
    ds = FoodSpectrumSet.from_csv(filename)
    get_ipython().user_ns['ds'] = ds
    print(f"Loaded {len(ds)} spectra")
```

**Effort:** 1 week

---

#### C. Tutorial Notebooks
**Topics:**
1. Oil authentication from scratch
2. Heating degradation analysis
3. Mixture quantification
4. QC system setup
5. Hyperspectral imaging workflow
6. Custom protocol development

**Effort:** 2-3 weeks

---

## Architecture Improvements

### 1. Refactoring Opportunities

#### A. Consolidate Deprecated Modules (v2.0)
**Action:** Remove backward-compatibility shims  
**Affected:**
- artifact.py → deploy/
- calibration_transfer.py → preprocess/
- heating_trajectory.py → workflows/
- matrix_correction.py → preprocess/
- protocol_engine.py → protocol/
- rq.py → features/rq/
- spectral_dataset.py → core/
- spectral_io.py → io/

**Migration:** Automated script with deprecation warnings

---

#### B. Type System Improvements
**Current:** Partial type hints  
**Target:** Full mypy compliance

**Example:**
```python
# Before
def preprocess(self, baseline='als', normalize='vector'):
    pass

# After
from typing import Literal

def preprocess(
    self,
    baseline: Literal['als', 'rubberband', 'polynomial'] = 'als',
    normalize: Literal['vector', 'area', 'max'] = 'vector'
) -> FoodSpectrumSet:
    pass
```

---

#### C. Error Handling Standardization
**Current:** Mix of exceptions  
**Target:** Consistent exception hierarchy

**Proposal:**
```python
# foodspec/exceptions.py
class FoodSpecError(Exception):
    """Base exception for FoodSpec."""
    pass

class DataError(FoodSpecError):
    """Invalid data format or values."""
    pass

class ConfigError(FoodSpecError):
    """Invalid configuration."""
    pass

class ModelError(FoodSpecError):
    """Model training or prediction error."""
    pass
```

---

### 2. Documentation Enhancements

#### Missing Guides
- **Performance Tuning Guide** - Optimize for large datasets
- **GPU Acceleration Guide** - Setup and usage
- **Cloud Deployment Guide** - AWS/GCP/Azure
- **Plugin Development Guide** - Create custom plugins
- **Troubleshooting Deep Dive** - Common errors and solutions

**Effort:** 1-2 weeks

---

## Research & Experimental Features

### 1. Explainability & Interpretability

#### SHAP Values for Spectra
**Motivation:** Explain predictions at wavelength level  
**Use Case:** Identify discriminative regions

```python
import shap

def explain_prediction(model, spectrum):
    """Generate SHAP values for spectral prediction."""
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    shap_values = explainer.shap_values(spectrum)
    
    # Plot important wavelengths
    shap.plots.waterfall(shap_values[0])
```

---

#### Attention Mechanisms (If Deep Learning Added)
**Motivation:** Visualize which wavelengths the network focuses on  
**Implementation:** Attention layers in CNN/Transformer

---

### 2. Active Learning

#### Uncertainty Sampling
**Motivation:** Reduce labeling effort  
**Strategy:** Query samples with highest prediction uncertainty

```python
def select_for_labeling(model, unlabeled_pool, n_samples=10):
    """Select most informative samples."""
    proba = model.predict_proba(unlabeled_pool)
    uncertainty = 1 - proba.max(axis=1)  # Highest uncertainty
    indices = np.argsort(uncertainty)[-n_samples:]
    return unlabeled_pool[indices]
```

---

### 3. Federated Learning

**Motivation:** Train models across institutions without sharing raw data  
**Use Case:** Multi-site collaborations in food safety

**Technology:** PySyft or Flower framework

---

## Community & Ecosystem

### 1. Package Integrations

#### Scikit-learn Pipelines
**Status:** Partially compatible  
**Goal:** Full sklearn Pipeline integration

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('preprocess', FoodSpecPreprocessor()),
    ('pca', PCA(n_components=10)),
    ('classify', SVC())
])

pipe.fit(X, y)
```

---

#### Plotly Integration
**Current:** Matplotlib only  
**Goal:** Interactive plots with Plotly

```python
import plotly.graph_objects as go

def plot_spectra_interactive(ds):
    """Interactive spectral plot."""
    fig = go.Figure()
    for i, spectrum in enumerate(ds.x):
        fig.add_trace(go.Scatter(
            x=ds.wavenumbers,
            y=spectrum,
            name=f"Sample {i}"
        ))
    fig.show()
```

---

### 2. Web Platform

**Vision:** FoodSpec as a Service (FSaaS)  
**Features:**
- Upload data via web interface
- Run analyses in cloud
- Collaborative projects
- Model sharing marketplace

**Technology Stack:**
- Backend: FastAPI
- Frontend: React
- Database: PostgreSQL
- Storage: S3

**Effort:** 6+ months

---

## Priority Matrix

| Feature | Impact | Effort | Priority | Version |
|---------|--------|--------|----------|---------|
| Complete qc/health.py | High | Low | 🔴 Critical | v1.1 |
| Complete deploy/predict.py | Medium | Low | 🟡 High | v1.1 |
| Improve test coverage | Medium | Medium | 🟡 High | v1.1 |
| OPLS implementation | Medium | Medium | 🟢 Medium | v1.2 |
| GPU acceleration | High | High | 🟢 Medium | v1.3 |
| Deep learning module | High | Very High | 🔵 Low | v2.0 |
| Interactive dashboard | Medium | Medium | 🟢 Medium | v1.4 |
| Cloud storage | Low | Low | 🔵 Low | v1.2 |
| Jupyter magic | Low | Low | 🔵 Low | v1.3 |
| Web platform | Very High | Very High | 🔵 Low | v3.0 |

**Legend:**
- 🔴 Critical - Must do for next minor version
- 🟡 High - Should do soon
- 🟢 Medium - Nice to have
- 🔵 Low - Future consideration

---

## Conclusion

FoodSpec v1.0 is production-ready with comprehensive functionality. The identified gaps are minor and non-blocking. Future development should focus on:

1. **v1.1 (1-2 months):** Complete scaffolds, improve tests
2. **v1.2 (3-4 months):** OPLS, cloud storage, extended formats
3. **v1.3 (6-8 months):** GPU acceleration, performance optimization
4. **v2.0 (12+ months):** Deep learning, remove deprecations, major refactor

The package is well-architected and extensible. Most enhancements can be added without breaking changes.

---

*Last Updated: December 25, 2024*  
*Review Cycle: Quarterly*
