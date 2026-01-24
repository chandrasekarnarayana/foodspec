# FoodSpec Engineering Rules

**Document Version**: 1.0  
**Last Updated**: 2026-01-24  
**Status**: Active

This document codifies the non-negotiable engineering principles for FoodSpec refactoring and ongoing development. These rules ensure the framework remains **reliable, reproducible, maintainable, and scientifically sound**.

---

## Table of Contents

1. [Overview](#overview)
2. [Rule Categories](#rule-categories)
3. [Detailed Rules](#detailed-rules)
4. [Tooling & Automation](#tooling--automation)
5. [Validation Checklist](#validation-checklist)
6. [FAQ](#faq)

---

## Overview

FoodSpec is transitioning to a **protocol-driven spectroscopy framework**. This refactor must not compromise scientific reproducibility, user trust, or code maintainability. The seven rules below are **mandatory**; exceptions require documented justification and approval.

### Goals

✓ Reproducibility: Same inputs + seed → identical outputs  
✓ Transparency: No hidden state, explicit dependencies  
✓ Usability: Clear errors, comprehensive docs  
✓ Quality: High test coverage, type safety  
✓ Interoperability: Serializable pipelines, clear interfaces  

---

## Rule Categories

| Rule | Focus | Severity |
|------|-------|----------|
| Deterministic Outputs | Reproducibility | CRITICAL |
| No Hidden Global State | Transparency | CRITICAL |
| Documented Public APIs | Usability | HIGH |
| Tests + Docs Required | Quality | HIGH |
| Metadata Validation | Correctness | HIGH |
| Serializable Pipelines | Interoperability | MEDIUM |
| Actionable Errors | User Experience | MEDIUM |

---

## Detailed Rules

### Rule 1: Deterministic Outputs

**Principle**: For any given input and random seed, the output must be 100% reproducible.

#### Rationale
Spectroscopy research demands reproducibility. Users must be able to rerun analyses and get identical results. This is non-negotiable for peer review, regulatory compliance, and scientific integrity.

#### Implementation

**1.1: Explicit Random Seed Parameter**

Every function that uses randomness must accept a `seed` or `random_state` parameter:

```python
def synthetic_raman_spectrum(n_peaks: int = 10, 
                             seed: int | None = None) -> np.ndarray:
    """Generate a synthetic Raman spectrum.
    
    Parameters
    ----------
    n_peaks : int
        Number of peaks to generate.
    seed : int, optional
        Random seed for reproducibility. If None, results are non-deterministic.
    
    Returns
    -------
    np.ndarray
        1D spectrum array.
    
    Examples
    --------
    >>> spec1 = synthetic_raman_spectrum(seed=42)
    >>> spec2 = synthetic_raman_spectrum(seed=42)
    >>> np.array_equal(spec1, spec2)
    True
    """
    rng = np.random.default_rng(seed)
    peaks = rng.uniform(1000, 3000, n_peaks)
    # ... generate spectrum using rng
    return spectrum
```

**1.2: Use numpy.random.default_rng, NOT np.random**

```python
# ✅ GOOD
rng = np.random.default_rng(seed)
samples = rng.normal(loc=0, scale=1, size=100)

# ❌ BAD
np.random.seed(seed)  # Modifies global state
samples = np.random.normal(loc=0, scale=1, size=100)
```

**1.3: For SciPy/Scikit-learn, Pass random_state**

```python
# ✅ GOOD
from scipy.stats import gaussian_kde
kde = gaussian_kde(data, bw_method='scott')  # Deterministic for fixed input

from sklearn.decomposition import PCA
pca = PCA(n_components=5, random_state=42)
pca.fit(X)
```

**1.4: Test Determinism**

```python
def test_synthetic_spectrum_determinism():
    """Verify identical seeds produce identical outputs."""
    spec1 = synthetic_raman_spectrum(seed=42)
    spec2 = synthetic_raman_spectrum(seed=42)
    np.testing.assert_array_equal(spec1, spec2)

def test_synthetic_spectrum_different_seed_different_output():
    """Verify different seeds produce different outputs."""
    spec1 = synthetic_raman_spectrum(seed=42)
    spec2 = synthetic_raman_spectrum(seed=43)
    assert not np.array_equal(spec1, spec2)
```

#### Anti-Patterns (Forbidden)

```python
# ❌ NO: Relies on global state
_SEED = 42
def analyze_spectrum(spectrum):
    np.random.seed(_SEED)
    ...

# ❌ NO: Non-deterministic, no seed option
def cluster_peaks(peaks):
    centers = kmeans(peaks, k=5)
    ...

# ❌ NO: Hidden randomness in library config
_CONFIG = {"random_state": None}
def fit_model(X):
    model = SomeModel(random_state=_CONFIG["random_state"])
    ...
```

---

### Rule 2: No Hidden Global State

**Principle**: All mutable state must be explicit, passed as parameters, and traceable.

#### Rationale
Global state creates invisible dependencies, makes testing difficult, causes race conditions in parallel execution, and violates the dependency injection principle.

#### Implementation

**2.1: Use Dataclasses/Pydantic for Configuration**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class BaselineConfig:
    """Baseline subtraction configuration."""
    method: Literal["als", "polynomial", "rubberband"] = "als"
    poly_order: int = 3  # For polynomial method
    lam: float = 1e4     # For ALS
    
    def __post_init__(self):
        if self.method == "polynomial" and self.poly_order < 1:
            raise ValueError("poly_order must be >= 1")

def apply_baseline(spectrum: np.ndarray, 
                   config: BaselineConfig = BaselineConfig()) -> np.ndarray:
    """Apply baseline correction."""
    if config.method == "als":
        return baseline_als(spectrum, lam=config.lam)
    elif config.method == "polynomial":
        return baseline_polynomial(spectrum, order=config.poly_order)
    ...
```

**2.2: Avoid Module-Level Mutable State**

```python
# ❌ BAD: Global config
_METADATA_SCHEMA = {}

def register_metadata_field(name, validator):
    _METADATA_SCHEMA[name] = validator  # Mutable global

# ✅ GOOD: Passed explicitly
class MetadataRegistry:
    def __init__(self):
        self.schema = {}
    
    def register(self, name, validator):
        self.schema[name] = validator

registry = MetadataRegistry()  # Instantiated, not hidden
```

**2.3: Immutable by Default**

```python
from dataclasses import dataclass

# ✅ GOOD: Frozen (immutable)
@dataclass(frozen=True)
class SpectrumMetadata:
    wavelength_start: float
    wavelength_end: float

# ❌ BAD: Mutable (allows accidental modifications)
@dataclass
class SpectrumMetadata:
    wavelength_start: float
    wavelength_end: float
```

**2.4: No Singletons (Except with Explicit Justification)**

```python
# ❌ BAD: Hidden singleton
class Logger:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# ✅ GOOD: Explicit dependency injection
def process_spectrum(spectrum, logger: Logger):
    logger.info("Processing...")
    ...

# If singleton is truly necessary, document it:
class ProtocolRegistry:
    """
    Singleton registry of all available spectroscopy protocols.
    
    JUSTIFICATION: Protocols are immutable, read-only, and must be 
    accessible globally. Registration happens at import time and is 
    never modified after initialization.
    
    WARNING: Do NOT use for mutable state or configuration.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ProtocolRegistry._initialized:
            return
        self._protocols = {}
        ProtocolRegistry._initialized = True
    
    def register(self, name: str, protocol):
        if ProtocolRegistry._initialized and self._protocols:
            raise RuntimeError("Cannot register after initialization")
        self._protocols[name] = protocol
```

#### Anti-Patterns (Forbidden)

```python
# ❌ NO: Module-level list
ACTIVE_EXPERIMENTS = []

def start_experiment():
    ACTIVE_EXPERIMENTS.append(...)

# ❌ NO: Hidden thread-local state
_thread_local = threading.local()

def set_context(ctx):
    _thread_local.context = ctx

# ❌ NO: Mutable default arguments
def update_config(new_settings, current_config={}):
    current_config.update(new_settings)  # Persists across calls!
    ...
```

---

### Rule 3: Every Public Function/Class Must Have Docstring + Example

**Principle**: All public APIs must be self-documenting with usage examples.

#### Rationale
Users cannot use what they don't understand. Comprehensive documentation prevents API misuse, reduces support burden, and enables IDE autocomplete.

#### Implementation

**3.1: NumPy-Style Docstrings**

```python
def compute_signal_quality_metrics(spectrum: np.ndarray,
                                   signal_region: tuple[int, int],
                                   noise_region: tuple[int, int],
                                   method: Literal["snr", "slr"] = "snr") -> float:
    """Compute signal quality metrics for a spectrum.
    
    Calculates signal-to-noise ratio (SNR) or signal-to-lower-region 
    ratio (SLR) to assess spectrum quality and detect anomalies.
    
    Parameters
    ----------
    spectrum : np.ndarray
        1D spectral intensity array (m/z or wavenumber axis).
    signal_region : tuple[int, int]
        (start, end) indices [0, len(spectrum)) of signal peak.
    noise_region : tuple[int, int]
        (start, end) indices of noise baseline region.
    method : {"snr", "slr"}, default "snr"
        Quality metric. SNR assumes noise region is uniform background.
        SLR compares signal to lower intensity region.
    
    Returns
    -------
    float
        Signal quality metric in decibels (dB).
    
    Raises
    ------
    ValueError
        If signal_region or noise_region are invalid (start >= end,
        indices out of bounds, or regions overlap).
    TypeError
        If spectrum is not a 1D numpy array.
    
    Notes
    -----
    SNR = 10 * log10(P_signal / P_noise), where P_* is average power.
    
    References
    ----------
    .. [1] ISO 13849-1:2015, Functional safety of machines and controls.
    
    Examples
    --------
    >>> import numpy as np
    >>> spectrum = np.array([0.1, 0.2, 1.5, 1.2, 0.3, 0.15, 0.1])
    >>> snr = compute_signal_quality_metrics(spectrum, 
    ...                                        signal_region=(1, 4),
    ...                                        noise_region=(4, 7))
    >>> snr > 3  # Expect positive SNR
    True
    
    >>> # Compare methods
    >>> snr = compute_signal_quality_metrics(spectrum, (1, 4), (4, 7), method="snr")
    >>> slr = compute_signal_quality_metrics(spectrum, (1, 4), (4, 7), method="slr")
    >>> snr != slr
    True
    """
```

**3.2: Type Hints Mandatory**

```python
# ✅ GOOD: Clear types
def load_spectrum(path: str, 
                  format: Literal["csv", "hdf5"] = "csv") -> np.ndarray:
    ...

def align_spectra(spectra: list[np.ndarray],
                  reference_idx: int = 0) -> list[np.ndarray]:
    ...

# ❌ BAD: Missing types
def load_spectrum(path, format="csv"):
    ...

def align_spectra(spectra, reference_idx=0):
    ...
```

**3.3: Examples Section Required**

```python
# ✅ GOOD
class BaselineCorrector:
    """Apply baseline correction to spectra.
    
    Examples
    --------
    >>> import numpy as np
    >>> from foodspec import BaselineCorrector
    >>> spectrum = np.array([0.1, 0.15, 1.0, 0.95, 0.2, 0.12])
    >>> corrector = BaselineCorrector(method="als", lam=1e4)
    >>> corrected = corrector.fit_transform(spectrum)
    >>> corrected.shape
    (6,)
    """

# ❌ BAD: No example
class BaselineCorrector:
    """Apply baseline correction to spectra."""

# ❌ BAD: Non-runnable example
class BaselineCorrector:
    """Apply baseline correction.
    
    Examples
    --------
    Use like: corrector = BaselineCorrector()
    """
```

**3.4: Document Private vs Public**

```python
def public_function():
    """User-facing API. Must have docstring + example."""
    ...

def _internal_helper():
    """
    Internal function for _public_function.
    
    Do NOT call directly; interface may change without warning.
    """
    ...
```

#### Anti-Patterns (Forbidden)

```python
# ❌ NO: Missing docstring
def normalize_spectrum(spectrum):
    return spectrum / np.sum(spectrum)

# ❌ NO: Incomplete signature
def cluster_spectra(spectra, k=5):
    """Cluster spectra."""
    ...

# ❌ NO: Non-runnable example
def smooth_spectrum(spectrum):
    """Smooth using Savitzky-Golay filter.
    
    Examples
    --------
    Use: result = smooth_spectrum(my_data, window=5, poly_order=2)
    """
    ...
```

---

### Rule 4: Every New Feature Must Include Tests + Docs

**Principle**: Code without tests or documentation should not be merged.

#### Rationale
Tests catch regressions and verify expected behavior. Docs ensure users can adopt the feature. Together, they define "done."

#### Implementation

**4.1: Test Coverage ≥ 80% (New Code)**

```bash
# Run tests with coverage report
pytest tests/ --cov=src/foodspec --cov-report=html

# HTML report: htmlcov/index.html
# Target: ≥80% coverage for any new module
```

**4.2: Test File Organization**

```
src/foodspec/
├── core/
│   ├── baseline.py          # Implementation
│   └── __init__.py
tests/
├── core/
│   ├── test_baseline.py     # Tests mirror structure
│   └── __init__.py
```

**4.3: Test Structure Template**

```python
import pytest
import numpy as np
from foodspec.core.baseline import BaselineALS, BaselineConfig

class TestBaselineALS:
    """Test suite for ALS baseline subtraction."""
    
    @pytest.fixture
    def sample_spectrum(self):
        """Create a noisy synthetic spectrum with baseline."""
        np.random.seed(42)
        x = np.arange(100)
        signal = 10 * np.exp(-0.05 * (x - 50)**2)
        baseline = 0.1 * x
        noise = np.random.normal(0, 0.05, 100)
        return signal + baseline + noise
    
    def test_initialization_with_defaults(self):
        """Test BaselineALS initializes with default parameters."""
        corrector = BaselineALS()
        assert corrector.lam == 1e4
        assert corrector.p == 0.01
    
    def test_initialization_with_custom_params(self):
        """Test BaselineALS accepts custom parameters."""
        corrector = BaselineALS(lam=1e5, p=0.001)
        assert corrector.lam == 1e5
        assert corrector.p == 0.001
    
    def test_deterministic_with_seed(self, sample_spectrum):
        """Test that processing is deterministic (no random state)."""
        corrector1 = BaselineALS(seed=42)
        corrector2 = BaselineALS(seed=42)
        result1 = corrector1.fit_transform(sample_spectrum)
        result2 = corrector2.fit_transform(sample_spectrum)
        np.testing.assert_array_equal(result1, result2)
    
    def test_corrects_positive_baseline(self, sample_spectrum):
        """Test that baseline correction removes positive trends."""
        corrector = BaselineALS()
        corrected = corrector.fit_transform(sample_spectrum)
        # Corrected should have lower offset than original
        assert np.mean(corrected) < np.mean(sample_spectrum)
    
    def test_invalid_spectrum_raises(self):
        """Test that invalid input raises ValueError."""
        corrector = BaselineALS()
        with pytest.raises(ValueError, match="spectrum must be 1D"):
            corrector.fit_transform(np.zeros((10, 10)))
    
    def test_negative_lam_raises(self):
        """Test that negative lambda raises ValueError."""
        with pytest.raises(ValueError, match="lam must be positive"):
            BaselineALS(lam=-100)
```

**4.4: Documentation Template**

Add to `docs/reference/` or `docs/methods/`:

```markdown
# Baseline Correction

## Overview

Baseline correction removes the underlying baseline trend from spectra, revealing true peak heights and areas. This is critical for quantitative analysis and spectral library matching.

## Available Methods

- **Asymmetric Least Squares (ALS)**: Fast, robust, suitable for most workflows
- **Polynomial**: Simple, interpretable, good for smooth trends
- **Rubberband**: Non-parametric, preserves peak shapes

## Usage

```python
from foodspec import BaselineCorrector

spectrum = load_spectrum("my_spectrum.csv")
corrector = BaselineCorrector(method="als", lam=1e4)
corrected = corrector.fit_transform(spectrum)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| method | str | "als" | Baseline method |
| lam | float | 1e4 | Smoothness (ALS) |
| p | float | 0.01 | Asymmetry (ALS) |

## References

[1] Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with asymmetric least squares smoothing. Leiden: University of Leiden.
```

#### Anti-Patterns (Forbidden)

```python
# ❌ NO: Feature with no tests
def new_filter_method(spectrum):
    return scipy.signal.medfilt(spectrum, kernel_size=5)

# ❌ NO: Feature with no documentation
# (Code without docs in issue or PR description)

# ❌ NO: Tests that don't verify the feature
def test_new_filter():
    result = new_filter_method(np.array([1, 2, 3]))
    assert result is not None  # Useless test
```

---

### Rule 5: Metadata Schema Must Be Validated Early

**Principle**: Validate all structured data at entry points; raise exceptions immediately if invalid.

#### Rationale
Early validation prevents cascading errors deep in pipelines, where failures are hard to debug. It also enforces consistent data quality upstream.

#### Implementation

**5.1: Use Pydantic Models for Validation**

```python
from pydantic import BaseModel, field_validator, ConfigDict
from datetime import datetime

class SpectrumMetadata(BaseModel):
    """Structured metadata for a spectrum.
    
    Examples
    --------
    >>> meta = SpectrumMetadata(
    ...     instrument_id="Raman-001",
    ...     timestamp="2025-01-24T10:30:00",
    ...     wavenumber_start=400,
    ...     wavenumber_end=3500
    ... )
    >>> meta.instrument_id
    'Raman-001'
    """
    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)
    
    instrument_id: str
    timestamp: str  # ISO 8601
    wavenumber_start: float
    wavenumber_end: float
    sample_name: str | None = None
    operator: str | None = None
    
    @field_validator('instrument_id')
    @classmethod
    def validate_instrument_id(cls, v):
        if not v.strip():
            raise ValueError("instrument_id cannot be empty")
        return v
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(
                f"timestamp '{v}' must be ISO 8601 format (e.g., 2025-01-24T10:30:00). "
                "Use datetime.datetime.now().isoformat()."
            )
        return v
    
    @field_validator('wavenumber_end')
    @classmethod
    def validate_wavenumber_range(cls, v, info):
        if 'wavenumber_start' in info.data:
            if v <= info.data['wavenumber_start']:
                raise ValueError(
                    f"wavenumber_end ({v}) must be > wavenumber_start ({info.data['wavenumber_start']})"
                )
        return v

# Usage
try:
    meta = SpectrumMetadata(
        instrument_id="Raman-001",
        timestamp="invalid-date",
        wavenumber_start=400,
        wavenumber_end=3500
    )
except ValueError as e:
    print(f"Invalid metadata: {e}")
    # Output: Invalid metadata: timestamp 'invalid-date' must be ISO 8601 format...
```

**5.2: Validate on Input**

```python
def process_spectra(spectra: list[np.ndarray],
                    metadata: list[dict]) -> list[np.ndarray]:
    """Process spectra with metadata.
    
    Parameters
    ----------
    spectra : list[np.ndarray]
        List of 1D spectra.
    metadata : list[dict]
        List of metadata dictionaries.
    
    Raises
    ------
    ValueError
        If lengths don't match or metadata is invalid.
    """
    # Validate lengths
    if len(spectra) != len(metadata):
        raise ValueError(
            f"Number of spectra ({len(spectra)}) must match metadata ({len(metadata)})"
        )
    
    # Validate each metadata entry
    validated_metadata = []
    for i, meta_dict in enumerate(metadata):
        try:
            meta = SpectrumMetadata(**meta_dict)
            validated_metadata.append(meta)
        except ValueError as e:
            raise ValueError(f"metadata[{i}] is invalid: {e}")
    
    # Now safe to process
    results = []
    for spectrum, meta in zip(spectra, validated_metadata):
        results.append(_process_single(spectrum, meta))
    
    return results
```

**5.3: Serialization-Roundtrip Testing**

```python
def test_metadata_serialization():
    """Test that metadata survives serialize→deserialize cycle."""
    original = SpectrumMetadata(
        instrument_id="Raman-001",
        timestamp="2025-01-24T10:30:00",
        wavenumber_start=400,
        wavenumber_end=3500
    )
    
    # Serialize to dict
    data = original.model_dump()
    
    # Deserialize from dict
    restored = SpectrumMetadata(**data)
    
    # Should be identical
    assert restored == original
```

#### Anti-Patterns (Forbidden)

```python
# ❌ NO: Defer validation
def load_spectrum_file(path: str) -> dict:
    data = yaml.safe_load(open(path))
    return data  # Unvalidated! Errors happen downstream

# ❌ NO: Partial validation
def validate_metadata(meta: dict):
    if 'timestamp' not in meta:
        raise ValueError("missing timestamp")
    # But doesn't validate timestamp format, instrument_id, etc.

# ❌ NO: Silent failures
def process_metadata(meta: dict):
    instrument = meta.get('instrument_id', 'UNKNOWN')  # Silent default
    timestamp = meta.get('timestamp', datetime.now())  # Silent default
    # User never knows data is corrupt
```

---

### Rule 6: Pipelines Must Be Serializable

**Principle**: All workflow configurations must serialize to JSON/YAML and be reproducible.

#### Rationale
Serialization enables:
- **Reproducibility**: Save and rerun the exact same pipeline.
- **Sharing**: Send pipeline configs between collaborators.
- **Archival**: Store historical configurations for audit trails.
- **Automation**: Integrate with job schedulers, CI/CD, etc.

#### Implementation

**6.1: Use Dataclasses for Pipeline Configuration**

```python
from dataclasses import dataclass, asdict
import json

@dataclass
class PreprocessingStep:
    """Single preprocessing step."""
    method: str  # "baseline", "normalize", "smooth", etc.
    parameters: dict  # Method-specific parameters
    enabled: bool = True
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

@dataclass
class PreprocessingPipeline:
    """Complete preprocessing workflow."""
    name: str
    description: str = ""
    steps: list[PreprocessingStep] | None = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
    
    def add_step(self, method: str, parameters: dict, enabled: bool = True):
        """Add a preprocessing step."""
        self.steps.append(PreprocessingStep(method, parameters, enabled))
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps]
        }, indent=2)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps]
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        pipeline = cls(name=d['name'], description=d.get('description', ''))
        for step_data in d.get('steps', []):
            pipeline.steps.append(PreprocessingStep.from_dict(step_data))
        return pipeline
    
    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

# Usage
pipeline = PreprocessingPipeline(name="Standard Raman Preprocessing")
pipeline.add_step("baseline", {"method": "als", "lam": 1e4})
pipeline.add_step("normalize", {"method": "l2"})
pipeline.add_step("smooth", {"method": "savgol", "window": 5, "poly_order": 2})

# Serialize
json_str = pipeline.to_json()
print(json_str)

# Deserialize
restored = PreprocessingPipeline.from_json(json_str)
assert restored.name == pipeline.name
assert len(restored.steps) == len(pipeline.steps)
```

**6.2: Save/Load from File**

```python
import json
from pathlib import Path

def save_pipeline_config(pipeline: PreprocessingPipeline, path: str):
    """Save pipeline configuration to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(pipeline.to_json())

def load_pipeline_config(path: str) -> PreprocessingPipeline:
    """Load pipeline configuration from JSON file."""
    with open(path, 'r') as f:
        return PreprocessingPipeline.from_json(f.read())

# Usage
save_pipeline_config(pipeline, "pipelines/standard_raman.json")
loaded = load_pipeline_config("pipelines/standard_raman.json")
```

**6.3: Versioning**

```python
@dataclass
class PreprocessingPipeline:
    version: str = "1.0"  # Semantic versioning
    # ... other fields
    
    def __post_init__(self):
        # Validate version format
        if not self.version.count('.') >= 1:
            raise ValueError("version must be semantic (e.g., '1.0', '1.2.3')")
```

**6.4: Tests for Serialization**

```python
def test_pipeline_serialization_roundtrip():
    """Test that pipeline survives serialize→deserialize cycle."""
    original = PreprocessingPipeline(name="Test Pipeline")
    original.add_step("baseline", {"method": "als", "lam": 1e4})
    original.add_step("normalize", {"method": "l2"})
    
    # Serialize to JSON and back
    json_str = original.to_json()
    restored = PreprocessingPipeline.from_json(json_str)
    
    # Should be identical
    assert restored.name == original.name
    assert len(restored.steps) == len(original.steps)
    for orig_step, rest_step in zip(original.steps, restored.steps):
        assert orig_step.method == rest_step.method
        assert orig_step.parameters == rest_step.parameters
```

#### Anti-Patterns (Forbidden)

```python
# ❌ NO: Non-serializable config
class PipelineConfig:
    def __init__(self):
        self.preprocessors = [lambda x: x / np.sum(x)]  # Functions not serializable!
        self.validator = check_metadata  # Function reference

# ❌ NO: Config with global references
@dataclass
class Pipeline:
    global_cache = {}  # Mutable, non-serializable
    ...

# ❌ NO: Implicit config (hardcoded in code)
def run_analysis(spectra):
    # Magic numbers everywhere, no way to save/load this config
    baseline = baseline_als(spectra, lam=1e4, p=0.01)
    normalized = normalize(baseline, method='l2')
    ...
```

---

### Rule 7: Errors Must Be Actionable

**Principle**: Every error message must clearly state **what failed**, **why**, and **how to fix it**.

#### Rationale
Users waste time debugging cryptic error messages. Actionable errors reduce support burden and improve user satisfaction.

#### Implementation

**7.1: Error Message Template**

```
[ERROR TYPE]: [WHAT FAILED]
[WHY IT FAILED / CONTEXT]
[HOW TO FIX IT / SUGGESTED ACTION]
```

**7.2: Good Error Messages**

```python
# ✅ GOOD
try:
    metadata = SpectrumMetadata(timestamp="invalid")
except ValueError as e:
    # Error: metadata['timestamp'] = 'invalid' is not ISO 8601 format.
    # Expected format: 'YYYY-MM-DDTHH:MM:SS' (e.g., 2025-01-24T10:30:00).
    # Fix: Use datetime.datetime.now().isoformat() to generate valid timestamps.
    ...

# ✅ GOOD
def load_spectrum_file(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Spectrum file not found: '{path}'.\n"
            f"Searched: {Path(path).resolve()}\n"
            f"Fix: Check path and ensure file exists. "
            f"Supported formats: CSV, HDF5, NetCDF."
        )
    ...

# ✅ GOOD
def align_spectra(spectra, reference_idx=0):
    if reference_idx >= len(spectra):
        raise ValueError(
            f"reference_idx ({reference_idx}) out of range [0, {len(spectra) - 1}].\n"
            f"You provided {len(spectra)} spectra.\n"
            f"Fix: Use reference_idx in range [0, {len(spectra) - 1}]."
        )
    ...
```

**7.3: Use Specific Exception Types**

```python
# ✅ GOOD: Specific, informative
raise ValueError("...")  # For bad values
raise TypeError("...")   # For wrong types
raise FileNotFoundError("...")  # For missing files
raise RuntimeError("...")  # For operational failures

# ❌ BAD: Generic
raise Exception("Error!")
raise RuntimeError("Something went wrong")
```

**7.4: Include Context**

```python
# ✅ GOOD: Includes context
def process_spectrum_batch(spectra: list, batch_size=100):
    if len(spectra) == 0:
        raise ValueError(
            "Cannot process empty spectrum list. "
            "You passed an empty list to process_spectrum_batch().\n"
            "Fix: Load or generate at least one spectrum before processing."
        )

# ❌ BAD: No context
def process_spectrum_batch(spectra: list):
    assert len(spectra) > 0  # No explanation
```

**7.5: Nested Exception Context**

```python
# ✅ GOOD: Chains exceptions and adds context
def load_library(path: str):
    try:
        return json.load(open(path))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Library file '{path}' contains invalid JSON.\n"
            f"JSON error: {e}\n"
            f"Fix: Validate JSON syntax using jsonlint or your editor.\n"
            f"Example: python -m json.tool {path}"
        ) from e
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Library file not found: {path}\n"
            f"Checked: {Path(path).resolve()}\n"
            f"Fix: Provide correct path. Use absolute path or relative to current directory."
        ) from e
```

**7.6: Test Error Messages**

```python
def test_error_message_clarity():
    """Verify error messages are actionable."""
    with pytest.raises(ValueError, match="reference_idx.*out of range"):
        align_spectra([spectrum1, spectrum2], reference_idx=10)
    
    # Verify message includes: what failed, why, how to fix
    try:
        align_spectra([spectrum1], reference_idx=10)
    except ValueError as e:
        assert "reference_idx" in str(e)
        assert "out of range" in str(e)
        assert "0" in str(e)  # Lower bound
```

#### Anti-Patterns (Forbidden)

```python
# ❌ NO: Vague error
raise ValueError("Invalid input")

# ❌ NO: No context
raise RuntimeError("Error in processing")

# ❌ NO: Cryptic
raise Exception("Check docstring")

# ❌ NO: Don't swallow exceptions
try:
    metadata = parse_metadata(data)
except Exception:
    pass  # Silent failure!

# ❌ NO: Assertions for user input (they disappear in production with -O flag)
def load_spectrum(path):
    assert Path(path).exists()  # WRONG! Use if+raise
```

---

## Tooling & Automation

### Enforcing Rules with Tools

| Rule | Tool | Command | Config |
|------|------|---------|--------|
| 1 (Deterministic) | Manual review | N/A | Code review checklist |
| 2 (No Global State) | Pylint, Manual | `ruff check` | `.ruff.toml` |
| 3 (Docstrings) | Pydocstyle, Pylint | `pydocstyle src/` | `.pylintrc` |
| 4 (Tests + Docs) | Pytest, Manual | `pytest --cov` | `pyproject.toml` |
| 5 (Validation) | Pydantic | Auto via `pydantic` | Model definitions |
| 6 (Serializable) | Unit tests | `pytest tests/test_*serialization*` | Manual test |
| 7 (Actionable Errors) | Manual review | N/A | Code review checklist |

### Recommended Setup

```bash
# Setup pre-commit hooks
pip install pre-commit

# Create .pre-commit-config.yaml (see CONTRIBUTING.md)
pre-commit install

# Run all checks locally
ruff format src/ tests/
ruff check src/ tests/ --fix
mypy src/ --strict
pytest tests/ --cov=src/foodspec
```

### CI/CD Integration

```yaml
# .github/workflows/tests.yml
name: Tests & Validation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: ruff format --check src/ tests/
      - run: ruff check src/ tests/
      - run: mypy src/ --strict
      - run: pytest tests/ --cov=src/foodspec --cov-fail-under=80
```

---

## Validation Checklist

Use this checklist **before submitting any PR**:

```markdown
## Engineering Rules Validation

### Rule 1: Deterministic Outputs
- [ ] All functions using randomness have `seed` or `random_state` parameter
- [ ] Using `np.random.default_rng()`, NOT `np.random.seed()`
- [ ] Test: `test_determinism_with_seed()` and `test_different_seeds_different_results()`

### Rule 2: No Hidden Global State
- [ ] No module-level mutable defaults (lists, dicts)
- [ ] Using dataclasses/pydantic for config, not dicts
- [ ] Config passed explicitly to functions
- [ ] No singletons without documented justification

### Rule 3: Documented Public APIs
- [ ] All public functions have docstring (NumPy style)
- [ ] All parameters documented with type hints
- [ ] Returns/Raises sections present
- [ ] Examples section with runnable code
- [ ] All imports resolvable in examples

### Rule 4: Tests + Docs
- [ ] Unit tests for all new code paths
- [ ] `pytest --cov` shows ≥80% coverage
- [ ] Documentation added to `docs/`
- [ ] If user-facing, add to API reference

### Rule 5: Metadata Validation
- [ ] Input validation at entry points (not deferred)
- [ ] Using pydantic models or dataclass validators
- [ ] Clear, actionable error messages

### Rule 6: Serializable Pipelines
- [ ] Configs use dataclasses/pydantic, not plain dict
- [ ] `.to_dict()` / `.from_dict()` methods
- [ ] Serialization roundtrip tests pass
- [ ] JSON/YAML compatible

### Rule 7: Actionable Errors
- [ ] Error messages include: what, why, how to fix
- [ ] Specific exception types (ValueError, TypeError, etc.)
- [ ] Helpful context and suggestions
- [ ] No bare `Exception` or `RuntimeError`
```

---

## FAQ

**Q: Can I use global state if it's read-only?**  
A: No. Even read-only globals create hidden dependencies. Use dependency injection instead.

**Q: What if my algorithm isn't deterministic?**  
A: All user-facing outputs must be deterministic via seeds. Internal optimizations may be non-deterministic if the final result is reproducible.

**Q: Is it okay to skip tests for "obvious" code?**  
A: No. Tests are requirements, not optional. Simple tests are still valuable (they catch regressions).

**Q: Can I use a different validation library instead of Pydantic?**  
A: Yes, as long as it validates at entry and raises immediately. Clear error messages required.

**Q: What's the minimum docstring?**  
A: Summary line (1 sentence) + Parameters + Returns + Example. Full sections required for public APIs.

**Q: Do I need to deprecate before breaking changes?**  
A: Yes (except for v0.x). See [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md).

---

## Related Documents

- [CONTRIBUTING.md](../../CONTRIBUTING.md) — Contributor guide
- [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md) — Backward compatibility strategy
- [JOSS_DOCS_AUDIT_REPORT.md](../../JOSS_DOCS_AUDIT_REPORT.md) — Documentation audit

---

**Last Updated**: 2026-01-24  
**Maintained by**: FoodSpec Core Team  
**Questions?** Open an issue or email chandrasekarnarayana@gmail.com
