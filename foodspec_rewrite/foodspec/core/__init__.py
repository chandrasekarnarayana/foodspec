"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
Core protocols and base classes for FoodSpec.

Protocol Definitions:
  Spectrum          - Single spectral measurement with metadata
  SpectralDataset   - Collection of spectra with shared metadata
  Preprocessor      - Pipeline step for data transformation
  FeatureExtractor  - Feature computation from spectra
  Model             - Trainable/predictive model
  QCCheck          - Quality control validator

Registry & Composition:
  Registry         - Extensible component registry
  Orchestrator     - Workflow composition and execution
  Manifest         - Reproducibility metadata
  ArtifactBundle   - Output collection and serialization
  Cache            - Performance layer
"""

from typing import Protocol, Any, Dict, List
import dataclasses
from abc import ABC, abstractmethod

from .protocol import ProtocolV2
from .registry import ComponentRegistry
from .artifacts import ArtifactRegistry
from .manifest import RunManifest
from .orchestrator import ExecutionEngine, RunResult
from .pipeline import Pipeline, PipelineNode


@dataclasses.dataclass
class SpectrumMetadata:
    """Metadata for a single spectrum."""
    sample_id: str
    modality: str  # 'raman', 'ftir', 'nir'
    date: str
    operator: str = ""
    notes: str = ""
    custom: Dict[str, Any] = dataclasses.field(default_factory=dict)


class Spectrum(Protocol):
    """Protocol for a single spectral measurement.
    
    A spectrum is an immutable (wavenumbers, intensities) pair with metadata.
    """
    
    @property
    def wavenumbers(self) -> list[float]:
        """X-axis: wavenumber/wavelength in cm⁻¹."""
        ...
    
    @property
    def intensities(self) -> list[float]:
        """Y-axis: intensity/counts."""
        ...
    
    @property
    def metadata(self) -> SpectrumMetadata:
        """Associated metadata."""
        ...


class SpectralDataset(Protocol):
    """Protocol for a collection of aligned spectra.
    
    All spectra share same wavenumber axis, can have different metadata.
    """
    
    @property
    def wavenumbers(self) -> list[float]:
        """Shared X-axis."""
        ...
    
    @property
    def spectra(self) -> list[list[float]]:
        """(n_samples, n_wavenumbers) intensity matrix."""
        ...
    
    @property
    def metadata(self) -> list[SpectrumMetadata]:
        """Per-spectrum metadata."""
        ...


class Preprocessor(ABC):
    """Base class for pipeline steps.
    
    Each preprocessor:
    - Accepts dataset, returns transformed dataset
    - Is chainable (can compose multiple steps)
    - Logs transformation parameters
    """
    
    @abstractmethod
    def __call__(self, dataset: SpectralDataset) -> SpectralDataset:
        """Apply transformation."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Preprocessor":
        """Deserialize configuration."""
        pass


class FeatureExtractor(ABC):
    """Extract features from spectra."""
    
    @abstractmethod
    def extract(self, dataset: SpectralDataset) -> Dict[str, list]:
        """Extract features, return {feature_name: [values]}."""
        pass


class Model(ABC):
    """Trainable/predictive model."""
    
    @abstractmethod
    def fit(self, X: list[list[float]], y: list):
        """Train model."""
        pass
    
    @abstractmethod
    def predict(self, X: list[list[float]]) -> list:
        """Make predictions."""
        pass


class QCCheck(ABC):
    """Quality control validator."""
    
    @abstractmethod
    def validate(self, dataset: SpectralDataset) -> Dict[str, Any]:
        """Run QC checks, return {check_name: status}."""
        pass


# Core components (to be implemented)
class Registry:
    """Component registry for extensibility."""
    pass


class Orchestrator:
    """Workflow composition and execution."""
    pass


class Manifest:
    """Reproducibility metadata."""
    pass


class ArtifactBundle:
    """Output collection and serialization."""
    pass


class Cache:
    """Performance layer."""
    pass


__all__ = [
    "ProtocolV2",
    "ComponentRegistry",
    "ArtifactRegistry",
    "RunManifest",
    "ExecutionEngine",
    "RunResult",
    "Pipeline",
    "PipelineNode",
    "SpectrumMetadata",
    "Spectrum",
    "SpectralDataset",
    "Preprocessor",
    "FeatureExtractor",
    "Model",
    "QCCheck",
    "Registry",
    "Orchestrator",
    "Manifest",
    "ArtifactBundle",
    "Cache",
]
