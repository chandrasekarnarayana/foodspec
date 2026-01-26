"""Core FoodSpec data models and entry point."""

from .api import FoodSpec
from .artifacts import ArtifactRegistry
from foodspec.data_objects.spectra_set import FoodSpectrumSet
from .hyperspectral import HyperSpectralCube
from .manifest import RunManifest
from .multimodal import MultiModalDataset
from .orchestrator import ExecutionEngine
from .output_bundle import OutputBundle
from .protocol import ProtocolV2
from .registry import ComponentRegistry
from .run_record import RunRecord
from foodspec.data_objects.spectrum import Spectrum
from .time import TimeSpectrumSet

# Philosophy and new manifest system
from .philosophy import (
    DESIGN_PRINCIPLES,
    PhilosophyError,
    TaskType,
    DesignPrinciples,
    validate_all_principles,
    get_principles,
)
from .run_manifest import (
    RunManifest as NewRunManifest,
    RunStatus,
    ManifestBuilder,
    ProtocolSnapshot,
    DataSnapshot,
    EnvironmentSnapshot,
)

__all__ = [
    # Existing
    "ArtifactRegistry",
    "ComponentRegistry",
    "ExecutionEngine",
    "FoodSpec",
    "FoodSpectrumSet",
    "HyperSpectralCube",
    "MultiModalDataset",
    "OutputBundle",
    "ProtocolV2",
    "RunManifest",
    "RunRecord",
    "Spectrum",
    "TimeSpectrumSet",
    # Philosophy and execution
    "DESIGN_PRINCIPLES",
    "PhilosophyError",
    "TaskType",
    "DesignPrinciples",
    "validate_all_principles",
    "get_principles",
    "NewRunManifest",
    "RunStatus",
    "ManifestBuilder",
    "ProtocolSnapshot",
    "DataSnapshot",
    "EnvironmentSnapshot",
]
