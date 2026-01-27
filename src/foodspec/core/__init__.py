"""Core FoodSpec data models and entry point."""

from foodspec.data_objects.spectra_set import FoodSpectrumSet
from foodspec.data_objects.spectrum import Spectrum

from .api import FoodSpec
from .artifacts import ArtifactRegistry
from .hyperspectral import HyperSpectralCube
from .manifest import RunManifest
from .multimodal import MultiModalDataset
from .orchestrator import ExecutionEngine
from .output_bundle import OutputBundle

# Philosophy and new manifest system
from .philosophy import (
    DESIGN_PRINCIPLES,
    DesignPrinciples,
    PhilosophyError,
    TaskType,
    get_principles,
    validate_all_principles,
)
from .protocol import ProtocolV2
from .registry import ComponentRegistry
from .run_manifest import (
    DataSnapshot,
    EnvironmentSnapshot,
    ManifestBuilder,
    ProtocolSnapshot,
    RunStatus,
)
from .run_manifest import (
    RunManifest as NewRunManifest,
)
from .run_record import RunRecord
from .time import TimeSpectrumSet

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
