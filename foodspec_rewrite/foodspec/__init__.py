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
FoodSpec: Clean architecture rewrite - Protocol-driven spectroscopy framework.

This is the new clean architecture with:
- Protocol-based design (duck typing, Liskov substitution)
- Registry pattern for extensibility
- Orchestrator for workflow composition
- Artifact-based reproducibility
- Cache layer for performance

Modules:
  core/       - Protocol definitions, registry, orchestrator, manifest, artifacts, cache
  io/         - Data loading, format detection, library management
  preprocess/ - Baseline correction, harmonization, normalization
  qc/         - Quality control, outlier detection, balance checks
  features/   - Feature extraction (spectral, statistical, domain-specific)
  models/     - ML models, training, prediction, validation
  validation/ - Cross-validation, train/test split, stratification
  trust/      - Uncertainty quantification, confidence intervals
  viz/        - Plotting, interactive visualization, report generation
  reporting/  - Report templates, export formats
  deploy/     - Model serving, batch prediction, API
  cli/        - Command-line interface, plugins
"""

__version__ = "2.0.0-alpha"
__author__ = "FoodSpec Contributors"
__license__ = "MIT"

# Clean imports from core (aligned with v2 rewrite module names)
from .core.registry import ComponentRegistry as Registry
from .core.orchestrator import ExecutionEngine as Orchestrator
from .core.manifest import RunManifest as Manifest
from .core.artifacts import ArtifactRegistry as ArtifactBundle

__all__ = [
    "Registry",
    "Orchestrator",
    "Manifest",
    "ArtifactBundle",
]
