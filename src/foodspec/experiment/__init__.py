"""Orchestration layer for end-to-end FoodSpec runs.

Provides:
  - Experiment: Main orchestration class
  - RunResult: Result bundle
  - RunMode: research/regulatory/monitoring
  - ValidationScheme: lobo/loso/nested
  - ExperimentConfig: Configuration dataclass
"""

from foodspec.experiment.experiment import (
    BatchRunResult,
    Experiment,
    ExperimentConfig,
    RunMode,
    RunResult,
    ValidationScheme,
)

__all__ = [
    "BatchRunResult",
    "Experiment",
    "ExperimentConfig",
    "RunMode",
    "RunResult",
    "ValidationScheme",
]
