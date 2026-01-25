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
QC module: Quality control checks, validation, reporting.

Running QC checks on spectral data:
    from foodspec.qc import QCMetric, QCSummary
    summary = QCSummary({"snr": {"min": 3.0}})
    result = summary.evaluate(metrics_df)
"""
from foodspec.qc.base import QCMetric, QCSummary
from foodspec.qc.dataset import DatasetQC
from foodspec.qc.policy import Policy, apply_qc_policy
from foodspec.qc.spectral import SpectralQC

__all__ = [
    "QCMetric",
    "QCSummary",
    "SpectralQC",
    "DatasetQC",
    "Policy",
    "apply_qc_policy",
]
