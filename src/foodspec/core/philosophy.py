"""
FoodSpec Core Design Philosophy System

Principles enforced at execution time.
Every run must satisfy these principles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ============================================================================
# Strategic Goals & Non-Goals
# ============================================================================

GOALS: List[str] = [
    "Protocol-driven spectroscopy workflows",
    "Reproducibility by default",
    "Trust and uncertainty as first-class outputs",
    "QC is mandatory, not optional",
    "Designed for food matrices with complex backgrounds",
]

NON_GOALS: List[str] = [
    "Not a general deep learning framework",
    "Not a vendor replacement tool",
    "Not claiming clinical or regulatory approval",
]

FEATURE_MAP: List[Dict[str, str]] = [
    {
        "mindmap_node": "Data Objects",
        "module": "foodspec.data_objects",
        "public_api": "Spectrum, SpectraSet, SpectralDataset",
        "cli": "foodspec io validate",
        "artifacts": "protocol.yaml, run_summary.json",
    },
    {
        "mindmap_node": "Data Extraction",
        "module": "foodspec.io",
        "public_api": "read_spectra, detect_format",
        "cli": "foodspec io validate",
        "artifacts": "ingest logs",
    },
    {
        "mindmap_node": "Programming Engine",
        "module": "foodspec.engine",
        "public_api": "preprocessing pipeline",
        "cli": "foodspec preprocess run",
        "artifacts": "preprocess logs",
    },
    {
        "mindmap_node": "Quality Control",
        "module": "foodspec.qc",
        "public_api": "qc engine, dataset_qc",
        "cli": "foodspec qc spectral|dataset",
        "artifacts": "qc_results.json",
    },
    {
        "mindmap_node": "Feature Engineering",
        "module": "foodspec.features",
        "public_api": "peaks, ratios, chemometrics",
        "cli": "foodspec features extract",
        "artifacts": "features tables",
    },
    {
        "mindmap_node": "Modeling & Validation",
        "module": "foodspec.modeling",
        "public_api": "training and validation",
        "cli": "foodspec model run",
        "artifacts": "metrics.json",
    },
    {
        "mindmap_node": "Trust & Uncertainty",
        "module": "foodspec.trust",
        "public_api": "calibration, conformal",
        "cli": "foodspec trust calibrate|conformal",
        "artifacts": "uncertainty_metrics.json",
    },
    {
        "mindmap_node": "Visualization & Reporting",
        "module": "foodspec.viz, foodspec.reporting",
        "public_api": "figures, report builders",
        "cli": "foodspec report",
        "artifacts": "html/pdf reports",
    },
]


def feature_map() -> List[Dict[str, str]]:
    """Return the mindmap-to-module mapping table."""
    return list(FEATURE_MAP)


# ============================================================================
# Formal Design Principles (Enforced at Runtime)
# ============================================================================


class PhilosophyError(Exception):
    """Raised when a design principle is violated"""

    pass


class TaskType(str, Enum):
    """Canonical task types for FoodSpec"""

    AUTHENTICATION = "authentication"
    ADULTERATION = "adulteration"
    MONITORING = "monitoring"


@dataclass
class DesignPrinciples:
    """
    FoodSpec's formal design principles enforced at execution time.
    """

    # Core Principles
    TASK_FIRST: List[str] = None  # Task must be one of these
    QC_FIRST: bool = True  # QC must run before modeling
    TRUST_FIRST: bool = True  # Trust stack required outputs
    PROTOCOL_IS_SOURCE_OF_TRUTH: bool = True  # Protocol immutable
    REPRODUCIBILITY_REQUIRED: bool = True  # Seeds, versions, env captured
    DUAL_API: bool = True  # CLI and programmatic both
    REPORT_FIRST: bool = True  # Reports on every run

    def __post_init__(self):
        if self.TASK_FIRST is None:
            self.TASK_FIRST = [
                TaskType.AUTHENTICATION.value,
                TaskType.ADULTERATION.value,
                TaskType.MONITORING.value,
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Export principles to dict"""
        return {
            "task_first": self.TASK_FIRST,
            "qc_first": self.QC_FIRST,
            "trust_first": self.TRUST_FIRST,
            "protocol_is_source_of_truth": self.PROTOCOL_IS_SOURCE_OF_TRUTH,
            "reproducibility_required": self.REPRODUCIBILITY_REQUIRED,
            "dual_api": self.DUAL_API,
            "report_first": self.REPORT_FIRST,
        }


# Global singleton
DESIGN_PRINCIPLES = DesignPrinciples()


# ============================================================================
# Enforcement Functions
# ============================================================================


def enforce_task_first(config: Dict[str, Any]) -> None:
    """
    Enforce: Task must be one of TASK_FIRST types.

    Raises:
        PhilosophyError: If task not in TASK_FIRST
    """
    if not DESIGN_PRINCIPLES.TASK_FIRST:
        return

    task = config.get("task") if isinstance(config, dict) else getattr(config, "task", None)

    if task is None:
        raise PhilosophyError("Task not specified in protocol")

    if task not in DESIGN_PRINCIPLES.TASK_FIRST:
        raise PhilosophyError(
            f"Task '{task}' not in TASK_FIRST: {DESIGN_PRINCIPLES.TASK_FIRST}. "
            f"Use one of: {', '.join(DESIGN_PRINCIPLES.TASK_FIRST)}"
        )

    logger.info(f"✓ Task '{task}' is in TASK_FIRST principle")


def enforce_protocol_truth(protocol: Any) -> None:
    """Enforce: Protocol is immutable source of truth"""
    if not DESIGN_PRINCIPLES.PROTOCOL_IS_SOURCE_OF_TRUTH:
        return

    required = ["task", "modality", "model", "validation"]
    for attr in required:
        if not hasattr(protocol, attr):
            raise PhilosophyError(f"Protocol missing required attribute '{attr}' for source-of-truth")

    logger.info("✓ Protocol is complete source of truth")


def enforce_qc_first(qc_results: Dict[str, Any]) -> None:
    """Enforce: QC runs before modeling and blocks bad data"""
    if not DESIGN_PRINCIPLES.QC_FIRST:
        return

    if not qc_results:
        raise PhilosophyError("QC results not available - QC must run first")

    if "critical_failures" in qc_results and qc_results["critical_failures"]:
        raise PhilosophyError(f"QC_FIRST violated: Critical failures: {qc_results['critical_failures']}")

    status = qc_results.get("status", "unknown")
    if status not in ["pass", "warning"]:
        raise PhilosophyError(f"QC_FIRST violated: QC status is '{status}', must be 'pass' or 'warning'")

    pass_rate = qc_results.get("pass_rate", 0.0)
    if pass_rate < 0.5:
        raise PhilosophyError(f"QC_FIRST violated: Pass rate {pass_rate:.1%} below 50% threshold")

    logger.info(f"✓ QC passed (status={status}, pass_rate={pass_rate:.1%})")


def enforce_trust_first(trust_outputs: Dict[str, Any]) -> None:
    """Enforce: Trust stack outputs are generated"""
    if not DESIGN_PRINCIPLES.TRUST_FIRST:
        return

    if not trust_outputs:
        raise PhilosophyError("Trust outputs not available")

    required_keys = ["calibration", "conformal"]
    available_keys = [k for k in required_keys if k in trust_outputs]

    if len(available_keys) < len(required_keys):
        missing = set(required_keys) - set(available_keys)
        raise PhilosophyError(f"TRUST_FIRST violated: Missing {missing}. Available: {list(trust_outputs.keys())}")

    logger.info(f"✓ Trust stack outputs available: {list(trust_outputs.keys())}")


def enforce_reproducibility(manifest: Dict[str, Any]) -> None:
    """Enforce: Run is reproducible (seed, versions, environment)"""
    if not DESIGN_PRINCIPLES.REPRODUCIBILITY_REQUIRED:
        return

    if not manifest:
        raise PhilosophyError("Manifest not available")

    required = ["seed", "python_version", "package_versions", "os_info", "data_fingerprint"]
    missing = [f for f in required if f not in manifest or manifest[f] is None]

    if missing:
        raise PhilosophyError(
            f"REPRODUCIBILITY_REQUIRED violated: Missing {missing}. Available: {list(manifest.keys())}"
        )

    logger.info(f"✓ Reproducibility verified (seed={manifest.get('seed')}, python={manifest.get('python_version')})")


def enforce_report_first(artifacts: Dict[str, Path]) -> None:
    """Enforce: Reports are generated for every run"""
    if not DESIGN_PRINCIPLES.REPORT_FIRST:
        return

    if not artifacts:
        raise PhilosophyError("No artifacts produced")

    report_keys = ["report_html", "card_json", "card_markdown"]
    available = [k for k in report_keys if k in artifacts]

    if not available:
        raise PhilosophyError(
            f"REPORT_FIRST violated: No report artifacts. Expected: {report_keys}, Got: {list(artifacts.keys())}"
        )

    logger.info(f"✓ Reports generated: {available}")


def validate_all_principles(
    config: Dict[str, Any],
    protocol: Any,
    qc_results: Dict[str, Any],
    trust_outputs: Dict[str, Any],
    manifest: Dict[str, Any],
    artifacts: Dict[str, Path],
) -> None:
    """
    Run all philosophy enforcement checks.

    Raises:
        PhilosophyError: If any principle violated
    """
    logger.info("=" * 70)
    logger.info("PHILOSOPHY ENFORCEMENT: Validating all design principles")
    logger.info("=" * 70)

    errors = []

    try:
        enforce_task_first(config)
    except PhilosophyError as e:
        errors.append(f"Task-First: {e}")

    try:
        enforce_protocol_truth(protocol)
    except PhilosophyError as e:
        errors.append(f"Protocol-Truth: {e}")

    try:
        enforce_qc_first(qc_results)
    except PhilosophyError as e:
        errors.append(f"QC-First: {e}")

    try:
        enforce_trust_first(trust_outputs)
    except PhilosophyError as e:
        errors.append(f"Trust-First: {e}")

    try:
        enforce_reproducibility(manifest)
    except PhilosophyError as e:
        errors.append(f"Reproducibility: {e}")

    try:
        enforce_report_first(artifacts)
    except PhilosophyError as e:
        errors.append(f"Report-First: {e}")

    if errors:
        error_msg = "\n".join(f"  ✗ {e}" for e in errors)
        logger.error(f"\nPHILOSOPHY VIOLATIONS:\n{error_msg}")
        raise PhilosophyError(f"Philosophy enforcement failed ({len(errors)} violations):\n{error_msg}")

    logger.info("=" * 70)
    logger.info("✓ ALL PHILOSOPHY PRINCIPLES SATISFIED")
    logger.info("=" * 70)


def get_principles() -> DesignPrinciples:
    """Get global principles instance"""
    return DESIGN_PRINCIPLES
