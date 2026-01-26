from __future__ import annotations
"""
FoodSpec Experiment Cards system.

An Experiment Card is a concise summary of a model run with risk assessment,
confidence levels, and deployment readiness recommendations.

Examples
--------
Build and export an experiment card::

    context = ReportContext.load(Path("/run/dir"))
    card = build_experiment_card(context, mode=ReportMode.RESEARCH)
    card.to_json(Path("/run/dir/card.json"))
    card.to_markdown(Path("/run/dir/card.md"))
"""


import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from foodspec.reporting.base import ReportContext
from foodspec.reporting.modes import ReportMode
from foodspec.reporting.schema import RunBundle


class ConfidenceLevel(str, Enum):
    """Model confidence level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DeploymentReadiness(str, Enum):
    """Model deployment readiness status."""

    NOT_READY = "not_ready"
    PILOT = "pilot"
    READY = "ready"


@dataclass
class ExperimentCard:
    """Concise summary of a model run with risk assessment and deployment readiness.

    Attributes
    ----------
    run_id : str
        Unique run identifier.
    timestamp : str
        ISO 8601 timestamp of the run.
    task : str
        Task type (e.g., 'classification', 'regression').
    modality : str
        Data modality (e.g., 'raman', 'ftir').
    model : str
        Model name or type.
    validation_scheme : str
        Validation scheme (e.g., 'stratified_kfold', 'random').
    macro_f1 : Optional[float]
        Macro F1 score (for classification).
    auroc : Optional[float]
        Area under ROC curve.
    ece : Optional[float]
        Expected Calibration Error.
    coverage : Optional[float]
        Coverage (fraction of non-abstained predictions, 0-1).
    abstain_rate : Optional[float]
        Abstention rate (fraction of abstained predictions, 0-1).
    auto_summary : str
        Auto-generated short narrative (1-3 sentences).
    key_risks : List[str]
        List of identified risks.
    confidence_level : ConfidenceLevel
        Confidence level (LOW, MEDIUM, HIGH).
    confidence_reasoning : str
        Explanation for confidence level.
    deployment_readiness : DeploymentReadiness
        Deployment readiness (NOT_READY, PILOT, READY).
    readiness_reasoning : str
        Explanation for deployment readiness.
    metrics_dict : Dict[str, Any] = field(default_factory=dict)
        Full metrics dict for reference.
    """

    run_id: str
    timestamp: str
    task: str
    modality: str
    model: str
    validation_scheme: str
    macro_f1: Optional[float] = None
    auroc: Optional[float] = None
    ece: Optional[float] = None
    coverage: Optional[float] = None
    abstain_rate: Optional[float] = None
    mean_set_size: Optional[float] = None
    drift_score: Optional[float] = None
    qc_pass_rate: Optional[float] = None
    auto_summary: str = ""
    key_risks: List[str] = field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    confidence_reasoning: str = ""
    deployment_readiness: DeploymentReadiness = DeploymentReadiness.NOT_READY
    readiness_reasoning: str = ""
    regulatory_readiness_score: Optional[float] = None
    regulatory_readiness_notes: List[str] = field(default_factory=list)
    metrics_dict: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, out_path: Path) -> Path:
        """Export card to JSON.

        Parameters
        ----------
        out_path : Path
            Path where JSON will be written.

        Returns
        -------
        Path
            Path to written file.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert enums to strings for JSON
        data = asdict(self)
        data["confidence_level"] = self.confidence_level.value
        data["deployment_readiness"] = self.deployment_readiness.value
        out_path.write_text(json.dumps(data, indent=2))
        return out_path

    def to_markdown(self, out_path: Path) -> Path:
        """Export card to Markdown.

        Parameters
        ----------
        out_path : Path
            Path where Markdown will be written.

        Returns
        -------
        Path
            Path to written file.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Experiment Card",
            "",
            f"**Run ID**: {self.run_id}  ",
            f"**Timestamp**: {self.timestamp}  ",
            f"**Task**: {self.task} | **Modality**: {self.modality}  ",
            "",
            "## Model",
            "",
            f"- **Type**: {self.model}",
            f"- **Validation**: {self.validation_scheme}",
            "",
            "## Performance Metrics",
            "",
        ]

        metrics_items = []
        if self.macro_f1 is not None:
            metrics_items.append(f"- **Macro F1**: {self.macro_f1:.3f}")
        if self.auroc is not None:
            metrics_items.append(f"- **AUROC**: {self.auroc:.3f}")
        if self.ece is not None:
            metrics_items.append(f"- **ECE**: {self.ece:.4f}")
        if self.coverage is not None:
            metrics_items.append(f"- **Coverage**: {self.coverage:.1%}")
        if self.abstain_rate is not None:
            metrics_items.append(f"- **Abstention Rate**: {self.abstain_rate:.1%}")
        if self.mean_set_size is not None:
            metrics_items.append(f"- **Mean Set Size**: {self.mean_set_size:.2f}")
        if self.drift_score is not None:
            metrics_items.append(f"- **Drift Score**: {self.drift_score:.3f}")
        if self.qc_pass_rate is not None:
            metrics_items.append(f"- **QC Pass Rate**: {self.qc_pass_rate:.1%}")

        if metrics_items:
            lines.extend(metrics_items)
        else:
            lines.append("*No metrics available*")

        lines.extend([
            "",
            "## Summary",
            "",
            self.auto_summary,
            "",
            "## Risk Assessment",
            "",
        ])

        if self.key_risks:
            for risk in self.key_risks:
                lines.append(f"- {risk}")
        else:
            lines.append("*No significant risks identified*")

        lines.extend([
            "",
            "## Confidence & Readiness",
            "",
            f"**Confidence Level**: {self.confidence_level.value.upper()}  ",
            f"*{self.confidence_reasoning}*",
            "",
            f"**Deployment Readiness**: {self.deployment_readiness.value.upper()}  ",
            f"*{self.readiness_reasoning}*",
            "",
        ])
        if self.regulatory_readiness_score is not None:
            lines.append(f"**Regulatory Readiness Score**: {self.regulatory_readiness_score:.1f}/100  ")
            if self.regulatory_readiness_notes:
                lines.append("**Readiness Notes**:")
                for note in self.regulatory_readiness_notes:
                    lines.append(f"- {note}")
                lines.append("")

        out_path.write_text("\n".join(lines))
        return out_path


def _extract_metrics(context: ReportContext) -> Dict[str, float]:
    """Extract headline metrics from context.

    Parameters
    ----------
    context : ReportContext
        Report context with loaded artifacts.

    Returns
    -------
    dict
        Dict with macro_f1, auroc, ece, coverage, abstain_rate, mean_set_size,
        drift_score, qc_pass_rate keys. Missing metrics are None.
    """
    metrics = {
        "macro_f1": None,
        "auroc": None,
        "ece": None,
        "coverage": None,
        "abstain_rate": None,
        "mean_set_size": None,
        "drift_score": None,
        "qc_pass_rate": None,
    }

    # Extract from metrics table (first row or average)
    if context.metrics:
        first_row = context.metrics[0]
        metrics["macro_f1"] = _get_float(first_row, ["macro_f1", "macroF1", "f1"])
        metrics["auroc"] = _get_float(first_row, ["auroc", "auc", "roc_auc"])
        metrics["ece"] = _get_float(first_row, ["ece", "expected_calibration_error"])

    # Extract from trust outputs (uncertainty/abstention)
    if context.trust_outputs:
        trust = context.trust_outputs
        metrics["ece"] = metrics["ece"] or _get_float(trust, ["ece", "expected_calibration_error"])
        calibration = trust.get("calibration") if isinstance(trust, dict) else {}
        if isinstance(calibration, dict):
            metrics_after = calibration.get("metrics_after", {})
            metrics["ece"] = metrics["ece"] or _get_float(metrics_after, ["ece"])

        metrics["coverage"] = _get_float(trust, ["coverage", "coverage_rate"])
        conformal = trust.get("conformal") if isinstance(trust, dict) else {}
        if isinstance(conformal, dict):
            metrics["coverage"] = metrics["coverage"] or _get_float(conformal, ["coverage", "coverage_rate"])
            metrics["mean_set_size"] = _get_float(conformal, ["mean_set_size", "avg_set_size"])

        metrics["abstain_rate"] = _get_float(trust, ["abstain_rate", "abstention_rate"])
        abstention = trust.get("abstention") if isinstance(trust, dict) else {}
        if isinstance(abstention, dict):
            metrics["abstain_rate"] = metrics["abstain_rate"] or _get_float(
                abstention, ["abstain_rate", "abstention_rate"]
            )
        
        # Extract drift score
        drift = trust.get("drift") if isinstance(trust, dict) else {}
        if isinstance(drift, dict):
            # Try batch drift first
            batch_drift = drift.get("batch_drift", {})
            if isinstance(batch_drift, dict):
                metrics["drift_score"] = _get_float(batch_drift, ["mean_drift", "avg_drift", "drift_score"])
            # Fall back to temporal drift
            if metrics["drift_score"] is None:
                temporal_drift = drift.get("temporal_drift", {})
                if isinstance(temporal_drift, dict):
                    metrics["drift_score"] = _get_float(temporal_drift, ["mean_drift", "avg_drift", "drift_score"])
        
        # Extract QC pass rate
        qc_summary = trust.get("qc_summary") if isinstance(trust, dict) else {}
        if isinstance(qc_summary, dict):
            metrics["qc_pass_rate"] = _get_float(qc_summary, ["pass_rate", "qc_pass_rate"])

    # Also check QC from context.qc
    if context.qc and not metrics["qc_pass_rate"]:
        # Count pass/fail in QC table
        total = len(context.qc)
        passed = sum(1 for row in context.qc if row.get("status") == "pass")
        if total > 0:
            metrics["qc_pass_rate"] = passed / total

    return metrics


def _extract_readiness(context: ReportContext) -> tuple[Optional[float], List[str]]:
    """Extract regulatory readiness score and notes from trust outputs."""
    if not isinstance(context.trust_outputs, dict):
        return None, []
    readiness = context.trust_outputs.get("readiness")
    if not isinstance(readiness, dict):
        return None, []
    score = None
    try:
        score = float(readiness.get("score")) if readiness.get("score") is not None else None
    except (TypeError, ValueError):
        score = None
    notes = [str(n) for n in readiness.get("notes", [])] if readiness.get("notes") else []
    return score, notes


def _get_float(data: Any, keys: List[str]) -> Optional[float]:
    """Try to extract float from dict using list of possible keys.

    Parameters
    ----------
    data : Any
        Dict or dict-like object.
    keys : list of str
        Possible key names.

    Returns
    -------
    float or None
        First found value, or None.
    """
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except (ValueError, TypeError):
                pass
    return None


def _assess_confidence(
    metrics: Dict[str, float],
    context: ReportContext,
    mode: ReportMode,
) -> tuple[ConfidenceLevel, str]:
    """Assess confidence level based on metrics and metadata.

    Rules for LOW confidence:
    - ECE > 0.1 (poor calibration)
    - Coverage < 0.9 (high abstention)
    - Abstain rate > 0.1 (high abstention)
    - Missing macro_f1 or auroc
    - Random CV validation scheme
    - Missing protocol hash or data fingerprint (regulatory mode)

    Parameters
    ----------
    metrics : dict
        Extracted metrics.
    context : ReportContext
        Report context.
    mode : ReportMode
        Reporting mode.

    Returns
    -------
    tuple of (ConfidenceLevel, str)
        Confidence level and reasoning.
    """
    risks = []

    # Check ECE
    if metrics.get("ece") is not None and metrics["ece"] > 0.1:
        risks.append("high ECE (>0.1)")

    # Check coverage
    if metrics.get("coverage") is not None and metrics["coverage"] < 0.9:
        risks.append(f"low coverage (<90%: {metrics['coverage']:.1%})")

    # Check abstention
    if metrics.get("abstain_rate") is not None and metrics["abstain_rate"] > 0.1:
        risks.append(f"high abstention (>10%: {metrics['abstain_rate']:.1%})")

    # Check required metrics
    if metrics.get("macro_f1") is None or metrics.get("auroc") is None:
        risks.append("missing headline metrics")

    # Check validation scheme
    if "random" in str(context.manifest.protocol_snapshot.get("validation", {})).lower():
        risks.append("random CV (non-stratified)")

    # Regulatory mode: check manifest hashes
    if mode == ReportMode.REGULATORY:
        if not context.manifest.protocol_hash:
            risks.append("missing protocol hash (regulatory mode)")
        if not context.manifest.data_fingerprint:
            risks.append("missing data fingerprint (regulatory mode)")

    # Determine confidence level
    if len(risks) >= 3:
        level = ConfidenceLevel.LOW
        reasoning = f"Multiple concerns: {'; '.join(risks[:3])}"
    elif len(risks) == 2:
        level = ConfidenceLevel.MEDIUM
        reasoning = f"Some concerns: {'; '.join(risks)}"
    elif len(risks) == 1:
        level = ConfidenceLevel.MEDIUM
        reasoning = f"Minor concern: {risks[0]}"
    else:
        level = ConfidenceLevel.HIGH
        reasoning = "No significant concerns identified"

    return level, reasoning


def _assess_deployment_readiness(
    confidence_level: ConfidenceLevel,
    context: ReportContext,
    mode: ReportMode,
) -> tuple[DeploymentReadiness, str]:
    """Assess deployment readiness based on confidence and mode.

    Rules:
    - NOT_READY: confidence is LOW, or regulatory mode with missing hashes
    - PILOT: confidence is MEDIUM
    - READY: confidence is HIGH and all mode requirements met

    Parameters
    ----------
    confidence_level : ConfidenceLevel
        Assessed confidence level.
    context : ReportContext
        Report context.
    mode : ReportMode
        Reporting mode.

    Returns
    -------
    tuple of (DeploymentReadiness, str)
        Readiness status and reasoning.
    """
    reasons = []

    # Check regulatory requirements
    if mode == ReportMode.REGULATORY:
        if not context.manifest.protocol_hash or not context.manifest.data_fingerprint:
            return (
                DeploymentReadiness.NOT_READY,
                "Regulatory mode requires manifest hashes for reproducibility",
            )

    # Map confidence to readiness
    if confidence_level == ConfidenceLevel.LOW:
        return (
            DeploymentReadiness.NOT_READY,
            "Low confidence model is not ready for deployment",
        )
    elif confidence_level == ConfidenceLevel.MEDIUM:
        return (
            DeploymentReadiness.PILOT,
            "Medium confidence model suitable for pilot deployment with monitoring",
        )
    else:  # HIGH
        return (
            DeploymentReadiness.READY,
            "High confidence model is ready for deployment",
        )


def _generate_auto_summary(
    context: ReportContext,
    metrics: Dict[str, float],
    confidence_level: ConfidenceLevel,
) -> str:
    """Generate short narrative summary.

    Parameters
    ----------
    context : ReportContext
        Report context.
    metrics : dict
        Extracted metrics.
    confidence_level : ConfidenceLevel
        Assessed confidence level.

    Returns
    -------
    str
        1-3 sentence summary.
    """
    parts = []

    # Sentence 1: Model and task
    protocol = context.manifest.protocol_snapshot
    task = protocol.get("task", {}).get("name", "unknown task")
    parts.append(f"Trained {context.manifest.seed and 'deterministic' or 'random'} model for {task}.")

    # Sentence 2: Performance
    if metrics.get("macro_f1") is not None:
        f1 = metrics["macro_f1"]
        auroc_str = f", AUROC {metrics['auroc']:.3f}" if metrics.get("auroc") else ""
        parts.append(f"Achieved macro F1 {f1:.3f}{auroc_str}.")

    # Sentence 3: Readiness
    if confidence_level == ConfidenceLevel.HIGH:
        parts.append("Ready for deployment.")
    elif confidence_level == ConfidenceLevel.MEDIUM:
        parts.append("Suitable for pilot deployment with monitoring.")
    else:
        parts.append("Needs further investigation before deployment.")

    return " ".join(parts)


def _identify_risks(
    metrics: Dict[str, float],
    context: ReportContext,
) -> List[str]:
    """Identify key risks.

    Parameters
    ----------
    metrics : dict
        Extracted metrics.
    context : ReportContext
        Report context.

    Returns
    -------
    list of str
        List of identified risks.
    """
    risks = []

    if metrics.get("ece") is not None:
        if metrics["ece"] > 0.15:
            risks.append(f"High miscalibration: ECE = {metrics['ece']:.4f}")
        elif metrics["ece"] > 0.1:
            risks.append(f"Moderate miscalibration: ECE = {metrics['ece']:.4f}")

    if metrics.get("coverage") is not None:
        if metrics["coverage"] < 0.8:
            risks.append(f"Very low coverage: {metrics['coverage']:.1%} predictions made")

    if metrics.get("abstain_rate") is not None:
        if metrics["abstain_rate"] > 0.2:
            risks.append(f"High abstention rate: {metrics['abstain_rate']:.1%} of predictions abstained")

    if metrics.get("macro_f1") is None or metrics.get("auroc") is None:
        risks.append("Missing required metrics (macro_f1 or auroc)")

    if len(context.qc) == 0:
        risks.append("No QC data available for validation")

    # Multivariate QC
    mv_qc = {}
    if isinstance(context.trust_outputs, dict):
        mv_qc = (context.trust_outputs.get("qc_summary") or {}).get("multivariate", {})
    outlier_info = mv_qc.get("outliers") if isinstance(mv_qc, dict) else None
    if outlier_info and isinstance(outlier_info, dict):
        n_flagged = outlier_info.get("n_flagged")
        if n_flagged is not None and n_flagged > 0:
            risks.append(f"Multivariate outliers flagged: {int(n_flagged)} samples")
    drift_info = mv_qc.get("drift") if isinstance(mv_qc, dict) else None
    if drift_info and isinstance(drift_info, list):
        if any(d.get("status") == "fail" for d in drift_info if isinstance(d, dict)):
            risks.append("Batch drift detected in latent space")

    return risks


def build_experiment_card(
    context: ReportContext,
    mode: ReportMode = ReportMode.RESEARCH,
    run_id: Optional[str] = None,
) -> ExperimentCard:
    """Build an experiment card from a report context.

    Parameters
    ----------
    context : ReportContext
        Loaded report context.
    mode : ReportMode, optional
        Reporting mode for mode-specific validation rules.
    run_id : str, optional
        Override run ID. Defaults to protocol hash[:8].

    Returns
    -------
    ExperimentCard
        Constructed experiment card.
    """
    # Extract basic metadata
    manifest = context.manifest
    protocol = manifest.protocol_snapshot
    _run_id = run_id or manifest.protocol_hash[:8]
    _timestamp = manifest.start_time
    _task = protocol.get("task", {}).get("name", "unknown")
    _modality = protocol.get("modality", "unknown")
    _model = protocol.get("model", {}).get("name", "unknown")
    _validation = protocol.get("validation", {}).get("scheme", "unknown")

    # Extract metrics
    metrics = _extract_metrics(context)

    # Assess confidence
    confidence_level, confidence_reasoning = _assess_confidence(metrics, context, mode)

    # Assess deployment readiness
    deployment_readiness, readiness_reasoning = _assess_deployment_readiness(
        confidence_level, context, mode
    )
    readiness_score, readiness_notes = _extract_readiness(context)

    # Generate summary
    auto_summary = _generate_auto_summary(context, metrics, confidence_level)

    # Identify risks
    key_risks = _identify_risks(metrics, context)

    return ExperimentCard(
        run_id=_run_id,
        timestamp=_timestamp,
        task=_task,
        modality=_modality,
        model=_model,
        validation_scheme=_validation,
        macro_f1=metrics["macro_f1"],
        auroc=metrics["auroc"],
        ece=metrics["ece"],
        coverage=metrics["coverage"],
        abstain_rate=metrics["abstain_rate"],
        mean_set_size=metrics["mean_set_size"],
        drift_score=metrics["drift_score"],
        qc_pass_rate=metrics["qc_pass_rate"],
        auto_summary=auto_summary,
        key_risks=key_risks,
        confidence_level=confidence_level,
        confidence_reasoning=confidence_reasoning,
        deployment_readiness=deployment_readiness,
        readiness_reasoning=readiness_reasoning,
        regulatory_readiness_score=readiness_score,
        regulatory_readiness_notes=readiness_notes,
        metrics_dict=metrics,
    )


def build_experiment_card_from_bundle(
    bundle: RunBundle,
    mode: ReportMode = ReportMode.RESEARCH,
) -> ExperimentCard:
    """Build an ExperimentCard from a RunBundle."""
    context = _bundle_to_context(bundle)
    metrics = _extract_metrics(context)
    confidence_level, confidence_reasoning = _assess_confidence(metrics, context, mode)
    deployment_readiness, readiness_reasoning = _assess_deployment_readiness(
        confidence_level,
        context,
        mode,
    )
    readiness_score, readiness_notes = _extract_readiness(context)
    auto_summary = _generate_auto_summary(context, metrics, confidence_level)
    key_risks = _identify_risks(metrics, context)

    manifest = context.manifest
    protocol = manifest.protocol_snapshot
    return ExperimentCard(
        run_id=bundle.run_id,
        timestamp=str(bundle.manifest.get("timestamp") or ""),
        task=protocol.get("task", {}).get("name", "unknown"),
        modality=protocol.get("modality", "unknown"),
        model=protocol.get("model", {}).get("name", "unknown"),
        validation_scheme=protocol.get("validation", {}).get("scheme", "unknown"),
        macro_f1=metrics["macro_f1"],
        auroc=metrics["auroc"],
        ece=metrics["ece"],
        coverage=metrics["coverage"],
        abstain_rate=metrics["abstain_rate"],
        mean_set_size=metrics["mean_set_size"],
        drift_score=metrics["drift_score"],
        qc_pass_rate=metrics["qc_pass_rate"],
        auto_summary=auto_summary,
        key_risks=key_risks,
        confidence_level=confidence_level,
        confidence_reasoning=confidence_reasoning,
        deployment_readiness=deployment_readiness,
        readiness_reasoning=readiness_reasoning,
        regulatory_readiness_score=readiness_score,
        regulatory_readiness_notes=readiness_notes,
        metrics_dict=metrics,
    )


def _bundle_to_context(bundle: RunBundle) -> ReportContext:
    """Convert RunBundle to ReportContext for existing heuristics."""
    from foodspec.core.manifest import RunManifest

    manifest_path = bundle.run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = RunManifest.load(manifest_path)
        except Exception:
            manifest = RunManifest.build(
                protocol_snapshot=bundle.manifest.get("protocol_snapshot", {}),
                data_path=None,
                seed=bundle.seed or 0,
                artifacts=bundle.artifacts,
            )
    else:
        manifest = RunManifest.build(
            protocol_snapshot=bundle.manifest.get("protocol_snapshot", {}),
            data_path=None,
            seed=bundle.seed or 0,
            artifacts=bundle.artifacts,
        )
    return ReportContext(
        run_dir=bundle.run_dir,
        manifest=manifest,
        protocol_snapshot=bundle.manifest.get("protocol_snapshot", {}),
        metrics=bundle.metrics,
        predictions=bundle.predictions,
        qc=[bundle.qc_report] if bundle.qc_report else [],
        trust_outputs=bundle.trust_outputs,
        figures={},
    )
