"""Reporting schema objects for run artifact bundles."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _select_artifact_path(
    artifacts: Dict[str, Any],
    key: str,
    candidates: Sequence[Path],
) -> Path:
    artifact_path = artifacts.get(key)
    if artifact_path:
        return Path(artifact_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_trust_outputs(run_dir: Path, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    trust_payload: Dict[str, Any] = {}
    trust_path = _select_artifact_path(
        artifacts,
        "trust_outputs",
        [run_dir / "trust_outputs.json"],
    )
    if trust_path.exists():
        trust_payload.update(_load_json(trust_path))

    trust_dir = run_dir / "trust"
    if trust_dir.exists():
        for name in ("calibration", "conformal", "abstention", "coverage", "reliability", "readiness"):
            path = trust_dir / f"{name}.json"
            if path.exists():
                data = _load_json(path)
                if data:
                    trust_payload[name] = data

    drift_dir = run_dir / "drift"
    if drift_dir.exists():
        drift_data: Dict[str, Any] = {}
        for name in ("batch_drift", "temporal_drift", "stage_differences", "replicate_similarity"):
            path = drift_dir / f"{name}.json"
            if path.exists():
                data = _load_json(path)
                if data:
                    drift_data[name] = data
        if drift_data:
            trust_payload["drift"] = drift_data

    qc_dir = run_dir / "qc"
    qc_summary_path = qc_dir / "qc_summary.json"
    if qc_summary_path.exists():
        qc_data = _load_json(qc_summary_path)
        if qc_data:
            trust_payload["qc_summary"] = qc_data
    qc_control_path = qc_dir / "control_charts.json"
    if qc_control_path.exists():
        qc_data = _load_json(qc_control_path)
        if qc_data:
            trust_payload["qc_control_charts"] = qc_data

    return trust_payload


def _collect_figures(run_dir: Path) -> List[Path]:
    image_extensions = {".png", ".svg", ".pdf", ".jpg", ".jpeg"}
    scan_dirs = [
        run_dir / "figures",
        run_dir / "plots" / "viz",
        run_dir / "trust" / "plots",
        run_dir / "drift" / "plots",
        run_dir / "qc" / "plots",
        run_dir / "plots",
    ]
    results: List[Path] = []
    for base_dir in scan_dirs:
        if not base_dir.exists():
            continue
        for img_path in base_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                results.append(img_path)
    return sorted(set(results))


@dataclass
class RunBundle:
    """Bundle of run artifacts for reporting.

    RunBundle loads a run directory and exposes normalized artifacts for
    reporting pipelines. It is intentionally tolerant of missing artifacts
    (fields default to empty collections).
    """

    run_dir: Path
    manifest: Dict[str, Any]
    run_summary: Dict[str, Any]
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    qc_report: Dict[str, Any] = field(default_factory=dict)
    trust_outputs: Dict[str, Any] = field(default_factory=dict)
    figures: List[Path] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_artifacts(self) -> List[str]:
        """List available artifact categories for validation and reporting."""
        available: List[str] = ["manifest"]
        if self.manifest.get("protocol_snapshot"):
            available.append("protocol_snapshot")
        if self.manifest.get("data_fingerprint"):
            available.append("data_fingerprint")
        if self.metrics:
            available.append("metrics")
        if self.predictions:
            available.append("predictions")
        if self.qc_report:
            available.append("qc")
        if self.trust_outputs:
            available.append("trust_outputs")
        if self.figures:
            available.append("figures")
        for key in self.artifacts.keys():
            available.append(str(key))
        return sorted(set(available))

    @property
    def seed(self) -> Optional[int]:
        seed = self.run_summary.get("seed") or self.manifest.get("seed")
        try:
            return int(seed) if seed is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id") or self.run_summary.get("run_id") or self.run_dir.name)

    @classmethod
    def from_run_dir(cls, run_dir: Path | str) -> "RunBundle":
        """Load a RunBundle from a run directory.

        Parameters
        ----------
        run_dir : Path | str
            Directory containing manifest.json and run_summary.json.
        """
        run_dir = Path(run_dir)
        manifest = _load_json(run_dir / "manifest.json")
        run_summary = _load_json(run_dir / "run_summary.json")

        artifacts = run_summary.get("artifacts", {}) if isinstance(run_summary.get("artifacts"), dict) else {}
        metrics_path = _select_artifact_path(
            artifacts,
            "metrics",
            [run_dir / "tables" / "metrics.csv", run_dir / "metrics.csv"],
        )
        predictions_path = _select_artifact_path(
            artifacts,
            "predictions",
            [run_dir / "tables" / "predictions.csv", run_dir / "predictions.csv"],
        )
        qc_path = _select_artifact_path(
            artifacts,
            "qc_report",
            [run_dir / "qc_report.json", run_dir / "qc" / "qc_summary.json"],
        )

        metrics = _load_csv(metrics_path)
        predictions = _load_csv(predictions_path)
        qc_report = _load_json(qc_path)
        trust_outputs = _load_trust_outputs(run_dir, artifacts)
        figures = _collect_figures(run_dir)

        return cls(
            run_dir=run_dir,
            manifest=manifest,
            run_summary=run_summary,
            metrics=metrics,
            predictions=predictions,
            qc_report=qc_report,
            trust_outputs=trust_outputs,
            figures=figures,
            artifacts=artifacts,
        )


__all__ = ["RunBundle"]
