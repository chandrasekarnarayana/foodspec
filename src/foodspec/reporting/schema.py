"""Reporting schema objects for run artifact bundles."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _collect_figures(figures_dir: Path) -> List[Path]:
    if not figures_dir.exists():
        return []
    results: List[Path] = []
    for ext in ("*.png", "*.svg", "*.pdf"):
        results.extend(sorted(figures_dir.rglob(ext)))
    return results


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
        metrics_path = Path(artifacts.get("metrics", run_dir / "tables" / "metrics.csv"))
        predictions_path = Path(artifacts.get("predictions", run_dir / "tables" / "predictions.csv"))
        qc_path = Path(artifacts.get("qc_report", run_dir / "qc_report.json"))
        trust_path = Path(artifacts.get("trust_outputs", run_dir / "trust_outputs.json"))

        metrics = _load_csv(metrics_path)
        predictions = _load_csv(predictions_path)
        qc_report = _load_json(qc_path)
        trust_outputs = _load_json(trust_path)
        figures = _collect_figures(run_dir / "figures")

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
