"""Regulatory compliance checks for run artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ComplianceResult:
    standard: str
    score: float
    passed: bool
    checks: Dict[str, bool]
    findings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "standard": self.standard,
            "score": float(self.score),
            "passed": bool(self.passed),
            "checks": self.checks,
            "findings": self.findings,
        }


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _check_common(run_dir: Path) -> Dict[str, bool]:
    return {
        "manifest": (run_dir / "manifest.json").exists(),
        "run_summary": (run_dir / "run_summary.json").exists(),
        "logs": (run_dir / "logs" / "run.log").exists(),
    }


def check_iso_17025(run_dir: Path) -> ComplianceResult:
    """Evaluate ISO 17025 readiness based on artifact traceability."""
    run_dir = Path(run_dir)
    checks = _check_common(run_dir)
    checks.update(
        {
            "protocol_snapshot": bool(_load_manifest(run_dir).get("protocol_snapshot")),
            "metrics": (run_dir / "tables" / "metrics.csv").exists(),
            "qc_report": (run_dir / "qc_report.json").exists(),
        }
    )
    findings = [k for k, v in checks.items() if not v]
    score = 100.0 * sum(checks.values()) / max(1, len(checks))
    passed = score >= 80.0
    return ComplianceResult(
        standard="ISO_17025",
        score=score,
        passed=passed,
        checks=checks,
        findings=findings,
    )


def check_cfr_part11(run_dir: Path) -> ComplianceResult:
    """Evaluate FDA 21 CFR Part 11 readiness (electronic records & audit trail)."""
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    checks = _check_common(run_dir)
    checks.update(
        {
            "timestamp": bool(manifest.get("timestamp")),
            "git_commit": bool(manifest.get("git_commit")),
            "model_card": (run_dir / "model_card.md").exists() or (run_dir / "cards" / "model_card.md").exists(),
            "dataset_card": (run_dir / "dataset_card.md").exists() or (run_dir / "cards" / "dataset_card.md").exists(),
        }
    )
    findings = [k for k, v in checks.items() if not v]
    score = 100.0 * sum(checks.values()) / max(1, len(checks))
    passed = score >= 80.0
    return ComplianceResult(
        standard="FDA_21_CFR_PART_11",
        score=score,
        passed=passed,
        checks=checks,
        findings=findings,
    )


def run_compliance_suite(run_dir: Path) -> Dict[str, ComplianceResult]:
    """Run all compliance checks and return results keyed by standard."""
    return {
        "iso_17025": check_iso_17025(run_dir),
        "cfr_part_11": check_cfr_part11(run_dir),
    }


__all__ = ["ComplianceResult", "check_iso_17025", "check_cfr_part11", "run_compliance_suite"]
