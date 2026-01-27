"""Regulatory mode enforcement and restrictions.

Provides:
- Approved model registry
- Override governance (tracking + restrictions)
- Mandatory trust stack requirements
- Mandatory reporting requirements
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

# Approved models for regulatory mode
APPROVED_MODELS: Set[str] = {
    "LogisticRegression",
    "SVC",
    "RandomForest",
    "GradientBoosting",
    "ExtraTreesClassifier",
}

# Trust stack components
TRUST_COMPONENTS: Set[str] = {
    "calibration",
    "conformal",
    "abstention",
}

# Reporting requirements
REPORTING_MODES: Set[str] = {
    "html",
    "pdf",
    "json",
}


@dataclass
class OverrideGovern:
    """Track overrides and enforce governance."""

    overrides: List[Dict[str, Any]]
    """List of approved overrides."""

    max_overrides_per_run: int = 1
    """Maximum overrides allowed in regulatory mode."""

    require_justification: bool = True
    """Require justification text for overrides."""


def enforce_model_approved(model_name: str, mode: str, approved_models: Optional[Set[str]] = None) -> tuple[bool, str]:
    """Check if model is approved for regulatory mode.

    Parameters
    ----------
    model_name : str
        Name of model to check.
    mode : str
        Workflow mode: "research" or "regulatory".
    approved_models : Optional[Set[str]]
        Custom approved models set. If None, uses APPROVED_MODELS.

    Returns
    -------
    tuple[bool, str]
        (is_approved, message)
    """
    if mode == "research":
        return True, "Research mode allows any model"

    if approved_models is None:
        approved_models = APPROVED_MODELS

    if model_name in approved_models:
        return True, f"Model '{model_name}' is approved for regulatory use"

    return False, f"Model '{model_name}' not in approved registry: {sorted(approved_models)}"


def enforce_trust_stack(
    enable_trust: bool,
    mode: str,
    components: Optional[Set[str]] = None,
) -> tuple[bool, str]:
    """Enforce trust stack requirements.

    Parameters
    ----------
    enable_trust : bool
        Whether trust stack is enabled.
    mode : str
        Workflow mode: "research" or "regulatory".
    components : Optional[Set[str]]
        Required trust components. If None, all are required in regulatory mode.

    Returns
    -------
    tuple[bool, str]
        (is_satisfied, message)
    """
    if mode == "research":
        return True, "Trust stack is advisory in research mode"

    if not enable_trust:
        return False, "Trust stack is mandatory in regulatory mode but not enabled"

    if components is None:
        components = TRUST_COMPONENTS

    missing = TRUST_COMPONENTS - components
    if missing:
        return False, f"Regulatory mode requires all trust components: missing {sorted(missing)}"

    return True, f"Trust stack satisfied with {len(components)} component(s)"


def enforce_reporting(
    enable_report: bool,
    mode: str,
) -> tuple[bool, str]:
    """Enforce reporting requirements.

    Parameters
    ----------
    enable_report : bool
        Whether reporting is enabled.
    mode : str
        Workflow mode: "research" or "regulatory".

    Returns
    -------
    tuple[bool, str]
        (is_satisfied, message)
    """
    if mode == "research":
        return True, "Reporting is advisory in research mode"

    if not enable_report:
        return False, "Reporting is mandatory in regulatory mode but not enabled"

    return True, "Regulatory reporting enabled"


def check_override_governance(
    override_count: int,
    override_justifications: Optional[List[str]] = None,
    max_allowed: int = 1,
) -> tuple[bool, str]:
    """Check override governance rules.

    Parameters
    ----------
    override_count : int
        Number of QC gate overrides applied.
    override_justifications : Optional[List[str]]
        Justification text for each override.
    max_allowed : int
        Maximum overrides allowed per run.

    Returns
    -------
    tuple[bool, str]
        (is_compliant, message)
    """
    if override_count > max_allowed:
        return False, f"Too many overrides: {override_count} > {max_allowed}"

    if override_count > 0 and override_justifications:
        # Check all have justification
        missing_just = sum(1 for j in override_justifications if not j or len(j.strip()) < 10)
        if missing_just > 0:
            return False, f"{missing_just} override(s) lack proper justification (min 10 chars)"

    return True, f"Override governance satisfied ({override_count} override(s))"


def get_regulatory_restrictions(mode: str) -> Dict[str, Any]:
    """Get all regulatory restrictions for a mode.

    Parameters
    ----------
    mode : str
        Workflow mode: "research" or "regulatory".

    Returns
    -------
    Dict[str, Any]
        Dictionary of restrictions/requirements.
    """
    if mode == "research":
        return {
            "model_approval_required": False,
            "trust_stack_required": False,
            "reporting_required": False,
            "override_allowed": True,
            "max_overrides": None,
        }

    return {
        "model_approval_required": True,
        "approved_models": sorted(APPROVED_MODELS),
        "trust_stack_required": True,
        "required_trust_components": sorted(TRUST_COMPONENTS),
        "reporting_required": True,
        "override_allowed": True,  # But tracked/justified
        "max_overrides": 1,
        "require_override_justification": True,
    }
