from __future__ import annotations
"""
Reporting modes and their configurations.

Each mode defines which sections to include, strictness of validation,
required artifacts, and default plots to embed. This enables switching
report content and emphasis without modifying reporting logic.

Modes:
  - RESEARCH: Emphasizes analysis, stats, and interpretability
  - REGULATORY: Emphasizes QC, traceability, and compliance
  - MONITORING: Emphasizes drift detection and baseline comparison
"""


from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ReportMode(str, Enum):
    """Reporting mode enumeration."""

    RESEARCH = "research"
    REGULATORY = "regulatory"
    MONITORING = "monitoring"


@dataclass
class ModeConfig:
    """Configuration for a reporting mode."""

    mode: ReportMode
    enabled_sections: List[str]
    required_artifacts: List[str]
    default_plots: List[str]
    strictness_level: int  # 1=research, 2=regulatory, 3=monitoring (can check QC strictness etc)
    warnings_as_errors: bool
    description: str


# Mode configurations
_MODE_CONFIGS: Dict[ReportMode, ModeConfig] = {
    ReportMode.RESEARCH: ModeConfig(
        mode=ReportMode.RESEARCH,
        enabled_sections=[
            "summary",
            "dataset",
            "methods",
            "metrics",
            "multivariate",
            "uncertainty",
            "limitations",
        ],
        required_artifacts=["manifest", "metrics"],
        default_plots=[
            "confusion_matrix",
            "roc_curve",
            "feature_importance",
            "uncertainty_distribution",
        ],
        strictness_level=1,
        warnings_as_errors=False,
        description="Comprehensive analysis emphasizing interpretability and statistical insights",
    ),
    ReportMode.REGULATORY: ModeConfig(
        mode=ReportMode.REGULATORY,
        enabled_sections=[
            "summary",
            "dataset",
            "methods",
            "qc",
            "metrics",
            "multivariate",
            "uncertainty",
            "readiness",
        ],
        required_artifacts=["manifest", "metrics", "qc", "protocol_snapshot", "data_fingerprint"],
        default_plots=[
            "qc_pass_rate",
            "metrics_by_fold",
            "calibration_curve",
            "uncertainty_by_batch",
        ],
        strictness_level=3,
        warnings_as_errors=True,
        description="Compliance-focused with mandatory QC, traceability, and full reproducibility",
    ),
    ReportMode.MONITORING: ModeConfig(
        mode=ReportMode.MONITORING,
        enabled_sections=[
            "summary",
            "dataset",
            "metrics",
            "uncertainty",
        ],
        required_artifacts=["manifest", "metrics", "previous_run_baseline"],
        default_plots=[
            "batch_drift",
            "metric_trend",
            "comparison_radar",
            "performance_leaderboard",
        ],
        strictness_level=2,
        warnings_as_errors=False,
        description="Operational monitoring with emphasis on drift detection and baseline comparison",
    ),
}


def get_mode_config(mode: ReportMode | str) -> ModeConfig:
    """Retrieve configuration for a reporting mode.

    Parameters
    ----------
    mode : ReportMode or str
        The reporting mode (RESEARCH, REGULATORY, or MONITORING).
        Can be passed as enum or string (case-insensitive).

    Returns
    -------
    ModeConfig
        Configuration dataclass with sections, artifacts, plots, strictness, and warnings_as_errors.

    Raises
    ------
    ValueError
        If mode is not recognized.

    Examples
    --------
    >>> config = get_mode_config(ReportMode.RESEARCH)
    >>> assert "interpretation" in config.enabled_sections
    >>> assert not config.warnings_as_errors

    >>> config = get_mode_config("regulatory")
    >>> assert config.warnings_as_errors
    >>> assert "qc" in config.enabled_sections
    """
    if isinstance(mode, str):
        mode_str = mode.upper()
        try:
            mode = ReportMode(mode_str.lower())
        except ValueError:
            raise ValueError(
                f"Unknown reporting mode '{mode}'. "
                f"Valid modes: {sorted([m.value for m in ReportMode])}"
            )

    if mode not in _MODE_CONFIGS:
        raise ValueError(f"No configuration for mode {mode}")

    return _MODE_CONFIGS[mode]


def list_modes() -> Dict[str, str]:
    """List all available modes and their descriptions.

    Returns
    -------
    dict
        Mapping of mode names to descriptions.
    """
    return {m.value: _MODE_CONFIGS[m].description for m in ReportMode}


def validate_artifacts(
    mode: ReportMode | str,
    available_artifacts: List[str],
    warnings_as_errors: bool = False,
) -> tuple[bool, List[str]]:
    """Validate that required artifacts for a mode are available.

    Parameters
    ----------
    mode : ReportMode or str
        The reporting mode.
    available_artifacts : list of str
        List of artifact names/paths available in the run.
    warnings_as_errors : bool
        If True, treat missing artifacts as errors (for REGULATORY mode).

    Returns
    -------
    valid : bool
        True if all required artifacts are present (or if not warnings_as_errors).
    missing : list of str
        List of missing artifact names.
    """
    config = get_mode_config(mode)
    required = config.required_artifacts
    missing = [a for a in required if a not in available_artifacts]

    if config.warnings_as_errors and missing:
        return False, missing
    return True, missing


__all__ = ["ReportMode", "ModeConfig", "get_mode_config", "list_modes", "validate_artifacts"]
