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
Reporting module: Report templates, export formats, formatting utilities.

Generating reports and experiment cards:
    from foodspec.reporting import ReportBuilder, ReportContext, build_experiment_card, ReportMode
    context = ReportContext.load(Path("/run/dir"))
    builder = ReportBuilder(context)
    html_path = builder.build_html(Path("/run/dir/report.html"), mode=ReportMode.RESEARCH)
    card = build_experiment_card(context)
    card.to_json(Path("/run/dir/card.json"))
"""

from foodspec.reporting.base import ReportBuilder, ReportContext, collect_figures
from foodspec.reporting.cards import (
    ConfidenceLevel,
    DeploymentReadiness,
    ExperimentCard,
    build_experiment_card,
)
from foodspec.reporting.report import DEFAULT_TEMPLATE, generate_html_report
from foodspec.reporting.modes import ReportMode, ModeConfig, get_mode_config, list_modes, validate_artifacts
from foodspec.reporting.engine import ReportingEngine, ReportOutputs

__all__ = [
    "ReportBuilder",
    "ReportContext",
    "collect_figures",
    "ExperimentCard",
    "ConfidenceLevel",
    "DeploymentReadiness",
    "build_experiment_card",
    "generate_html_report",
    "DEFAULT_TEMPLATE",
    "ReportMode",
    "ModeConfig",
    "get_mode_config",
    "list_modes",
    "validate_artifacts",
    "ReportingEngine",
    "ReportOutputs",
]
