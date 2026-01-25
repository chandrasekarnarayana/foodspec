"""
Automated experiment reporting and analysis infrastructure.

Provides protocol-aware HTML report generation, experiment card summarization
with confidence/deployment readiness assessment, and multi-modal risk scoring.

**Three-Layer Architecture:**

1. **Modes Layer** (modes.py)
   - ReportMode enum: RESEARCH, REGULATORY, MONITORING
   - Mode-specific configuration (sections, artifacts, strictness)
   - Artifact validation per mode

2. **Context & Builder Layer** (base.py)
   - ReportContext: Load run artifacts (manifest, metrics, QC, trust outputs, figures)
   - ReportBuilder: Generate HTML with Jinja2 templating
   - collect_figures(): Index visualization directories

3. **Confidence & Risk Layer** (cards.py)
   - ExperimentCard: Summarize runs with confidence/readiness assessment
   - build_experiment_card(): Factory with risk scoring
   - Risk rules: ECE, coverage, abstention, missing metrics, random CV, missing hashes

**Example Usage:**

```python
from foodspec.reporting import (
    ReportMode, ReportContext, ReportBuilder,
    build_experiment_card, ExperimentCard
)

# Load run artifacts
context = ReportContext.load("./protocol_runs/20260125_123456_run/")

# Build HTML report
ReportBuilder(context).build_html(
    out_path="report.html",
    mode=ReportMode.RESEARCH
)

# Generate experiment card with risk assessment
card = build_experiment_card(context, mode=ReportMode.RESEARCH)
card.to_json("card.json")     # Structured export
card.to_markdown("card.md")   # Human-readable export
```

**Validation Rules:**

- RESEARCH mode: Permissive (requires manifest + metrics)
- REGULATORY mode: Strict (requires 5+ artifacts including hashes)
- MONITORING mode: Balanced (requires manifest + metrics + baseline)

**Risk Scoring:**

Confidence levels determined by risk count:
- HIGH (0 risks): Deploy with confidence
- MEDIUM (1-2 risks): Deploy as pilot
- LOW (3+ risks): Do not deploy

Risk triggers: ECE > 0.1, coverage < 90%, abstention > 10%, missing metrics,
random CV scheme, missing hashes (regulatory mode), no QC data.
"""

from __future__ import annotations

from foodspec.reporting.base import ReportBuilder, ReportContext, collect_figures
from foodspec.reporting.cards import (
    ConfidenceLevel,
    DeploymentReadiness,
    ExperimentCard,
    build_experiment_card,
)
from foodspec.reporting.modes import (
    ModeConfig,
    ReportMode,
    get_mode_config,
    list_modes,
    validate_artifacts,
)

__all__ = [
    # Modes
    "ReportMode",
    "ModeConfig",
    "get_mode_config",
    "list_modes",
    "validate_artifacts",
    # Context & Builder
    "ReportContext",
    "ReportBuilder",
    "collect_figures",
    # Cards & Risk Assessment
    "ExperimentCard",
    "ConfidenceLevel",
    "DeploymentReadiness",
    "build_experiment_card",
]
