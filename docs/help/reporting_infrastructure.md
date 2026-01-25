# Reporting Infrastructure

FoodSpec provides a three-layer reporting system to turn protocol run artifacts into HTML reports and confidence-assessed experiment cards.

## Layers
- **Modes** (`ReportMode`, `ModeConfig`): Configure sections, required artifacts, and strictness for RESEARCH, REGULATORY, MONITORING.
- **Context & Builder** (`ReportContext`, `ReportBuilder`): Load run artifacts (manifest, metrics, qc, trust outputs, figures) and render HTML via Jinja2 templates.
- **Cards & Risk** (`ExperimentCard`, `build_experiment_card`): Summarize runs with headline metrics, risk scoring, confidence level (LOW/MEDIUM/HIGH), and deployment readiness (NOT_READY/PILOT/READY).

## Minimal Artifacts
- `manifest.json` (built with `RunManifest`)
- `metrics.csv` (must include macro_f1, auroc; optional ece)
- Optional: `trust_outputs.json` (ece, coverage, abstain_rate), `qc.csv`, figures under `plots/`.

## Risk Scoring
- Risks: ECE > 0.1, coverage < 0.90, abstain_rate > 0.10, missing macro_f1/auroc, random validation scheme, missing hashes (regulatory), no QC.
- Confidence: 0 risks → HIGH, 1-2 risks → MEDIUM, 3+ risks → LOW.
- Deployment: LOW → NOT_READY, MEDIUM → PILOT, HIGH → READY; regulatory without hashes → NOT_READY.

## Quickstart (Python)
```python
from pathlib import Path
from foodspec.reporting import ReportContext, ReportBuilder, build_experiment_card, ReportMode

run_dir = Path("./run/")
context = ReportContext.load(run_dir)
ReportBuilder(context).build_html(run_dir / "report.html", mode=ReportMode.RESEARCH)
card = build_experiment_card(context, mode=ReportMode.RESEARCH)
card.to_json(run_dir / "card.json")
card.to_markdown(run_dir / "card.md")
```

## CLI
```bash
foodspec report-run --run-dir ./run --mode research --format all
# Generates: report.html, card.json, card.md
```

## Mode Guide
- **RESEARCH**: Permissive; requires manifest + metrics; warnings not fatal.
- **REGULATORY**: Strict; requires manifest, metrics, protocol snapshot, data fingerprint/hashes, QC; warnings treated as errors.
- **MONITORING**: Balanced; requires manifest, metrics, baseline; focused on drift/coverage.

## Templates
- HTML template: `foodspec/reporting/templates/base.html` (sidebar navigation, sections conditional on mode).

## References
- API: `foodspec.reporting` exports `ReportMode`, `ModeConfig`, `ReportContext`, `ReportBuilder`, `ExperimentCard`, `ConfidenceLevel`, `DeploymentReadiness`, `build_experiment_card`, `collect_figures`.
- Tests: `tests/test_reporting_*.py` and `tests/test_reporting_integration_e2e.py` cover modes, context, builder, cards, and end-to-end generation.
