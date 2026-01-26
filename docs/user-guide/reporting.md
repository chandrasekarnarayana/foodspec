# Reporting and Visualization

FoodSpec reporting produces a complete, reproducible bundle for each run.
The reporting subsystem builds HTML and PDF reports, experiment cards, paper-ready
figures, dossiers, and comparison dashboards from run artifacts.

## Reporting modes

FoodSpec supports three modes with different strictness and content:

| Mode | Emphasis | Typical use |
| --- | --- | --- |
| research | interpretability and analysis | exploratory or publication workflows |
| regulatory | QC, traceability, compliance | formal review and audit readiness |
| monitoring | drift detection, baseline tracking | production monitoring |

Modes control which sections appear in the report and how strictly QC is enforced.

## CLI usage

Generate a report directly from a protocol run:

```bash
foodspec report run --input data.csv --protocol protocol.yaml --mode research --outdir runs/report_demo
```

Compare multiple runs:

```bash
foodspec report compare --runs runs/run_a runs/run_b --outdir runs/report_compare
```

## Output layout

Each reporting run produces a standard artifact tree:

```
{OUTDIR}/
  manifest.json
  run_summary.json
  logs/run.log
  reports/report.html
  reports/report.pdf
  figures/*.png + *.svg + *.pdf
  cards/experiment_card.md
  dossier/dossier.md
  dossier/appendices/qc.md
  dossier/appendices/uncertainty.md
  compare/dashboard.html
  compare/leaderboard.csv
  compare/radar.png
```

PDF output is optional and requires `weasyprint`. If it is not installed,
FoodSpec will keep the HTML output and emit a warning.

## Deterministic figures

All reporting figures are generated with a seed and exported with stable
filenames. Set `--seed` on the CLI to make plots deterministic.

## API usage

The reporting API can be used directly with a `RunBundle`:

```python
from foodspec.reporting import HtmlReportBuilder, ReportMode, RunBundle

bundle = RunBundle.from_run_dir("runs/report_demo")
HtmlReportBuilder(bundle, ReportMode.RESEARCH).build("runs/report_demo")
```

## Example script

See `examples/reporting/report_run_demo.py` for a minimal, standalone example
that builds a report from a synthetic run directory.
