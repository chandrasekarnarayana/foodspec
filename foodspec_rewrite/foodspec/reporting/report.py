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

Minimal HTML report generator using Jinja2 templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Environment, select_autoescape

from foodspec.core.artifacts import ArtifactRegistry


DEFAULT_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>FoodSpec Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 1.5rem; }
    h1, h2, h3 { color: #1f2937; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
    th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; }
    th { background: #f3f4f6; }
    .section { margin-bottom: 2rem; }
    .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
    .plot-grid img { width: 100%; border: 1px solid #e5e7eb; }
  </style>
</head>
<body>
  <h1>FoodSpec Run Report</h1>

  <div class="section">
    <h2>Run Summary</h2>
    <ul>
      <li><strong>Version:</strong> {{ run_summary.version }}</li>
      <li><strong>Seed:</strong> {{ run_summary.seed }}</li>
      <li><strong>Data path:</strong> {{ run_summary.data_path }}</li>
    </ul>
  </div>

  <div class="section">
    <h2>Dataset Summary</h2>
    <ul>
    {% for key, val in dataset_summary.items() %}
      <li><strong>{{ key }}:</strong> {{ val }}</li>
    {% endfor %}
    </ul>
  </div>

  <div class="section">
    <h2>Preprocessing Steps</h2>
    <ol>
    {% for step in preprocessing_steps %}
      <li>{{ step }}</li>
    {% endfor %}
    </ol>
  </div>

  <div class="section">
    <h2>Quality Control</h2>
    <table>
      <thead><tr>{% for col in qc_columns %}<th>{{ col }}</th>{% endfor %}</tr></thead>
      <tbody>
        {% for row in qc_table %}
          <tr>{% for col in qc_columns %}<td>{{ row.get(col, "") }}</td>{% endfor %}</tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Main Metrics</h2>
    <table>
      <thead><tr>{% for col in metric_columns %}<th>{{ col }}</th>{% endfor %}</tr></thead>
      <tbody>
        {% for row in metrics %}
          <tr>{% for col in metric_columns %}<td>{{ row.get(col, "") }}</td>{% endfor %}</tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Plots</h2>
    <div class="plot-grid">
      {% for plot in plots %}
        <div><img src="{{ plot }}" alt="plot"></div>
      {% endfor %}
    </div>
  </div>

  <div class="section">
    <h2>Uncertainty Summary</h2>
    <ul>
    {% for key, val in uncertainty.items() %}
      <li><strong>{{ key }}:</strong> {{ val }}</li>
    {% endfor %}
    </ul>
  </div>

</body>
</html>
"""


def _to_run_summary(manifest: Mapping[str, Any]) -> dict:
    return {
        "version": manifest.get("version", "unknown"),
        "seed": manifest.get("seed", ""),
        "data_path": manifest.get("data_path", ""),
    }


def generate_html_report(
    artifacts: ArtifactRegistry,
    manifest: Mapping[str, Any] | Any,
    dataset_summary: Mapping[str, Any],
    preprocessing_steps: Sequence[str],
    qc_table: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    plots: Sequence[str],
    uncertainty: Mapping[str, Any],
    template_str: str = DEFAULT_TEMPLATE,
) -> Path:
    """Render an HTML report and write to artifacts.report_html_path.

    Parameters
    ----------
    artifacts : ArtifactRegistry
        Registry managing output paths.
    manifest : mapping or object
        Run manifest with at least version, seed, data_path.
    dataset_summary : mapping
        Key-value summary of the dataset.
    preprocessing_steps : sequence of str
        Ordered preprocessing steps.
    qc_table : sequence of mappings
        QC table rows (dicts) to render.
    metrics : sequence of mappings
        Main metrics rows.
    plots : sequence of str
        Paths (relative or absolute) to plot images to embed.
    uncertainty : mapping
        Uncertainty summary key-values.
    template_str : str
        Jinja2 template string to render.
    """

    artifacts.ensure_layout()
    if hasattr(manifest, "model_dump"):
        manifest_data = manifest.model_dump(mode="python")
    elif hasattr(manifest, "__dict__"):
        manifest_data = dict(manifest.__dict__)
    else:
        manifest_data = dict(manifest)

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(template_str)

    qc_columns = list(qc_table[0].keys()) if qc_table else []
    metric_columns = list(metrics[0].keys()) if metrics else []

    html = template.render(
        run_summary=_to_run_summary(manifest_data),
        dataset_summary=dataset_summary,
        preprocessing_steps=list(preprocessing_steps),
        qc_table=list(qc_table),
        qc_columns=qc_columns,
        metrics=list(metrics),
        metric_columns=metric_columns,
        plots=list(plots),
        uncertainty=uncertainty,
    )

    path = artifacts.report_html_path
    path.write_text(html)
    return path


__all__ = ["generate_html_report", "DEFAULT_TEMPLATE"]
