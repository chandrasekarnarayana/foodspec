"""Simple HTML report generation for oil authentication results."""
from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from foodspec.apps.oils import OilAuthResult
from foodspec.viz.classification import plot_confusion_matrix

__all__ = ["render_html_report_oil_auth"]


def render_html_report_oil_auth(result: OilAuthResult, output_path: PathLike) -> Path:
    """Render an HTML report for oil authentication results.

    Saves confusion matrix plot and embeds metrics.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = output_path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    cm_path = assets_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_confusion_matrix(result.confusion_matrix, class_names=result.class_labels, ax=ax)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    metrics_html = result.cv_metrics.to_html(index=False)
    feature_importances_html: Optional[str] = None
    if result.feature_importances is not None:
        feature_importances_html = result.feature_importances.sort_values(ascending=False).to_frame().to_html()

    html = f"""
    <html>
      <head>
        <title>Oil Authentication Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
          h1, h2 {{ color: #333; }}
          .section {{ margin-bottom: 2rem; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
        </style>
      </head>
      <body>
        <h1>Oil Authentication Report</h1>
        <div class="section">
          <h2>Cross-validation Metrics</h2>
          {metrics_html}
        </div>
        <div class="section">
          <h2>Confusion Matrix</h2>
          <img src="{cm_path.name}" alt="Confusion matrix" />
        </div>
    """

    if feature_importances_html is not None:
        html += f"""
        <div class="section">
          <h2>Feature Importances</h2>
          {feature_importances_html}
        </div>
        """

    html += """
      </body>
    </html>
    """
    output_path.write_text(html, encoding="utf-8")
    return output_path
