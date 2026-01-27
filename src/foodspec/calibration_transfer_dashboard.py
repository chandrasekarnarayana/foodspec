"""Simple HTML dashboard generator for calibration transfer success metrics.

Generates a standalone HTML report summarizing pre/post metrics and improvement ratios.
"""

from __future__ import annotations

from typing import Any, Dict

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Calibration Transfer Dashboard</title>
<style>
  body { font-family: Arial, sans-serif; margin: 24px; }
  h1 { font-size: 22px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
  .metric { font-size: 14px; margin: 4px 0; }
  .good { color: #1a7f37; }
  .warn { color: #b54708; }
  .bad { color: #b91c1c; }
</style>
</head>
<body>
<h1>Calibration Transfer Dashboard</h1>
<p>Overview of transfer success metrics. Values are best-effort based on available metrics.</p>
<div class="grid">
  <div class="card">
    <h2>Pre-Transfer</h2>
    <div class="metric">RMSE: {pre_rmse}</div>
    <div class="metric">R²: {pre_r2}</div>
    <div class="metric">MAE: {pre_mae}</div>
  </div>
  <div class="card">
    <h2>Post-Transfer</h2>
    <div class="metric">RMSE: {post_rmse}</div>
    <div class="metric">R²: {post_r2}</div>
    <div class="metric">MAE: {post_mae}</div>
  </div>
</div>
<div class="card" style="margin-top:16px">
  <h2>Improvement Ratios</h2>
  <div class="metric">RMSE improvement: {rmse_improvement}</div>
  <div class="metric">R² improvement: {r2_improvement}</div>
  <div class="metric">MAE improvement: {mae_improvement}</div>
</div>
<div class="card" style="margin-top:16px">
  <h2>Standards & Transformation</h2>
  <div class="metric">n_standards: {n_standards}</div>
  <div class="metric">condition_number: {condition_number}</div>
</div>
</body>
</html>
"""


def build_dashboard_html(metrics: Dict[str, Any]) -> str:
    """Render calibration transfer metrics into a simple HTML dashboard.

    Expected keys in metrics (best-effort; missing keys become 'N/A'):
    - 'pre_rmse', 'pre_r2', 'pre_mae'
    - 'post_rmse', 'post_r2', 'post_mae'
    - 'rmse_improvement', 'r2_improvement', 'mae_improvement'
    - 'n_standards', 'condition_number'
    """

    def get(k):
        v = metrics.get(k)
        return f"{v:.4f}" if isinstance(v, (int, float)) else (str(v) if v is not None else "N/A")

    html = HTML_TEMPLATE.format(
        pre_rmse=get("pre_rmse"),
        pre_r2=get("pre_r2"),
        pre_mae=get("pre_mae"),
        post_rmse=get("post_rmse"),
        post_r2=get("post_r2"),
        post_mae=get("post_mae"),
        rmse_improvement=get("rmse_improvement"),
        r2_improvement=get("r2_improvement"),
        mae_improvement=get("mae_improvement"),
        n_standards=get("n_standards"),
        condition_number=get("condition_number"),
    )
    return html
