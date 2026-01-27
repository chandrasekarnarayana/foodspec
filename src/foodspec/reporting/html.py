"""HTML report builder for FoodSpec reporting subsystem."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Dict, List

from foodspec.reporting.cards import build_experiment_card_from_bundle
from foodspec.reporting.modes import ReportMode, get_mode_config, validate_artifacts
from foodspec.reporting.schema import RunBundle


def _read_image_as_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = "image/png"
    if path.suffix.lower() == ".svg":
        mime = "image/svg+xml"
    elif path.suffix.lower() == ".pdf":
        mime = "application/pdf"
    return f"data:{mime};base64,{b64}"


class HtmlReportBuilder:
    """Build HTML reports from RunBundle and mode configuration."""

    def __init__(self, bundle: RunBundle, mode: ReportMode | str, title: str = "FoodSpec Report") -> None:
        self.bundle = bundle
        self.mode = ReportMode(mode) if isinstance(mode, str) else mode
        self.title = title

    def build(self, out_dir: Path | str, *, embed_images: bool = False) -> Path:
        """Write report HTML to reports/report.html and return its path."""
        out_dir = Path(out_dir)
        reports_dir = out_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "report.html"

        mode_config = get_mode_config(self.mode)
        card = build_experiment_card_from_bundle(self.bundle, mode=self.mode)

        _, missing = validate_artifacts(self.mode, self.bundle.available_artifacts)
        sections = self._build_sections(
            mode_config.enabled_sections,
            embed_images,
            missing,
            base_dir=reports_dir,
        )
        html = self._render_html(sections, card)
        report_path.write_text(html, encoding="utf-8")
        return report_path

    def _build_sections(
        self,
        enabled_sections: List[str],
        embed_images: bool,
        missing_artifacts: List[str],
        *,
        base_dir: Path,
    ) -> Dict[str, str]:
        content: Dict[str, str] = {}
        if "summary" in enabled_sections:
            summary = self.bundle.run_summary.get("summary", card_summary_text(self.bundle))
            missing_html = ""
            if missing_artifacts:
                missing_list = ", ".join(missing_artifacts)
                missing_html = f"<p><strong>Missing artifacts:</strong> {missing_list}</p>"
            content["summary"] = f"<p>{summary}</p>{missing_html}"
        if "dataset" in enabled_sections:
            content["dataset"] = self._render_key_values(
                {
                    "Run ID": self.bundle.run_id,
                    "Inputs": ", ".join(str(p) for p in self.bundle.manifest.get("inputs", [])),
                    "Protocol": self.bundle.manifest.get("protocol_path", "unknown"),
                    "Seed": self.bundle.seed,
                }
            )
        if "methods" in enabled_sections:
            content["methods"] = self._render_block(
                "Protocol Configuration",
                self.bundle.manifest.get("config", {}) or self.bundle.manifest.get("protocol_snapshot", {}),
            )
        if "metrics" in enabled_sections:
            content["metrics"] = self._render_table(self.bundle.metrics)
        if "qc" in enabled_sections:
            content["qc"] = self._render_block("QC Report", self.bundle.qc_report)
        if "uncertainty" in enabled_sections:
            content["uncertainty"] = self._render_block("Trust Outputs", self.bundle.trust_outputs)
        if "readiness" in enabled_sections:
            readiness = {}
            if isinstance(self.bundle.trust_outputs, dict):
                readiness = self.bundle.trust_outputs.get("readiness", {})
            content["readiness"] = self._render_block("Regulatory Readiness", readiness)
        if "limitations" in enabled_sections:
            content["limitations"] = "<p>Limitations and known risks are documented in the experiment card.</p>"

        fig_html = self._render_figures(embed_images, base_dir)
        if fig_html:
            content["figures"] = fig_html
        return content

    def _render_key_values(self, rows: Dict[str, object]) -> str:
        items = []
        for key, value in rows.items():
            items.append(f"<tr><th>{key}</th><td>{value}</td></tr>")
        return "<table class='kv'>" + "".join(items) + "</table>"

    def _render_block(self, title: str, payload: object) -> str:
        return f"<h4>{title}</h4><pre>{payload}</pre>"

    def _render_table(self, rows: List[Dict[str, object]]) -> str:
        if not rows:
            return "<p>No metrics available.</p>"
        headers = rows[0].keys()
        header_html = "".join([f"<th>{h}</th>" for h in headers])
        body_html = ""
        for row in rows:
            body_html += "<tr>" + "".join([f"<td>{row.get(h, '')}</td>" for h in headers]) + "</tr>"
        return f"<table class='metrics'><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"

    def _render_figures(self, embed_images: bool, base_dir: Path) -> str:
        if not self.bundle.figures:
            return ""
        cards = []
        for fig_path in self.bundle.figures:
            if fig_path.suffix.lower() not in {".png", ".svg"}:
                continue
            if embed_images:
                src = _read_image_as_data_uri(fig_path)
            else:
                rel = os.path.relpath(fig_path, base_dir)
                src = str(rel).replace("\\", "/")
            cards.append(f"<div class='figure'><img src='{src}' alt='{fig_path.stem}'></div>")
        return "<div class='figures'>" + "".join(cards) + "</div>"

    def _render_html(self, sections: Dict[str, str], card) -> str:
        section_blocks = []
        for name, body in sections.items():
            label = name.replace("_", " ").title()
            section_blocks.append(f"<section><h2>{label}</h2>{body}</section>")

        card_html = (
            f"<div class='card'>"
            f"<h3>Experiment Card</h3>"
            f"<p><strong>Summary:</strong> {card.auto_summary}</p>"
            f"<p><strong>Confidence:</strong> {card.confidence_level.value}</p>"
            f"<p><strong>Readiness:</strong> {card.deployment_readiness.value}</p>"
            f"</div>"
        )

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self.title}</title>
  <style>
    body {{
      font-family: "DejaVu Sans", Arial, sans-serif;
      margin: 0;
      padding: 24px;
      background: #f6f6f2;
      color: #1f2428;
    }}
    h1 {{ margin-bottom: 8px; }}
    .card {{
      background: #fff7ec;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid #f1d4b8;
      margin-bottom: 24px;
    }}
    section {{
      background: #ffffff;
      padding: 16px;
      border-radius: 8px;
      margin-bottom: 16px;
      border: 1px solid #e4e4e4;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 6px;
      text-align: left;
    }}
    .figures {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .figure img {{
      width: 100%;
      border-radius: 6px;
      border: 1px solid #e1e1e1;
    }}
  </style>
</head>
<body>
  <h1>{self.title}</h1>
  <p>Mode: {self.mode.value}</p>
  {card_html}
  {''.join(section_blocks)}
</body>
</html>
"""


def card_summary_text(bundle: RunBundle) -> str:
    return bundle.run_summary.get("summary") or "Report summary unavailable."


__all__ = ["HtmlReportBuilder"]
