"""Public reporting helpers for programmatic reuse.

Provides a stable callable used by CLI and orchestrators to generate
reports/cards from a run directory without reimplementing CLI logic.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

from foodspec.reporting import ReportBuilder, ReportContext, build_experiment_card
from foodspec.reporting.cards import build_experiment_card_from_bundle
from foodspec.reporting.html import HtmlReportBuilder
from foodspec.reporting.modes import ReportMode
from foodspec.reporting.pdf import export_pdf
from foodspec.reporting.schema import RunBundle


def _resolve_mode(mode: str) -> ReportMode:
    mode_lower = mode.lower()
    if mode_lower == "research":
        return ReportMode.RESEARCH
    if mode_lower == "regulatory":
        return ReportMode.REGULATORY
    if mode_lower == "monitoring":
        return ReportMode.MONITORING
    raise ValueError("Invalid mode. Choose: research|regulatory|monitoring")


def build_report_from_run(
    run_dir: Path | str,
    *,
    out_dir: Path | str | None = None,
    mode: str = "research",
    pdf: bool = False,
    title: str = "FoodSpec Report",
) -> Dict[str, str]:
    """Generate HTML (+ optional PDF) report and cards from a run directory.

    Returns a mapping of artifact names to paths for easy manifest wiring.
    """

    run_path = Path(run_dir).resolve()
    out_path = Path(out_dir or run_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    report_mode = _resolve_mode(mode)
    run_summary_path = run_path / "run_summary.json"
    if run_summary_path.exists():
        bundle = RunBundle.from_run_dir(run_path)
        html_builder = HtmlReportBuilder(bundle, mode=report_mode, title=title)
        html_structured = html_builder.build(out_path)
        html_path = out_path / "report.html"
        if html_structured != html_path:
            shutil.copy2(html_structured, html_path)

        card = build_experiment_card_from_bundle(bundle, mode=report_mode)
        card_json = out_path / "card.json"
        card_md = out_path / "card.md"
        card.to_json(card_json)
        card.to_markdown(card_md)
        cards_dir = out_path / "cards"
        card.to_json(cards_dir / "experiment_card.json")
        card.to_markdown(cards_dir / "experiment_card.md")

        pdf_path = None
        if pdf:
            pdf_path = export_pdf(html_path, out_path / "report.pdf")

        artifacts = {
            "report_html": str(html_path),
            "card_json": str(card_json),
            "card_markdown": str(card_md),
        }
        if html_structured != html_path:
            artifacts["report_html_structured"] = str(html_structured)
        if pdf_path:
            artifacts["report_pdf"] = str(pdf_path)
        return artifacts

    context = ReportContext.load(run_path)

    # HTML report
    html_path = out_path / "report.html"
    ReportBuilder(context).build_html(html_path, mode=report_mode, title=title)

    # Experiment card
    card = build_experiment_card(context, mode=report_mode)
    card_json = out_path / "card.json"
    card_md = out_path / "card.md"
    card.to_json(card_json)
    card.to_markdown(card_md)

    pdf_path = None
    if pdf:
        pdf_path = export_pdf(html_path, out_path / "report.pdf")

    artifacts = {
        "report_html": str(html_path),
        "card_json": str(card_json),
        "card_markdown": str(card_md),
    }
    if pdf_path:
        artifacts["report_pdf"] = str(pdf_path)
    return artifacts


__all__ = ["build_report_from_run"]
