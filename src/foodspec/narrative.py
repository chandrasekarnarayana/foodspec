"""
Narrative and publication helpers.

Takes a run folder (protocol bundle) and produces:
- Methods-style narrative text (Markdown)
- Figure panel PDF assembling key plots
- Supplemental summary CSV (stub)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_methods_text(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text()) if (run_dir / "metadata.json").exists() else {}
    report_txt = (run_dir / "report.txt").read_text() if (run_dir / "report.txt").exists() else ""
    proto = meta.get("protocol", "unknown")
    proto_ver = meta.get("protocol_version", "unknown")
    val_strategy = meta.get("validation_strategy", "standard")
    preprocessing = meta.get("preprocessing", {})
    # Assemble a simple Methods narrative
    lines: List[str] = []
    lines.append(f"# Methods – FoodSpec protocol {proto} (v{proto_ver})")
    lines.append("")
    lines.append("## Data & Protocol")
    lines.append(f"- Protocol: {proto} (version {proto_ver})")
    lines.append(f"- Inputs: {meta.get('inputs', [])}")
    lines.append(f"- Validation strategy: {val_strategy}")
    lines.append("")
    lines.append("## Preprocessing")
    if preprocessing:
        lines.append(
            "- Baseline: "
            f"{preprocessing.get('baseline_method','als')}, "
            f"lambda={preprocessing.get('baseline_lambda','')}, "
            f"p={preprocessing.get('baseline_p','')}"
        )
        lines.append(
            "- Smoothing: "
            f"window={preprocessing.get('smooth_window','')}, "
            f"poly={preprocessing.get('smooth_polyorder','')}"
        )
        lines.append(f"- Normalization: {preprocessing.get('normalization','reference')}")
    else:
        lines.append("- See run report for preprocessing details.")
    lines.append("")
    lines.append("## Analyses")
    lines.append("- Ratio-Quality (stability, discrimination, trends, oil-vs-chips, minimal panel, clustering).")
    lines.append("- Validation metrics and guardrails as reported below.")
    lines.append("")
    lines.append("## Summary of Results")
    lines.append(report_txt[:2000])  # truncated report snippet
    return "\n".join(lines)


def _select_figures(fig_paths: List[Path], limit: Optional[int] = None, profile: str = "standard") -> List[Path]:
    profile = profile.lower()
    if profile == "qa":
        priority_tags = ["confusion", "discrimin", "stability", "trend"]
    elif profile == "quicklook":
        priority_tags = ["confusion", "stability"]
    elif profile == "publication":
        priority_tags = ["stability", "discrimin", "trend", "confusion", "roc"]
    else:
        priority_tags = ["stability", "discrimin", "trend", "confusion", "roc"]
    scored = []
    for p in fig_paths:
        name = p.name.lower()
        score = sum(tag in name for tag in priority_tags)
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1].name))
    selected = [p for _, p in scored if _ >= 0]
    if limit:
        return selected[:limit]
    return selected


def build_figure_panel(
    run_dir: Path,
    out_pdf: Path,
    fig_limit: Optional[int] = None,
    include_all: bool = False,
    profile: str = "standard",
):
    run_dir = Path(run_dir)
    figs_dir = run_dir / "figures"
    images = list(figs_dir.glob("*.png"))
    if not images:
        return
    chosen = images if include_all else _select_figures(images, fig_limit, profile=profile)
    if not chosen:
        return
    with PdfPages(out_pdf) as pdf:
        ncols = 2
        nrows = max(1, (len(chosen) + 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, img_path in zip(axes, chosen):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(img_path.name)
        # hide unused
        for ax in axes[len(chosen) :]:
            ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def save_markdown_bundle(
    run_dir: Path,
    out_dir: Path,
    fig_limit: Optional[int] = None,
    include_all: bool = False,
    profile: str = "standard",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods_text = generate_methods_text(run_dir)
    (out_dir / "methods.md").write_text(methods_text, encoding="utf-8")
    # Copy report.txt as supplement
    rep = Path(run_dir) / "report.txt"
    if rep.exists():
        (out_dir / "report.txt").write_text(rep.read_text(), encoding="utf-8")
    # Figure panel
    build_figure_panel(run_dir, out_dir / "figures.pdf", fig_limit=fig_limit, include_all=include_all, profile=profile)
