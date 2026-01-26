from __future__ import annotations
"""Multivariate reporting section builder.

Detects multivariate artifacts under a run directory, renders publication-grade
figures, and returns structured context for HTML/experiment cards.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from foodspec.reporting.figures import FigureExporter, FigureStyle


@dataclass
class MultivariateMethodArtifacts:
    name: str
    summary: Dict[str, Any]
    scores_path: Optional[Path]
    loadings_path: Optional[Path]
    variance_path: Optional[Path]
    figures: List[Path]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _detect_label_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cand_labels = ["label", "class", "target", "y"]
    cand_batch = ["batch", "batch_id"]
    cand_stage = ["stage", "phase", "step"]
    def _pick(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None
    return {
        "label": _pick(cand_labels),
        "batch": _pick(cand_batch),
        "stage": _pick(cand_stage),
    }


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _plot_scores(df: pd.DataFrame, name: str, label_col: Optional[str], batch_col: Optional[str], stage_col: Optional[str], out_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    exporter = FigureExporter(style=FigureStyle.JOSS, size_preset="double", formats=("png",))
    num_cols = _numeric_columns(df)
    if "sample_id" in num_cols:
        num_cols.remove("sample_id")
    if len(num_cols) < 2:
        return []
    xcol, ycol = num_cols[:2]
    figs: List[Path] = []

    if stage_col and stage_col in df.columns:
        stages = list(df[stage_col].dropna().unique())[:4]
        n_stage = len(stages)
        ncols = min(2, n_stage)
        nrows = int(np.ceil(n_stage / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 3.0))
        axes = np.atleast_1d(axes).flatten()
        for ax, stage in zip(axes, stages):
            sdf = df[df[stage_col] == stage]
            colors = sdf[label_col] if label_col and label_col in sdf else None
            scatter = ax.scatter(sdf[xcol], sdf[ycol], c=colors, cmap="viridis", alpha=0.85, edgecolors="none")
            ax.set_title(f"{name.upper()} – {stage}")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            if colors is not None:
                legend = ax.legend(*scatter.legend_elements(), title=label_col)
                ax.add_artist(legend)
        figs.extend(exporter.export(fig, out_dir, f"{name}_scores_stage"))
        plt.close(fig)

    # Label-colored view
    fig, ax = plt.subplots()
    colors = df[label_col] if label_col and label_col in df else None
    scatter = ax.scatter(df[xcol], df[ycol], c=colors, cmap="viridis", alpha=0.85, edgecolors="none")
    ax.set_title(f"{name.upper()} scores")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    if colors is not None:
        legend = ax.legend(*scatter.legend_elements(), title=label_col)
        ax.add_artist(legend)
    figs.extend(exporter.export(fig, out_dir, f"{name}_scores_label"))
    plt.close(fig)

    # Batch markers
    if batch_col and batch_col in df:
        fig, ax = plt.subplots()
        batches = df[batch_col].astype(str)
        uniq = batches.unique()
        markers = ["o", "s", "^", "D", "P", "X", "*", "v"]
        for idx, b in enumerate(uniq):
            mask = batches == b
            ax.scatter(df.loc[mask, xcol], df.loc[mask, ycol], marker=markers[idx % len(markers)], label=str(b), alpha=0.85)
        ax.set_title(f"{name.upper()} scores by batch")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.legend()
        figs.extend(exporter.export(fig, out_dir, f"{name}_scores_batch"))
        plt.close(fig)

    return figs


def _plot_loadings(loadings: pd.DataFrame, name: str, out_dir: Path, top_k: int = 12) -> List[Path]:
    import matplotlib.pyplot as plt

    exporter = FigureExporter(style=FigureStyle.JOSS, size_preset="single", formats=("png",))
    if loadings.empty:
        return []
    num_cols = _numeric_columns(loadings)
    if "component" in loadings.columns:
        num_cols = [c for c in num_cols if c != "component"]
    if not num_cols:
        return []
    first_row = loadings.iloc[0][num_cols]
    vals = first_row.to_numpy(dtype=float)
    names = np.array(num_cols)
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    vals = vals[order]
    names = names[order]

    fig, ax = plt.subplots()
    ax.bar(range(len(vals)), vals, color="#2563eb")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Loading weight")
    ax.set_title(f"{name.upper()} loadings (top {top_k})")
    figs = exporter.export(fig, out_dir, f"{name}_loadings_pc1")
    plt.close(fig)
    return figs


def _plot_heatmap(loadings: pd.DataFrame, name: str, out_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    exporter = FigureExporter(style=FigureStyle.JOSS, size_preset="double", formats=("png",))
    num_cols = _numeric_columns(loadings)
    if "component" in loadings.columns:
        num_cols = [c for c in num_cols if c != "component"]
    mat = loadings[num_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    ax.set_xlabel("Features")
    ax.set_ylabel("Components")
    ax.set_title(f"{name.upper()} loadings heatmap")
    fig.colorbar(im, ax=ax)
    figs = exporter.export(fig, out_dir, f"{name}_loadings_heatmap")
    plt.close(fig)
    return figs


def _write_variance_table(summary: Dict[str, Any], out_path: Path) -> Optional[Path]:
    ev = summary.get("explained_variance") or summary.get("explained_variance_sum")
    if ev is None:
        return None
    records: List[Dict[str, Any]] = []
    if isinstance(ev, (list, tuple, np.ndarray)):
        for i, val in enumerate(ev):
            records.append({"component": i + 1, "explained_variance": float(val)})
    else:
        records.append({"component": 1, "explained_variance": float(ev)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    return out_path


def build_multivariate_section(context) -> Dict[str, Any]:
    """Build multivariate section context and ensure figures/tables exist."""

    run_dir: Path = Path(getattr(context, "run_dir", "."))
    mv_root = run_dir / "multivariate"
    if not mv_root.exists():
        return {"methods": [], "qc": {}}

    figures_dir = run_dir / "figures" / "multivariate"
    tables_dir = run_dir / "tables" / "multivariate"
    methods: List[Dict[str, Any]] = []

    qc_summary = {}
    trust = getattr(context, "trust_outputs", {}) or {}
    if isinstance(trust, dict):
        qc_summary = trust.get("qc_summary", {})

    for method_dir in sorted([p for p in mv_root.iterdir() if p.is_dir()]):
        method = method_dir.name
        scores_path = method_dir / "scores.csv"
        loadings_path = method_dir / "loadings.csv"
        summary_path = method_dir / "summary.json"

        scores_df = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
        loadings_df = pd.read_csv(loadings_path) if loadings_path.exists() else pd.DataFrame()
        summary = _read_json(summary_path)

        label_info = _detect_label_cols(scores_df) if not scores_df.empty else {"label": None, "batch": None, "stage": None}
        figs: List[Path] = []
        if not scores_df.empty:
            figs.extend(
                _plot_scores(
                    scores_df,
                    method,
                    label_info.get("label"),
                    label_info.get("batch"),
                    label_info.get("stage"),
                    figures_dir,
                )
            )
        if not loadings_df.empty:
            figs.extend(_plot_loadings(loadings_df, method, figures_dir))
            if loadings_df.shape[0] > 1:
                figs.extend(_plot_heatmap(loadings_df, method, figures_dir))

        variance_path = _write_variance_table(summary, tables_dir / f"{method}_variance.csv") if summary else None

        methods.append(
            {
                "name": method,
                "summary": summary,
                "scores_path": scores_path if scores_path.exists() else None,
                "loadings_path": loadings_path if loadings_path.exists() else None,
                "variance_path": variance_path,
                "figures": [p.relative_to(run_dir) for p in figs],
                "head": scores_df.head(5).to_dict(orient="records") if not scores_df.empty else [],
            }
        )

    return {"methods": methods, "qc": qc_summary}


__all__ = ["build_multivariate_section", "MultivariateMethodArtifacts"]
