"""Consistent plotting API for FoodSpec visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foodspec._version import __version__
from foodspec.qc import control_charts as qc_stats
from foodspec.reporting.schema import RunBundle
from foodspec.stats.clustering import hierarchical_cluster
from foodspec.viz import control_charts as qc_viz
from foodspec.viz import drift as drift_viz
from foodspec.viz import interpretability as interp_viz
from foodspec.viz import provenance as prov_viz
from foodspec.viz.classification import plot_reliability_diagram as _plot_reliability_diagram
from foodspec.viz.clustering import plot_dendrogram as _plot_dendrogram
from foodspec.viz.distribution import plot_probability_plot as _plot_probability_plot
from foodspec.viz.embeddings import plot_pca_scatter as _plot_pca_scatter
from foodspec.viz.embeddings import plot_umap_scatter as _plot_umap_scatter
from foodspec.viz.save import save_figure
from foodspec.viz.style import apply_style


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _bundle_meta(bundle: Optional[RunBundle]) -> Dict[str, Any]:
    if bundle is None:
        return {}
    return {
        "run_id": bundle.run_id,
        "git_commit": bundle.manifest.get("git_commit"),
        "timestamp": bundle.manifest.get("timestamp"),
    }


def _figure_base(outdir: Optional[Path | str], name: str) -> Optional[Path]:
    if outdir is None:
        return None
    return Path(outdir) / "figures" / name


def _load_table(run_dir: Path, candidates: Sequence[str]) -> Optional[pd.DataFrame]:
    for name in candidates:
        path = run_dir / "tables" / name
        if path.exists():
            return pd.read_csv(path)
    return None


def _spectra_from_table(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    numeric_cols = []
    wns = []
    for col in df.columns:
        try:
            val = float(col)
            numeric_cols.append(col)
            wns.append(val)
        except ValueError:
            continue
    if not numeric_cols:
        raise ValueError("No numeric spectral columns found.")
    spectra = df[numeric_cols].astype(float).to_numpy()
    return np.asarray(wns, dtype=float), spectra


def _extract_spectra(bundle: Optional[RunBundle], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bundle is None:
        wn = np.linspace(400, 1800, 200)
        raw = np.sin(wn / 200.0) + _rng(seed).normal(0, 0.02, size=wn.size)
        processed = raw - raw.mean() * 0.1
        return wn, raw, processed
    df_raw = _load_table(bundle.run_dir, ["spectra_raw.csv", "raw_spectra.csv"])
    df_proc = _load_table(bundle.run_dir, ["spectra_processed.csv", "processed_spectra.csv"])
    if df_raw is not None and df_proc is not None:
        wn_raw, spectra_raw = _spectra_from_table(df_raw)
        wn_proc, spectra_proc = _spectra_from_table(df_proc)
        raw = spectra_raw[0]
        processed = spectra_proc[0]
        wn = wn_raw if wn_raw.size else wn_proc
        return wn, raw, processed
    wn = np.linspace(400, 1800, 200)
    raw = np.sin(wn / 200.0) + _rng(seed).normal(0, 0.02, size=wn.size)
    processed = raw - raw.mean() * 0.1
    return wn, raw, processed


def _extract_features(bundle: Optional[RunBundle], seed: int) -> np.ndarray:
    if bundle is None:
        return _rng(seed).normal(size=(50, 5))
    df = _load_table(bundle.run_dir, ["features.csv", "feature_matrix.csv"])
    if df is not None:
        return df.select_dtypes(include=["number"]).to_numpy()
    return _rng(seed).normal(size=(50, 5))


def _extract_predictions(bundle: Optional[RunBundle], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bundle is None:
        y_true = _rng(seed).integers(0, 2, size=80)
        y_prob = _rng(seed).random(80)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_true, y_pred, y_prob
    df = _load_table(bundle.run_dir, ["predictions.csv"])
    if df is None:
        y_true = _rng(seed).integers(0, 2, size=80)
        y_prob = _rng(seed).random(80)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_true, y_pred, y_prob
    y_true = df.get("y_true") or df.get("true") or df.get("label")
    y_pred = df.get("y_pred") or df.get("pred") or df.get("prediction")
    if y_true is None or y_pred is None:
        y_true = _rng(seed).integers(0, 2, size=len(df))
        y_pred = _rng(seed).integers(0, 2, size=len(df))
    y_prob = df.get("proba")
    if y_prob is None:
        prob_cols = [c for c in df.columns if str(c).startswith("prob_")]
        if prob_cols:
            y_prob = df[prob_cols].max(axis=1)
        else:
            y_prob = _rng(seed).random(len(df))
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(y_prob, dtype=float)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence[int]] = None) -> np.ndarray:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_idx[yt], label_to_idx[yp]] += 1
    return cm


def plot_raw_processed_overlay(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    """Plot raw vs processed spectral overlays."""
    apply_style()
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    wn = np.asarray(payload.get("wavenumbers", []), dtype=float)
    raw = np.asarray(payload.get("raw", []), dtype=float)
    processed = np.asarray(payload.get("processed", []), dtype=float)
    if wn.size == 0 or raw.size == 0 or processed.size == 0:
        wn, raw, processed = _extract_spectra(bundle, seed)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(wn, raw, label="raw", alpha=0.8)
    ax.plot(wn, processed, label="processed", alpha=0.8)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Raw vs Processed Overlay")
    ax.legend()
    ax.invert_xaxis()

    base = _figure_base(outdir, name or "raw_processed_overlay")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Raw vs processed overlay",
                "inputs": {"n_points": len(wn)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
                "preprocessing": payload.get("preprocessing"),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_spectra_heatmap(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    """Plot spectra matrix heatmap."""
    apply_style()
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    matrix = np.asarray(payload.get("matrix", []), dtype=float)
    if matrix.size == 0:
        if bundle is not None:
            df = _load_table(bundle.run_dir, ["spectra_matrix.csv", "spectra.csv"])
            if df is not None:
                matrix = df.select_dtypes(include=["number"]).to_numpy()
        if matrix.size == 0:
            matrix = _rng(seed).normal(size=(40, 200))
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Sample")
    ax.set_title("Spectra Heatmap")

    base = _figure_base(outdir, name or "spectra_heatmap")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Spectra matrix heatmap",
                "inputs": {"shape": list(matrix.shape)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
                "preprocessing": payload.get("preprocessing"),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_correlation_heatmap(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    """Plot correlation heatmap."""
    apply_style()
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    matrix = np.asarray(payload.get("matrix", []), dtype=float)
    if matrix.size == 0:
        matrix = _extract_features(bundle, seed)
    corr = np.corrcoef(matrix, rowvar=False)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation Heatmap")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")

    base = _figure_base(outdir, name or "correlation_heatmap")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Correlation heatmap",
                "inputs": {"shape": list(matrix.shape)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_pca_scatter(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    """Plot PCA scatter (delegates to embeddings module)."""
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, RunBundle):
        payload["X"] = _extract_features(data_bundle, seed)
    elif isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    return _plot_pca_scatter(payload, outdir=outdir, name=name, fmt=fmt, dpi=dpi, seed=seed)


def plot_umap_scatter(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    """Plot UMAP scatter (delegates to embeddings module)."""
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, RunBundle):
        payload["X"] = _extract_features(data_bundle, seed)
    elif isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    return _plot_umap_scatter(payload, outdir=outdir, name=name, fmt=fmt, dpi=dpi, seed=seed)


def plot_confusion_matrix(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    """Plot confusion matrix counts and normalized."""
    apply_style()
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    y_true = np.asarray(payload.get("y_true", []), dtype=int)
    y_pred = np.asarray(payload.get("y_pred", []), dtype=int)
    if y_true.size == 0 or y_pred.size == 0:
        y_true, y_pred, _ = _extract_predictions(bundle, seed)
    labels = sorted(set(y_true) | set(y_pred))
    cm = _confusion_matrix(y_true, y_pred, labels)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Counts)")

    base_name = name or "confusion_matrix"
    base = _figure_base(outdir, f"{base_name}_counts")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Confusion matrix counts",
                "inputs": {"n_samples": len(y_true)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )

    # Normalized
    if outdir is not None:
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, where=row_sums != 0)
        fig_norm, ax_norm = plt.subplots(figsize=(4, 3))
        im_norm = ax_norm.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        fig_norm.colorbar(im_norm, ax=ax_norm)
        ax_norm.set_xticks(range(len(labels)))
        ax_norm.set_xticklabels(labels)
        ax_norm.set_yticks(range(len(labels)))
        ax_norm.set_yticklabels(labels)
        ax_norm.set_xlabel("Predicted")
        ax_norm.set_ylabel("True")
        ax_norm.set_title("Confusion Matrix (Normalized)")
        save_figure(
            fig_norm,
            _figure_base(outdir, f"{base_name}_normalized"),
            metadata={
                "description": "Confusion matrix normalized",
                "inputs": {"n_samples": len(y_true)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
        plt.close(fig_norm)
    return fig


def plot_reliability_diagram(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    """Plot reliability diagram."""
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, RunBundle):
        y_true, _, y_prob = _extract_predictions(data_bundle, seed)
        payload = {"y_true": y_true, "y_prob": y_prob}
    elif isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    return _plot_reliability_diagram(
        payload,
        outdir=outdir,
        name=name,
        fmt=fmt,
        dpi=dpi,
        seed=seed,
    )


def plot_workflow_dag(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    fig = prov_viz.plot_workflow_dag(data_bundle, seed=seed)
    base = _figure_base(outdir, name or "workflow_dag")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Workflow DAG",
                "inputs": {"steps": len(getattr(bundle, "manifest", {}).get("protocol_snapshot", {}).get("steps", []))},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_parameter_map(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    fig = prov_viz.plot_parameter_map(data_bundle, seed=seed)
    base = _figure_base(outdir, name or "parameter_map")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Parameter map",
                "inputs": {"source": "protocol_snapshot"},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_data_lineage(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    fig = prov_viz.plot_data_lineage(data_bundle, seed=seed)
    base = _figure_base(outdir, name or "data_lineage")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Data lineage summary",
                "inputs": {"input_count": len(bundle.manifest.get("inputs", [])) if bundle else 0},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_reproducibility_badge(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    fig = prov_viz.plot_reproducibility_badge(data_bundle, seed=seed)
    base = _figure_base(outdir, name or "reproducibility_badge")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Reproducibility badge",
                "inputs": {"run_id": bundle.run_id if bundle else "run"},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_batch_drift(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    names = payload.get("batch_names") or [f"b{i + 1}" for i in range(5)]
    scores = payload.get("drift_scores") or _rng(seed).random(len(names))
    fig = drift_viz.plot_batch_drift(names, scores, seed=seed)
    base = _figure_base(outdir, name or "batch_drift")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Batch drift plot",
                "inputs": {"n_batches": len(names)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_stage_difference_spectra(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    wn = payload.get("wavenumbers") or np.linspace(400, 1800, 200)
    stages = payload.get("stage_spectra")
    if stages is None:
        base = np.sin(np.asarray(wn) / 200.0)
        stages = {"baseline": base, "stage1": base + 0.05}
    fig = drift_viz.plot_stage_difference_spectra(wn, stages, seed=seed)
    base = _figure_base(outdir, name or "stage_difference_spectra")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Stage-wise difference spectra",
                "inputs": {"n_stages": len(stages)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
                "preprocessing": payload.get("preprocessing"),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_replicate_similarity(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    matrix = np.asarray(payload.get("similarity_matrix", []), dtype=float)
    if matrix.size == 0:
        matrix = np.eye(5)
    fig = drift_viz.plot_replicate_similarity(matrix, seed=seed)
    base = _figure_base(outdir, name or "replicate_similarity")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Replicate similarity matrix",
                "inputs": {"shape": list(matrix.shape)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_temporal_drift(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    times = payload.get("time_points") or np.arange(6)
    values = payload.get("drift_values") or _rng(seed).random(len(times))
    fig = drift_viz.plot_temporal_drift(times, values, seed=seed)
    base = _figure_base(outdir, name or "temporal_drift")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Temporal drift trend",
                "inputs": {"n_points": len(times)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_importance_overlay(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    wn = payload.get("wavenumbers") or np.linspace(400, 1800, 200)
    spectrum = payload.get("spectrum") or np.sin(np.asarray(wn) / 200.0)
    importance = payload.get("importance") or _rng(seed).random(len(wn))
    fig = interp_viz.plot_importance_overlay(wn, spectrum, importance, seed=seed)
    base = _figure_base(outdir, name or "importance_overlay")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Importance overlay on spectra",
                "inputs": {"n_points": len(wn)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
                "preprocessing": payload.get("preprocessing"),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_marker_bands(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    wn = payload.get("wavenumbers") or np.linspace(400, 1800, 200)
    bands = payload.get("bands") or [(600, 650), (1200, 1260)]
    fig = interp_viz.plot_marker_bands(wn, bands, seed=seed)
    base = _figure_base(outdir, name or "marker_bands")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Marker band plot",
                "inputs": {"n_bands": len(bands)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
                "preprocessing": payload.get("preprocessing"),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_coefficient_heatmap(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    coef = np.asarray(payload.get("coefficients", _rng(seed).normal(size=(3, 6))), dtype=float)
    names = payload.get("feature_names") or [f"f{i + 1}" for i in range(coef.shape[1])]
    fig = interp_viz.plot_coefficient_heatmap(coef, names, seed=seed)
    base = _figure_base(outdir, name or "coefficient_heatmap")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Coefficient heatmap",
                "inputs": {"shape": list(coef.shape)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_feature_stability(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    matrix = np.asarray(payload.get("stability_matrix", _rng(seed).random((5, 6))), dtype=float)
    names = payload.get("feature_names") or [f"f{i + 1}" for i in range(matrix.shape[1])]
    fig = interp_viz.plot_feature_stability(matrix, names, seed=seed)
    base = _figure_base(outdir, name or "feature_stability")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Feature stability map",
                "inputs": {"shape": list(matrix.shape)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_confidence_map(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    probs_matrix = payload.get("probabilities")
    if probs_matrix is not None:
        apply_style()
        mat = np.asarray(probs_matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Class")
        ax.set_ylabel("Sample")
        ax.set_title("Confidence Map")
        probs = mat.max(axis=1) if mat.size else np.asarray([])
    else:
        from foodspec.viz.uncertainty import plot_confidence_map as _plot_conf_map

        probs = np.asarray(payload.get("confidences", []), dtype=float)
        if probs.size == 0:
            _, _, probs = _extract_predictions(bundle, seed)
        fig = _plot_conf_map(probs, save_path=None, dpi=dpi)
    base = _figure_base(outdir, name or "confidence_map")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Confidence map",
                "inputs": {"n_samples": len(probs)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_conformal_set_sizes(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    from foodspec.viz.uncertainty import plot_set_size_distribution as _plot_set_size

    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    set_sizes = np.asarray(payload.get("set_sizes", []), dtype=int)
    if set_sizes.size == 0:
        set_sizes = _rng(seed).integers(1, 4, size=50)
    fig = _plot_set_size(set_sizes, save_path=None, dpi=dpi)
    base = _figure_base(outdir, name or "conformal_set_sizes")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Conformal set size distribution",
                "inputs": {"n_samples": len(set_sizes)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_coverage_efficiency(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    from foodspec.viz.uncertainty import plot_coverage_efficiency as _plot_cov_eff

    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    alphas = np.asarray(payload.get("alphas", []), dtype=float)
    coverages = np.asarray(payload.get("coverages", []), dtype=float)
    sizes = np.asarray(payload.get("avg_sizes", []), dtype=float)
    if alphas.size == 0:
        alphas = np.linspace(0.05, 0.25, 5)
        coverages = 1 - alphas
        sizes = np.linspace(1.2, 2.5, 5)
    fig = _plot_cov_eff(alphas, coverages, sizes, save_path=None, dpi=dpi)
    base = _figure_base(outdir, name or "coverage_efficiency")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Coverage vs efficiency",
                "inputs": {"n_points": len(alphas)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_abstention_distribution(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    from foodspec.viz.uncertainty import plot_abstention_distribution as _plot_abstention

    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    payload = data_bundle if isinstance(data_bundle, Mapping) else {}
    abstain_flags = np.asarray(payload.get("abstain_flags", []), dtype=int)
    if abstain_flags.size == 0:
        abstain_flags = _rng(seed).integers(0, 2, size=80)
    fig = _plot_abstention(abstain_flags, save_path=None, dpi=dpi)
    base = _figure_base(outdir, name or "abstention_distribution")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Abstention distribution",
                "inputs": {"n_samples": len(abstain_flags)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_xbar_r_chart(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    values = np.asarray(payload.get("values", []), dtype=float)
    subgroup_size = int(payload.get("subgroup_size", 5))
    if values.size == 0:
        values = _rng(seed).normal(loc=0.0, scale=1.0, size=subgroup_size * 8)
    result = qc_stats.xbar_r_chart(values, subgroup_size=subgroup_size)
    fig = qc_viz.plot_control_chart_group(result, title="X-bar / R")
    base = _figure_base(outdir, name or "xbar_r_chart")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "X-bar and R control chart",
                "inputs": {"subgroup_size": subgroup_size},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(data_bundle if isinstance(data_bundle, RunBundle) else None),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_xbar_s_chart(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    values = np.asarray(payload.get("values", []), dtype=float)
    subgroup_size = int(payload.get("subgroup_size", 5))
    if values.size == 0:
        values = _rng(seed).normal(loc=0.0, scale=1.0, size=subgroup_size * 8)
    result = qc_stats.xbar_s_chart(values, subgroup_size=subgroup_size)
    fig = qc_viz.plot_control_chart_group(result, title="X-bar / S")
    base = _figure_base(outdir, name or "xbar_s_chart")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "X-bar and S control chart",
                "inputs": {"subgroup_size": subgroup_size},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(data_bundle if isinstance(data_bundle, RunBundle) else None),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_individuals_mr_chart(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    values = np.asarray(payload.get("values", []), dtype=float)
    if values.size == 0:
        values = _rng(seed).normal(loc=0.0, scale=1.0, size=40)
    result = qc_stats.individuals_mr_chart(values)
    fig = qc_viz.plot_control_chart_group(result, title="Individuals / MR")
    base = _figure_base(outdir, name or "individuals_mr_chart")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Individuals and moving range chart",
                "inputs": {"n_points": len(values)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(data_bundle if isinstance(data_bundle, RunBundle) else None),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_cusum_chart(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    values = np.asarray(payload.get("values", []), dtype=float)
    if values.size == 0:
        values = _rng(seed).normal(size=50)
    k = float(payload.get("k", 0.5))
    h = float(payload.get("h", 5.0))
    result = qc_stats.cusum_chart(values, k=k, h=h)
    fig = qc_viz.plot_cusum(result["pos"], result["neg"], result["h"], title="CUSUM")
    base = _figure_base(outdir, name or "cusum_chart")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "CUSUM chart",
                "inputs": {"n_points": len(values), "k": k, "h": h},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(data_bundle if isinstance(data_bundle, RunBundle) else None),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_ewma_chart(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    values = np.asarray(payload.get("values", []), dtype=float)
    if values.size == 0:
        values = _rng(seed).normal(size=50)
    lam = float(payload.get("lam", 0.2))
    l_val = float(payload.get("l", 3.0))
    result = qc_stats.ewma_chart(values, lam=lam, l_limit=l_val)
    fig = qc_viz.plot_ewma(result["ewma"], result["lcl"], result["ucl"], title="EWMA")
    base = _figure_base(outdir, name or "ewma_chart")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "EWMA chart",
                "inputs": {"n_points": len(values), "lam": lam, "l": l_val},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(data_bundle if isinstance(data_bundle, RunBundle) else None),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_levey_jennings_chart(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    values = np.asarray(payload.get("values", []), dtype=float)
    if values.size == 0:
        values = _rng(seed).normal(size=50)
    result = qc_stats.levey_jennings(values)
    fig = qc_viz.plot_control_chart(result, title="Levey-Jennings")
    base = _figure_base(outdir, name or "levey_jennings_chart")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Levey-Jennings chart",
                "inputs": {"n_points": len(values)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(data_bundle if isinstance(data_bundle, RunBundle) else None),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_probability_plot(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    values = np.asarray(payload.get("values", []), dtype=float)
    if values.size == 0:
        values = _rng(seed).normal(size=60)
    dist = str(payload.get("dist", "normal"))
    fig = _plot_probability_plot(values, dist=dist, title=f"{dist.title()} Probability Plot")
    base = _figure_base(outdir, name or "probability_plot")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Probability plot",
                "inputs": {"n_points": len(values), "dist": dist},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(data_bundle if isinstance(data_bundle, RunBundle) else None),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_dendrogram(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    X = np.asarray(payload.get("X", []), dtype=float)
    if X.size == 0:
        X = _extract_features(bundle, seed)
    result = hierarchical_cluster(X, n_clusters=2)
    if result.linkage_matrix is None:
        raise ValueError("Linkage matrix unavailable (scipy required).")
    fig = _plot_dendrogram(result.linkage_matrix, title="Hierarchical Dendrogram")
    base = _figure_base(outdir, name or "dendrogram")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Hierarchical clustering dendrogram",
                "inputs": {"n_samples": X.shape[0]},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_pareto_chart(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    categories = payload.get("categories")
    if categories is None:
        categories = [f"type_{i % 3}" for i in range(30)]
    counts = qc_stats.pareto_counts(list(categories))
    fig = qc_viz.plot_pareto(counts, title="Pareto Chart")
    base = _figure_base(outdir, name or "pareto_chart")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Pareto chart",
                "inputs": {"n_items": len(categories)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


def plot_runs_analysis(
    data_bundle: RunBundle | Mapping[str, Any],
    *,
    outdir: Path | None = None,
    name: str | None = None,
    fmt=("png", "svg"),
    dpi: int = 300,
    seed: int = 0,
) -> plt.Figure:
    payload: Dict[str, Any] = {}
    bundle = data_bundle if isinstance(data_bundle, RunBundle) else None
    if isinstance(data_bundle, Mapping):
        payload.update(data_bundle)
    values = np.asarray(payload.get("values", []), dtype=float)
    if values.size == 0:
        values = _rng(seed).normal(size=40)
    fig = qc_viz.plot_runs(values, title="Runs Analysis")
    base = _figure_base(outdir, name or "runs_analysis")
    if base is not None:
        save_figure(
            fig,
            base,
            metadata={
                "description": "Runs analysis plot",
                "inputs": {"n_points": len(values)},
                "code_version": __version__,
                "seed": seed,
                **_bundle_meta(bundle),
            },
            fmt=fmt,
            dpi=dpi,
        )
    return fig


__all__ = [
    "plot_raw_processed_overlay",
    "plot_spectra_heatmap",
    "plot_correlation_heatmap",
    "plot_pca_scatter",
    "plot_umap_scatter",
    "plot_confusion_matrix",
    "plot_reliability_diagram",
    "plot_workflow_dag",
    "plot_parameter_map",
    "plot_data_lineage",
    "plot_reproducibility_badge",
    "plot_batch_drift",
    "plot_stage_difference_spectra",
    "plot_replicate_similarity",
    "plot_temporal_drift",
    "plot_importance_overlay",
    "plot_marker_bands",
    "plot_coefficient_heatmap",
    "plot_feature_stability",
    "plot_confidence_map",
    "plot_conformal_set_sizes",
    "plot_coverage_efficiency",
    "plot_abstention_distribution",
    "plot_xbar_r_chart",
    "plot_xbar_s_chart",
    "plot_individuals_mr_chart",
    "plot_cusum_chart",
    "plot_ewma_chart",
    "plot_levey_jennings_chart",
    "plot_probability_plot",
    "plot_dendrogram",
    "plot_pareto_chart",
    "plot_runs_analysis",
]
