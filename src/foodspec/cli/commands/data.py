"""Data management commands: libraries, CSV conversion, similarity search."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from foodspec.core.api import FoodSpec
from foodspec.data.libraries import load_library
from foodspec.io import create_library, load_csv_spectra
from foodspec.library_search import overlay_plot, search_library
from foodspec.logging_utils import get_logger
from foodspec.model_registry import load_model as registry_load_model

logger = get_logger(__name__)

data_app = typer.Typer(help="Data management commands")


@data_app.command("csv-to-library")
def csv_to_library(
    csv_path: str = typer.Argument(..., help="Input CSV file with spectra."),
    output_hdf5: str = typer.Argument(..., help="Output HDF5 library path (will be created or overwritten)."),
    format: str = typer.Option(
        "wide",
        "--format",
        help="CSV layout: 'wide' (one column per spectrum) or 'long' (tidy format).",
        case_sensitive=False,
    ),
    modality: str = typer.Option(
        "raman",
        "--modality",
        help="Spectroscopy modality tag (e.g. 'raman', 'ftir').",
    ),
    wavenumber_column: str = typer.Option(
        "wavenumber",
        "--wavenumber-column",
        help="Name of the wavenumber column.",
    ),
    sample_id_column: str = typer.Option(
        "sample_id",
        "--sample-id-column",
        help="For 'long' format: sample identifier column.",
    ),
    intensity_column: str = typer.Option(
        "intensity",
        "--intensity-column",
        help="For 'long' format: intensity column.",
    ),
    label_column: str = typer.Option(
        "",
        "--label-column",
        help="Optional label column name (e.g. oil_type).",
    ),
):
    """
    Convert a CSV file of spectra into an HDF5 library usable by foodspec workflows.
    """
    label_column = label_column or None
    logger.info("Loading CSV spectra from %s", csv_path)
    ds = load_csv_spectra(
        csv_path=csv_path,
        format=format,
        wavenumber_column=wavenumber_column,
        sample_id_column=sample_id_column,
        intensity_column=intensity_column,
        label_column=label_column,
        modality=modality,
    )

    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving HDF5 library to %s", output_hdf5)
    create_library(path=output_hdf5, spectra=ds)
    logger.info("Done. Library contains %s spectra.", len(ds))


@data_app.command("library-search")
def library_search_command(
    query: str = typer.Option(..., help="Path to query CSV (one spectrum row)."),
    library: str = typer.Option(..., help="Path to library CSV (rows of spectra)."),
    label_col: str = typer.Option("label", help="Label column in library CSV."),
    k: int = typer.Option(5, help="Top-k matches to return."),
    metric: str = typer.Option("cosine", help="Similarity metric: cosine|pearson|euclidean|sid|sam"),
    overlay_out: Optional[str] = typer.Option(None, help="Save overlay plot (optional)."),
):
    """Spectral library search (unified CLI)."""
    qdf = pd.read_csv(query)
    ldf = pd.read_csv(library)

    def _num_cols(df: pd.DataFrame) -> list[str]:
        cols: list[str] = []
        for c in df.columns:
            try:
                float(c)
                cols.append(c)
            except Exception:
                continue
        if len(cols) < 3:
            raise typer.BadParameter("CSV must contain numeric wavenumber columns.")
        return cols

    q_cols = _num_cols(qdf)
    l_cols = _num_cols(ldf)
    if set(q_cols) != set(l_cols):
        raise typer.BadParameter("Query and library must have the same wavenumber columns.")
    wn = np.array(sorted([float(c) for c in q_cols]))
    cols_sorted = [str(c) for c in sorted([float(c) for c in q_cols])]
    query_vec = qdf[cols_sorted].to_numpy(dtype=float)
    if query_vec.shape[0] != 1:
        raise typer.BadParameter("Query CSV should contain exactly one spectrum row.")
    lib = ldf[cols_sorted].to_numpy(dtype=float)
    labels = ldf[label_col].tolist() if label_col in ldf.columns else None

    matches = search_library(query_vec[0], lib, labels=labels, k=k, metric=metric)
    print("Top matches:")
    for m in matches:
        print(f"- {m.label}: score={m.score:.4f} confidence={m.confidence:.2f} metric={m.metric}")
    if overlay_out:
        fig = overlay_plot(query_vec[0], wn, [(m.label, lib[m.index]) for m in matches], title=f"Top-{k} ({metric})")
        out_path = Path(overlay_out)
        fig.savefig(out_path)
        print(f"Saved overlay plot to {out_path}")


@data_app.command("library-auth")
def library_auth_command(
    query_hdf5: str = typer.Argument(..., help="Input HDF5 with query spectra."),
    library_hdf5: str = typer.Argument(..., help="Reference HDF5 library."),
    metric: str = typer.Option("cosine", help="Distance metric: euclidean/cosine/pearson/sid/sam."),
    top_k: int = typer.Option(5, help="Top-k matches per query."),
    output_dir: str = typer.Option("./out", help="Base output directory."),
):
    """Run library-based authentication: similarity search + overlay.

    Produces a similarity table and an overlay plot (first query vs top match)
    in the output diagnostics folder.
    """
    ds_q = load_library(query_hdf5)
    ds_lib = load_library(library_hdf5)
    fs = FoodSpec(ds_q)
    fs.library_similarity(ds_lib, metric=metric, top_k=top_k)
    out = fs.export(output_dir)
    typer.echo(f"Similarity table saved under {out}/diagnostics/similarity_table.csv")
    typer.echo(f"Overlay figure saved under {out}/diagnostics/overlay_query0_top1.png")


@data_app.command("model-info")
def model_info_command(
    path: str = typer.Argument(..., help="Base path of saved model (without extension)."),
):
    """Inspect saved model metadata."""
    model_base = Path(path)
    joblib_path = model_base.with_suffix(".joblib")
    json_path = model_base.with_suffix(".json")
    if not joblib_path.exists() or not json_path.exists():
        typer.echo("Model files not found (expected .joblib and .json).", err=True)
        raise typer.Exit(code=1)
    try:
        _, meta = registry_load_model(path)
    except Exception as exc:  # pragma: no cover - defensive
        typer.echo(f"Failed to load model metadata: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Name: {meta.name}")
    typer.echo(f"Version: {meta.version}")
    typer.echo(f"Foodspec version: {meta.foodspec_version}")
    typer.echo(f"Created at: {meta.created_at}")
    typer.echo("Extra:")
    typer.echo(json.dumps(meta.extra, indent=2))
