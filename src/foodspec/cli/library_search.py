#!/usr/bin/env python3

"""
CLI for spectral library search.

Reads a query CSV and a library CSV with numeric wavenumber columns and an optional label column.
Outputs top-k matches with scores and optional overlay plot saved to a file.

Usage:
    foodspec-library-search --query query.csv --library lib.csv --label-col label --k 5 --metric cosine --overlay-out overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from foodspec.library_search import overlay_plot, search_library


def _extract_wavenumber_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        try:
            float(c)
            cols.append(c)
        except Exception:
            continue
    if len(cols) < 3:
        raise ValueError("CSV must contain at least three numeric wavenumber columns.")
    return cols


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Spectral library search (top-k).")
    p.add_argument("--query", required=True, help="Path to query CSV (one row of spectra).")
    p.add_argument("--library", required=True, help="Path to library CSV (rows of spectra).")
    p.add_argument("--label-col", default="label", help="Label column name in library CSV (optional).")
    p.add_argument("--k", type=int, default=5, help="Top-k matches to return.")
    p.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "pearson", "euclidean", "sid", "sam"],
        help="Similarity metric.",
    )
    p.add_argument("--overlay-out", default=None, help="Path to save overlay plot (optional).")
    args = p.parse_args(argv)

    qdf = pd.read_csv(args.query)
    ldf = pd.read_csv(args.library)
    q_cols = _extract_wavenumber_columns(qdf)
    l_cols = _extract_wavenumber_columns(ldf)
    if set(q_cols) != set(l_cols):
        raise ValueError("Query and library must have the same wavenumber columns.")
    wn = np.array(sorted([float(c) for c in q_cols]))
    # Reorder columns consistently
    q_cols_sorted = [str(c) for c in sorted([float(c) for c in q_cols])]
    l_cols_sorted = q_cols_sorted

    query = qdf[q_cols_sorted].to_numpy(dtype=float)
    if query.shape[0] != 1:
        raise ValueError("Query CSV should contain exactly one row of spectral intensities.")
    query_vec = query[0]
    lib = ldf[l_cols_sorted].to_numpy(dtype=float)
    labels = ldf[args.label_col].tolist() if args.label_col in ldf.columns else None

    matches = search_library(query_vec, lib, labels=labels, k=args.k, metric=args.metric)
    print("Top matches:")
    for m in matches:
        print(f"- {m.label}: score={m.score:.4f} confidence={m.confidence:.2f} metric={m.metric}")

    if args.overlay_out:
        overlay = overlay_plot(
            query_vec, wn, [(m.label, lib[m.index]) for m in matches], title=f"Top-{args.k} ({args.metric})"
        )
        out_path = Path(args.overlay_out)
        overlay.savefig(out_path)
        print(f"Saved overlay plot to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
