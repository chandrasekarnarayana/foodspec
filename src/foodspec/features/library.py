"""Similarity search over spectra or features (top-k).

Provides a simple kNN-style search that returns a similarity table
with distances using chosen metrics, and optional overlay plots
using the existing visualization utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.stats.distances import compute_distances

__all__ = ["LibraryIndex", "similarity_search"]


@dataclass
class LibraryIndex:
    X: np.ndarray
    metadata: pd.DataFrame
    wavenumbers: Optional[np.ndarray] = None

    @classmethod
    def from_dataset(cls, ds: FoodSpectrumSet) -> "LibraryIndex":
        return cls(X=np.asarray(ds.x, dtype=float), metadata=ds.metadata.copy(deep=True), wavenumbers=ds.wavenumbers)

    @classmethod
    def from_features(cls, features: pd.DataFrame, metadata: Optional[pd.DataFrame] = None) -> "LibraryIndex":
        meta = metadata.copy(deep=True) if metadata is not None else pd.DataFrame(index=np.arange(features.shape[0]))
        return cls(X=np.asarray(features.values, dtype=float), metadata=meta, wavenumbers=None)

    def search(
        self,
        X_query: np.ndarray,
        metric: Literal["euclidean", "cosine", "pearson", "sid", "sam"] = "cosine",
        top_k: int = 5,
        query_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        D = compute_distances(X_query, self.X, metric=metric)
        n_q = D.shape[0]
        rows: List[Dict[str, object]] = []
        for i in range(n_q):
            dists = D[i]
            idx = np.argsort(dists)[:top_k]
            for rank, j in enumerate(idx, start=1):
                row = {
                    "query_index": i,
                    "query_id": query_ids[i] if query_ids and i < len(query_ids) else str(i),
                    "library_index": int(j),
                    "distance": float(dists[j]),
                    "rank": int(rank),
                }
                # attach useful metadata fields when present
                for col in ["sample_id", "label", "class", "batch_id", "kind"]:
                    if col in self.metadata.columns:
                        row[f"lib_{col}"] = self.metadata.iloc[j][col]
                rows.append(row)
        return pd.DataFrame(rows)


def similarity_search(
    query_ds: FoodSpectrumSet,
    library_ds: FoodSpectrumSet,
    metric: Literal["euclidean", "cosine", "pearson", "sid", "sam"] = "cosine",
    top_k: int = 5,
) -> pd.DataFrame:
    lib = LibraryIndex.from_dataset(library_ds)
    return lib.search(
        query_ds.x,
        metric=metric,
        top_k=top_k,
        query_ids=list(query_ds.metadata.get("sample_id", pd.Series(np.arange(len(query_ds))).astype(str))),
    )


def overlay_plot(query_x: np.ndarray, match_x: np.ndarray, wavenumbers: np.ndarray):
    """Create an overlay plot (query vs. match). Returns (fig, ax)."""
    from foodspec.core.dataset import FoodSpectrumSet
    from foodspec.viz.spectra import plot_spectra

    X = np.vstack([query_x, match_x])
    meta = pd.DataFrame({"role": ["query", "match"]})
    ds = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=meta)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_spectra(ds, sample_indices=[0, 1], color_by="role", ax=ax)
    ax.set_title("Query vs. Match")
    return fig, ax
