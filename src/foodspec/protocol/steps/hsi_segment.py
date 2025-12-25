"""
Hyperspectral imaging segmentation step.

Performs spatial segmentation of HSI datacubes using clustering methods
(k-means, hierarchical, etc.) to identify regions of interest.
"""

from typing import Any, Dict

import pandas as pd

from .base import Step


class HSISegmentStep(Step):
    """Segment hyperspectral imaging data."""

    name = "hsi_segment"

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize HSI segmentation step.
        
        Parameters
        ----------
        cfg : Dict[str, Any]
            Step configuration with method and n_clusters.
        """
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        """Execute HSI segmentation.
        
        Parameters
        ----------
        ctx : Dict[str, Any]
            Execution context with 'hsi' HyperspectralDataset.
        """
        hsi = ctx.get("hsi")
        if hsi is None:
            ctx["logs"].append("[hsi_segment] No HSI dataset; skipping.")
            return
        method = self.cfg.get("method", "kmeans")
        n_clusters = self.cfg.get("n_clusters", 3)
        labels = hsi.segment(method=method, n_clusters=n_clusters)
        ctx["hsi_labels"] = labels
        ctx["figures"]["hsi/label_map"] = labels
        counts = pd.Series(labels.ravel()).value_counts().reset_index()
        counts.columns = ["label", "pixels"]
        ctx["tables"]["hsi_label_counts"] = counts
        ctx["logs"].append(f"[hsi_segment] Segmented HSI with {method}, clusters={n_clusters}.")
