"""
Hyperspectral ROI extraction step.

Extracts representative spectra from segmented regions of interest (ROIs)
and converts them to 1D tabular format for downstream analysis.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from foodspec.features.rq import PeakDefinition, RatioDefinition, RatioQualityEngine, RQConfig

from .base import Step


class HSIRoiStep(Step):
    """Extract ROI spectra from hyperspectral data to 1D table."""

    name = "hsi_roi_to_1d"

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize HSI ROI extraction step.
        
        Parameters
        ----------
        cfg : Dict[str, Any]
            Step configuration with peaks and optional RQ analysis params.
        """
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        """Execute ROI extraction and optional RQ analysis.
        
        Parameters
        ----------
        ctx : Dict[str, Any]
            Execution context with 'hsi' and 'hsi_labels'.
        """
        hsi = ctx.get("hsi")
        labels = ctx.get("hsi_labels")
        if hsi is None or labels is None:
            ctx["logs"].append("[hsi_roi_to_1d] Missing HSI or labels; skipping.")
            return
        if ctx.get("cancel"):
            ctx["logs"].append("[hsi_roi_to_1d] cancelled")
            return
        dfs = []
        roi_masks = {}
        peak_defs_cfg = self.cfg.get("peaks", [])
        peak_defs = [
            PeakDefinition(
                name=p["name"],
                column=p.get("column", p["name"]),
                wavenumber=p.get("wavenumber"),
                window=tuple(p.get("window")) if p.get("window") else None,
            )
            for p in peak_defs_cfg
        ]
        for k in np.unique(labels):
            mask = labels == k
            roi_masks[f"label_{k}"] = mask
            roi_ds = hsi.roi_spectrum(mask)
            df_peaks = roi_ds.to_peaks(peak_defs)
            df_peaks["roi_label"] = k
            dfs.append(df_peaks)
        if dfs:
            roi_df = pd.concat(dfs, ignore_index=True)
            ctx["data"] = roi_df
            ctx["tables"]["hsi_roi_peaks"] = roi_df
            ctx["hsi"].roi_masks = roi_masks
            ctx["figures"]["hsi/labels_preview"] = labels
            ctx["logs"].append("[hsi_roi_to_1d] Extracted ROI spectra to 1D table.")
            if self.cfg.get("run_rq"):
                peaks_cfg = self.cfg.get("peaks", [])
                ratios_cfg = self.cfg.get("ratios", [])
                peaks = [
                    PeakDefinition(
                        name=p["name"],
                        column=p.get("column", p["name"]),
                        wavenumber=p.get("wavenumber"),
                        window=tuple(p.get("window")) if p.get("window") else None,
                    )
                    for p in peaks_cfg
                ]
                ratios = [
                    RatioDefinition(
                        name=r["name"],
                        numerator=r["numerator"],
                        denominator=r["denominator"],
                    )
                    for r in ratios_cfg
                ]
                rq_cfg = RQConfig(
                    oil_col=self.cfg.get("oil_col", "oil_type"),
                    matrix_col=self.cfg.get("matrix_col", "matrix"),
                    heating_col=self.cfg.get("heating_col", "heating_stage"),
                    random_state=self.cfg.get("random_state", 0),
                )
                engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=rq_cfg)
                res = engine.run_all(roi_df)
                ctx["tables"]["hsi_roi_rq"] = res.stability_summary
                ctx["figures"].update(getattr(res, "figures", {}) or {})
                ctx["report"] = res.text_report
                ctx["summary"] = " ".join(res.text_report.splitlines()[:10])
