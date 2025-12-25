"""
Preprocessing step for protocol execution.

Applies baseline correction, smoothing, normalization, and peak extraction
to spectral data using the preprocessing pipeline.
"""

from typing import Any, Dict

import pandas as pd

from foodspec.preprocessing_pipeline import PreprocessingConfig, detect_input_mode, run_full_preprocessing
from foodspec.features.rq import PeakDefinition

from .base import Step


class PreprocessStep(Step):
    """Apply preprocessing to spectral data."""

    name = "preprocess"

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize preprocessing step.
        
        Parameters
        ----------
        cfg : Dict[str, Any]
            Step configuration containing preprocessing parameters.
        """
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        """Execute preprocessing on data in context.
        
        Parameters
        ----------
        ctx : Dict[str, Any]
            Execution context with 'data' DataFrame.
        """
        df: pd.DataFrame = ctx["data"]
        if ctx.get("cancel"):
            ctx["logs"].append("[preprocess] cancelled")
            return
        peaks_cfg = self.cfg.get("peaks", [])
        peak_defs = [
            PeakDefinition(
                name=p.get("name"),
                column=p.get("column", p.get("name")),
                wavenumber=p.get("wavenumber"),
                window=tuple(p.get("window")) if p.get("window") else None,
            )
            for p in peaks_cfg
        ]
        pp_cfg = PreprocessingConfig(
            baseline_method=self.cfg.get("baseline_method", "als"),
            baseline_lambda=self.cfg.get("baseline_lambda", 1e5),
            baseline_p=self.cfg.get("baseline_p", 0.01),
            baseline_enabled=self.cfg.get("baseline_enabled", True),
            smooth_enabled=self.cfg.get("smooth_enabled", True),
            smoothing_window=self.cfg.get("smoothing_window", self.cfg.get("smooth_window", 7)),
            smoothing_polyorder=self.cfg.get("smoothing_polyorder", self.cfg.get("smooth_polyorder", 3)),
            normalization=self.cfg.get("normalization", "reference"),
            reference_wavenumber=self.cfg.get("reference_wavenumber", 2720.0),
            spike_removal=self.cfg.get("spike_removal", True),
            spike_zscore_thresh=self.cfg.get("spike_zscore_thresh", 8.0),
            peak_definitions=peak_defs,
        )
        ctx["logs"].append(f"[preprocess] mode={detect_input_mode(df)}, cfg={pp_cfg}")
        ctx["data"] = run_full_preprocessing(df, pp_cfg)
        ctx["metadata"]["preprocessing"] = pp_cfg.__dict__
