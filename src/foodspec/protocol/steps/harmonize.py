"""
Harmonization step for protocol execution.

Aligns spectral datasets from different instruments to a common wavenumber grid
and applies calibration transfer methods for cross-instrument compatibility.
"""

from typing import Any, Dict

import pandas as pd

from .base import Step


class HarmonizeStep(Step):
    """Harmonize multiple spectral datasets."""

    name = "harmonize"

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize harmonization step.
        
        Parameters
        ----------
        cfg : Dict[str, Any]
            Step configuration with target_wavenumbers and calibration_curves.
        """
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        """Execute harmonization on datasets in context.
        
        Parameters
        ----------
        ctx : Dict[str, Any]
            Execution context with 'datasets' list.
        """
        from foodspec.harmonization import harmonize_datasets_advanced, plot_harmonization_diagnostics
        from foodspec.spectral_io import align_wavenumbers

        datasets = ctx.get("datasets")
        if datasets is None or len(datasets) == 0:
            ctx["logs"].append("[harmonize] No datasets provided; skipping.")
            return
        target_grid = self.cfg.get("target_wavenumbers")
        if len(datasets) > 1:
            curves = self.cfg.get("calibration_curves", {}) or {}
            aligned, diag = harmonize_datasets_advanced(datasets, calibration_curves=curves)
            ctx["metadata"].setdefault("harmonization", {})["diagnostics"] = diag
            # save simple mean overlay plot into figures dict
            fig = plot_harmonization_diagnostics(aligned)
            ctx["figures"]["harmonization/mean_overlay"] = fig
            ctx["logs"].append(
                "[harmonize] Advanced harmonization across "
                f"{len(datasets)} datasets; residual_std_mean={diag.get('residual_std_mean'):.4g}"
            )
        else:
            aligned = align_wavenumbers(datasets, target_grid=target_grid)
            ctx["logs"].append("[harmonize] Completed wavenumber alignment.")
        ctx["datasets"] = aligned
        ctx["data"] = aligned[0].metadata.join(
            pd.DataFrame(aligned[0].spectra, columns=[f"{wn:.4f}" for wn in aligned[0].wavenumbers])
        )
        # record per-instrument info
        instruments = []
        for ds in aligned:
            instruments.append(ds.instrument_meta if hasattr(ds, "instrument_meta") else {})
        ctx["metadata"].setdefault("harmonization", {})["instruments"] = instruments
