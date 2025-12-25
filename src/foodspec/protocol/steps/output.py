"""
Output step for protocol execution.

Saves protocol execution results including reports, tables, figures,
metadata, and logs to a timestamped run folder.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from foodspec.output_bundle import (
    append_log,
    create_run_folder,
    save_figures,
    save_index,
    save_metadata,
    save_report_html,
    save_report_text,
    save_tables,
)

from .base import Step


class OutputStep(Step):
    """Save protocol execution outputs to disk."""

    name = "output"

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize output step.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Step configuration with output_dir parameter.
        """
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        """Save all outputs to run folder.

        Parameters
        ----------
        ctx : Dict[str, Any]
            Execution context with tables, figures, reports, metadata.
        """
        out_dir = Path(self.cfg.get("output_dir", "protocol_runs"))
        run_dir = create_run_folder(out_dir)
        ctx["run_dir"] = run_dir
        save_report_text(run_dir / "report.txt", ctx.get("report", ""))
        save_report_html(run_dir / "report.html", ctx.get("report", ""))
        save_tables(run_dir, ctx.get("tables", {}))
        save_figures(run_dir, ctx.get("figures", {}))
        ctx["metadata"]["logs"] = ctx["logs"]
        save_metadata(run_dir, ctx["metadata"])
        save_index(
            run_dir,
            ctx["metadata"],
            ctx.get("tables", {}),
            ctx.get("figures", {}),
            ctx.get("validation", {}).get("warnings", []),
        )
        # Persist HSI artifacts when available
        hsi_labels = ctx.get("hsi_labels")
        if hsi_labels is not None:
            np.save(run_dir / "hsi" / "label_map.npy", hsi_labels)
        hsi_obj = ctx.get("hsi")
        if hsi_obj is not None and getattr(hsi_obj, "roi_masks", None):
            for name, mask in hsi_obj.roi_masks.items():
                np.save(run_dir / "hsi" / f"{name}.npy", mask)
        for line in ctx["logs"]:
            append_log(run_dir, line)
