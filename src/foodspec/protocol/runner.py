"""
Protocol execution runner.

Orchestrates protocol execution across multiple steps with support for
multi-dataset inputs, HSI processing, and comprehensive result tracking.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from foodspec.core.spectral_dataset import HyperspectralDataset, SpectralDataset
from foodspec.output_bundle import save_figures, save_tables

from .config import ProtocolConfig, ProtocolRunResult
from .steps import STEP_REGISTRY
from .utils import validate_protocol


class ProtocolRunner:
    """Execute protocols with multiple steps and datasets."""

    def __init__(self, config: ProtocolConfig):
        """Initialize protocol runner.

        Parameters
        ----------
        config : ProtocolConfig
            Protocol configuration to execute.
        """
        self.config = config
        self._cancel = False

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ProtocolRunner":
        """Create runner from protocol file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to protocol YAML or JSON file.

        Returns
        -------
        ProtocolRunner
            Initialized runner with loaded configuration.
        """
        return cls(ProtocolConfig.from_file(path))

    def request_cancel(self):
        """Request cancellation of current protocol execution."""
        self._cancel = True

    def run(
        self,
        input_datasets: List[Union[pd.DataFrame, str, Path, SpectralDataset, HyperspectralDataset]],
    ) -> ProtocolRunResult:
        """Execute protocol on input datasets.

        Multi-dataset aware runner:
        - Supports multiple spectral datasets (for harmonize) and HSI datasets.
        - Uses the first table as primary data for steps that expect a single df.

        Parameters
        ----------
        input_datasets : List[Union[pd.DataFrame, str, Path, SpectralDataset, HyperspectralDataset]]
            Input datasets (DataFrames, file paths, or dataset objects).

        Returns
        -------
        ProtocolRunResult
            Execution result with logs, tables, figures, and metadata.

        Raises
        ------
        ValueError
            If no inputs provided or validation errors detected.
        """
        if not input_datasets and self.config.inputs:
            input_datasets = [Path(inp["path"]) for inp in self.config.inputs if "path" in inp]
        if not input_datasets:
            raise ValueError("No inputs provided.")

        datasets: List[SpectralDataset] = []
        hsi_list: List[HyperspectralDataset] = []
        tables: List[pd.DataFrame] = []

        def _load_one(raw_input):
            if isinstance(raw_input, HyperspectralDataset):
                hsi_list.append(raw_input)
                tables.append(raw_input.metadata.copy() if not raw_input.metadata.empty else pd.DataFrame())
                return
            if isinstance(raw_input, SpectralDataset):
                datasets.append(raw_input)
                df_spectra = pd.DataFrame(raw_input.spectra, columns=[f"{wn:.4f}" for wn in raw_input.wavenumbers])
                tables.append(pd.concat([raw_input.metadata.reset_index(drop=True), df_spectra], axis=1))
                return
            if isinstance(raw_input, (str, Path)):
                p = Path(raw_input)
                if p.suffix.lower() in {".h5", ".hdf5"}:
                    try:
                        hsi_obj = HyperspectralDataset.from_hdf5(p)
                        hsi_list.append(hsi_obj)
                        tables.append(hsi_obj.metadata.copy() if not hsi_obj.metadata.empty else pd.DataFrame())
                        return
                    except Exception:
                        spectral_obj = SpectralDataset.from_hdf5(p)
                        datasets.append(spectral_obj)
                        df_spectra = pd.DataFrame(
                            spectral_obj.spectra,
                            columns=[f"{wn:.4f}" for wn in spectral_obj.wavenumbers],
                        )
                        tables.append(pd.concat([spectral_obj.metadata.reset_index(drop=True), df_spectra], axis=1))
                        return
                tables.append(pd.read_csv(p))
                return
            tables.append(raw_input)

        for inp in input_datasets:
            _load_one(inp)

        primary_df = tables[0]
        diag = validate_protocol(self.config, primary_df)
        if diag["errors"]:
            raise ValueError("; ".join(diag["errors"]))
        ctx: Dict[str, Any] = {
            "data": primary_df,
            "logs": [],
            "metadata": {
                "protocol": self.config.name,
                "protocol_version": self.config.version,
                "min_foodspec_version": self.config.min_foodspec_version,
                "seed": self.config.seed,
                "inputs": [str(inp) for inp in input_datasets],
                "validation_strategy": self.config.validation_strategy,
            },
            "tables": {},
            "figures": {},
            "report": "",
            "summary": "",
            "run_dir": None,
            "validation": diag,
            "hsi": hsi_list[0] if hsi_list else None,
            "dataset": datasets[0] if datasets else None,
            "datasets": datasets,
            "cancel": False,
            "model_path": None,
            "registry_path": None,
        }
        start = time.time()
        # Reproducibility
        try:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
        except Exception:
            pass
        # Light guardrails
        guard_df = primary_df if isinstance(primary_df, pd.DataFrame) else None
        guard_hsi = hsi_list[0] if hsi_list else None
        if guard_df is not None:
            if guard_df.shape[0] * guard_df.shape[1] > 2_000_000:
                ctx["logs"].append(f"[warn] Large dataset ({guard_df.shape}); consider sub-sampling or dry-run first.")
            if guard_df.shape[1] > 10 * max(1, guard_df.shape[0]):
                ctx["logs"].append("[warn] High feature-to-sample ratio; RQ may auto-cap or warn.")
        if guard_hsi is not None and guard_hsi.spectra.size > 5_000_000:
            ctx["logs"].append("[warn] Large HSI cube; segmentation may be slow.")
        for step_cfg in self.config.steps:
            if self._cancel:
                ctx["logs"].append("[cancelled] User requested cancel; stopping protocol.")
                break
            step_name = step_cfg.get("type")
            step_params = step_cfg.get("params", {})
            # Auto-adjust CV folds if class counts are small
            if step_name == "rq_analysis" and isinstance(primary_df, pd.DataFrame):
                oil_col = self.config.expected_columns.get("oil_col", "oil_type")
                if oil_col in primary_df.columns:
                    min_count = primary_df[oil_col].value_counts(dropna=True).min()
                    default_splits = step_params.get("n_splits", 5)
                    if pd.notna(min_count) and min_count < default_splits:
                        new_splits = max(2, int(min_count)) if min_count >= 1 else 2
                        step_params["n_splits"] = new_splits
                        ctx["logs"].append(f"[auto] Reduced CV folds to {new_splits} due to small class counts.")
                        ctx["metadata"].setdefault("auto_adjustments", {})["cv_folds"] = new_splits
            step_cls = STEP_REGISTRY.get(step_name)
            if not step_cls:
                ctx["logs"].append(f"[skip] Unknown step {step_name}")
                continue
            step = step_cls(step_params)
            step.run(ctx)
            if self._cancel:
                ctx["logs"].append(f"[cancelled] Stopped after step {step_name}.")
                break
        ctx["metadata"]["duration_sec"] = time.time() - start
        return ProtocolRunResult(
            run_dir=ctx.get("run_dir"),
            logs=ctx["logs"],
            metadata=ctx["metadata"],
            tables=ctx["tables"],
            figures=ctx["figures"],
            report=ctx["report"],
            summary=ctx["summary"],
        )

    def save_outputs(self, result: ProtocolRunResult, output_dir: Union[str, Path]):
        """Save protocol execution results to directory.

        Parameters
        ----------
        result : ProtocolRunResult
            Execution result to save.
        output_dir : Union[str, Path]
            Directory to save outputs to.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.txt").write_text(result.report, encoding="utf-8")
        meta = result.metadata.copy()
        meta["protocol_version"] = self.config.version
        meta["min_foodspec_version"] = self.config.min_foodspec_version
        meta["logs"] = result.logs
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        save_tables(out_dir, result.tables)
        save_figures(out_dir, result.figures)
