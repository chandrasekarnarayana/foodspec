#!/usr/bin/env python3
"""
CLI entry point for FoodSpec automatic analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from modeling_gui.foodspec_adapter import default_preset, load_preset
from modeling_gui.foodspec_protocol import AutoAnalysisConfig, run_foodspec_auto_analysis
from foodspec.preprocessing_pipeline import PreprocessingConfig


def main():
    parser = argparse.ArgumentParser(description="Run FoodSpec auto-analysis and save a bundle.")
    parser.add_argument("--input-csv", required=True, help="Input CSV (raw spectra or peak table).")
    parser.add_argument("--output-dir", required=True, help="Directory to place the run folder.")
    parser.add_argument("--preset", default=None, help="Optional preset file (YAML/JSON).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--norm-modes", default="reference", help="Comma-separated normalization modes (reference,vector,area,max).")
    parser.add_argument("--oil-col", default="oil_type")
    parser.add_argument("--matrix-col", default="matrix")
    parser.add_argument("--heating-col", default="heating_stage")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    preset = load_preset(args.preset) if args.preset else default_preset()
    mappings = {
        "oil_col": args.oil_col,
        "matrix_col": args.matrix_col,
        "heating_col": args.heating_col,
    }
    cfg = AutoAnalysisConfig(
        random_state=args.seed,
        n_splits=args.cv_folds,
        output_dir=args.output_dir,
        preprocessing=PreprocessingConfig(peak_definitions=preset.peaks),
    )
    # propagate normalization modes into RQConfig via adapter
    preset.norm_modes = [m.strip() for m in args.norm_modes.split(",") if m.strip()]
    res = run_foodspec_auto_analysis(df, mappings, preset=preset, config=cfg)
    print("Run folder:", res.run_folder)
    print(res.summary)


if __name__ == "__main__":
    main()
