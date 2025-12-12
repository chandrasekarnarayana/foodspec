#!/usr/bin/env python3
"""
Apply a frozen FoodSpec model to new data.

Example:
    foodspec-predict --model runs/model --input data/new.csv --output preds.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from foodspec.model_lifecycle import FrozenModel


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply a frozen FoodSpec model to new data.")
    parser.add_argument("--model", required=True, help="Path prefix to frozen model (.json/.pkl).")
    parser.add_argument("--input", help="Input CSV/HDF5 with spectra/peaks.", action="append")
    parser.add_argument("--input-dir", help="Directory of inputs (use with --glob).")
    parser.add_argument("--glob", default="*.csv", help="Glob pattern for --input-dir.")
    parser.add_argument("--output", help="Output CSV for single input; for batch, use --output-dir.")
    parser.add_argument("--output-dir", help="Directory for batch predictions (one CSV per input).")
    parser.add_argument("--check-env", action="store_true", help="Print environment/dep status and exit.")
    args = parser.parse_args(argv)

    if args.check_env:
        from foodspec.check_env import check_env

        print(check_env())
        return 0

    inputs = []
    if args.input:
        inputs.extend([Path(p) for p in args.input])
    if args.input_dir:
        inputs.extend([Path(p) for p in Path(args.input_dir).glob(args.glob)])
    if not inputs:
        print("No inputs provided. Use --input or --input-dir.", file=sys.stderr)
        return 1

    model = FrozenModel.load(Path(args.model))
    if len(inputs) == 1:
        df = pd.read_csv(inputs[0])
        preds = model.predict(df)
        out_path = args.output or "predictions.csv"
        preds.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")
    else:
        out_dir = Path(args.output_dir or "predictions_batch")
        out_dir.mkdir(parents=True, exist_ok=True)
        for inp in inputs:
            df = pd.read_csv(inp)
            preds = model.predict(df)
            out_path = out_dir / f"{inp.stem}_preds.csv"
            preds.to_csv(out_path, index=False)
            print(f"[{inp.name}] -> {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
