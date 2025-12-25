#!/usr/bin/env python3
"""
CLI to run a FoodSpec protocol and produce a run bundle.

Examples
--------
Single file:
    foodspec-run-protocol --input data/oils.csv --protocol EdibleOil_Classification_v1 --output-dir runs

Glob / directory:
    foodspec-run-protocol --input-dir data/batch --glob \"*.csv\" --protocol \
EdibleOil_Classification_v1 --output-dir runs

Overrides:
    foodspec-run-protocol --input data/oils.csv --protocol EdibleOil_Classification_v1 --output-dir runs \\
        --seed 42 --cv-folds 3 --normalization-mode vector --baseline-method rubberband
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

from foodspec.core.spectral_dataset import HyperspectralDataset, SpectralDataset
from foodspec.logging_utils import setup_logging
from foodspec.protocol import ProtocolRunner, load_protocol, validate_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run FoodSpec protocol.")
    parser.add_argument("--input", help="Input CSV/HDF5 file (can be given multiple times).", action="append")
    parser.add_argument("--input-dir", help="Directory containing inputs (used with --glob).")
    parser.add_argument("--glob", default="*.csv", help="Glob pattern when using --input-dir.")
    parser.add_argument("--protocol", required=True, help="Protocol name (in examples/protocols) or path.")
    parser.add_argument("--output-dir", default="protocol_runs", help="Directory for run outputs.")
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--cv-folds", type=int, help="Override CV folds for RQ models.")
    parser.add_argument(
        "--normalization-mode",
        help="Override normalization mode for preprocess/RQ (e.g., reference, vector).",
    )
    parser.add_argument("--baseline-method", help="Override baseline method (als, rubberband, polynomial, none).")
    # Spike removal toggle (cosmic-ray correction in preprocessing)
    spike_group = parser.add_mutually_exclusive_group()
    spike_group.add_argument(
        "--spike-removal",
        dest="spike_removal",
        action="store_true",
        help="Enable cosmic-ray spike removal in preprocessing.",
    )
    spike_group.add_argument(
        "--no-spike-removal",
        dest="spike_removal",
        action="store_false",
        help="Disable cosmic-ray spike removal in preprocessing.",
    )
    parser.set_defaults(spike_removal=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (only errors).")
    parser.add_argument("--no-figures", action="store_true", help="(Reserved) Skip saving figures.")
    parser.add_argument(
        "--validation-strategy",
        choices=["standard", "batch_aware", "group_stratified"],
        default=None,
        help="Validation approach.",
    )
    parser.add_argument("--auto", action="store_true", help="Run publish after protocol completes.")
    parser.add_argument(
        "--report-level",
        choices=["summary", "standard", "full"],
        default="standard",
        help="Figure/text richness for auto publish.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate only; do not execute.")
    parser.add_argument("--check-env", action="store_true", help="Print environment/dep status and exit.")
    args = parser.parse_args(argv)

    if args.check_env:
        from foodspec.check_env import check_env

        print(check_env())
        return 0
    logger = setup_logging()

    # Collect inputs
    inputs: List[Path] = []
    if args.input:
        inputs.extend([Path(p) for p in args.input])
    if args.input_dir:
        inputs.extend([Path(p) for p in glob.glob(str(Path(args.input_dir) / args.glob))])
    if not inputs:
        print("No input files provided. Use --input or --input-dir.", file=sys.stderr)
        return 1

    # Load protocol
    proto_path = args.protocol
    if Path(proto_path).exists():
        runner = ProtocolRunner.from_file(proto_path)
    else:
        runner = ProtocolRunner(load_protocol(args.protocol))

    cfg = runner.config
    if args.seed is not None:
        cfg.seed = args.seed
    if args.validation_strategy:
        cfg.validation_strategy = args.validation_strategy
    # Apply overrides to steps
    for step in cfg.steps:
        if step.get("type") == "preprocess":
            if args.normalization_mode:
                step.setdefault("params", {})["normalization"] = args.normalization_mode
            if args.baseline_method:
                step.setdefault("params", {})["baseline_method"] = args.baseline_method
            if args.spike_removal is not None:
                step.setdefault("params", {})["spike_removal"] = args.spike_removal
        if step.get("type") == "rq_analysis":
            if args.cv_folds:
                step.setdefault("params", {})["n_splits"] = args.cv_folds
            if args.normalization_mode:
                step.setdefault("params", {})["normalization_modes"] = [args.normalization_mode]

    exit_code = 0
    if not args.quiet:
        logger.info("=== FoodSpec Protocol Runner ===")
        logger.info("Protocol: %s (v%s)", cfg.name, cfg.version)
    # Estimate for dry-run
    if args.dry_run:
        if not args.quiet:
            print("=== DRY RUN ===")
            print(f"Protocol: {cfg.name} (v{cfg.version})")
            print(f"Inputs: {len(inputs)} file(s)")
            if isinstance(inputs[0], Path) and inputs[0].exists():
                size_mb = inputs[0].stat().st_size / (1024 * 1024)
                print(f"First input size: {size_mb:.2f} MB")
            print("No execution performed.")
        return 0

    for path in inputs:
        if args.verbose:
            logger.info("[INFO] Running protocol on %s", path)
        df_or_obj = None
        if path.suffix.lower() in {".h5", ".hdf5"}:
            try:
                df_or_obj = SpectralDataset.from_hdf5(path)
            except Exception:
                df_or_obj = HyperspectralDataset.from_hdf5(path)
        else:
            df_or_obj = pd.read_csv(path)

        # Validation (only for DataFrame inputs)
        diag = {"errors": [], "warnings": []}
        if isinstance(df_or_obj, pd.DataFrame):
            diag = validate_protocol(cfg, df_or_obj)
            logger.info("[VALIDATION][%s]", path.name)
            if diag["errors"]:
                for e in diag["errors"]:
                    logger.error("ERROR: %s", e)
                exit_code = 1
                continue
            if diag["warnings"] and not args.quiet:
                for w in diag["warnings"]:
                    logger.warning("WARN: %s", w)
        elif args.verbose:
            print("[WARN] Validation skipped for non-DataFrame input.")

        # Run
        try:
            result = runner.run([df_or_obj])
        except Exception as exc:  # pragma: no cover - CLI error path
            # Retry with safer CV folds if fold count is too high for class sizes
            if "n_splits" in str(exc):
                for step in cfg.steps:
                    if step.get("type") == "rq_analysis":
                        est_rows = len(getattr(df_or_obj, "index", [])) or 2
                        step.setdefault("params", {})["n_splits"] = max(2, min(3, est_rows))
                try:
                    result = runner.run([df_or_obj])
                except Exception as exc2:
                    logger.error("[ERROR] Failed on %s: %s", path, exc2)
                    exit_code = 1
                    continue
            else:
                logger.error("[ERROR] Failed on %s: %s", path, exc)
                exit_code = 1
                continue
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{cfg.name}_{path.stem}"
        runner.save_outputs(result, target)
        run_dir = target
        # Annotate metadata with input info
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta["inputs"] = [str(p) for p in inputs]
                meta["multi_input"] = len(inputs) > 1
                meta_path.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass
        if args.auto:
            try:
                from foodspec.narrative import save_markdown_bundle

                publish_dir = run_dir / "publish"
                publish_dir.mkdir(parents=True, exist_ok=True)
                fig_limit = {"summary": 4, "standard": 8, "full": None}[args.report_level]
                save_markdown_bundle(
                    run_dir,
                    publish_dir,
                    fig_limit=fig_limit,
                    include_all=args.report_level == "full",
                    profile="standard" if args.report_level != "summary" else "quicklook",
                )
                if not args.quiet:
                    logger.info("[AUTO-PUBLISH] Bundle saved to %s", publish_dir)
            except Exception as exc:
                logger.warning("[WARN] Auto-publish failed: %s", exc)
        if not args.quiet:
            logger.info("[DONE] %s -> %s", path, run_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
