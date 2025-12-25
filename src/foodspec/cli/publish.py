#!/usr/bin/env python3
"""
CLI to generate Methods/figure bundle from a FoodSpec run folder.

Example:
    foodspec-publish --run-dir protocol_runs/20240101_run --out out_bundle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foodspec.narrative import save_markdown_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish a FoodSpec run as Methods+figures bundle.")
    parser.add_argument("--run-dir", required=True, help="Run folder containing metadata/report/figures.")
    parser.add_argument("--out", required=True, help="Output directory for bundle.")
    parser.add_argument("--fig-limit", type=int, default=None, help="Limit number of figures in panel.")
    parser.add_argument("--include-all-figures", action="store_true", help="Include all figures (ignore limit).")
    parser.add_argument(
        "--profile",
        choices=["quicklook", "qa", "standard", "publication"],
        default="standard",
        help="Figure selection profile.",
    )
    args = parser.parse_args(argv)

    save_markdown_bundle(
        Path(args.run_dir),
        Path(args.out),
        fig_limit=args.fig_limit,
        include_all=args.include_all_figures or args.profile == "publication",
        profile=args.profile,
    )
    print(f"Bundle saved to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
