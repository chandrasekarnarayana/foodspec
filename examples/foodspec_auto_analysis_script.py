"""
CLI-style demo: run the FoodSpec automatic analysis protocol on a CSV path.
"""
import argparse
from pathlib import Path

import pandas as pd

from modeling_gui.foodspec_adapter import default_preset
from modeling_gui.foodspec_protocol import AutoAnalysisConfig, run_foodspec_auto_analysis
from foodspec.protocol_engine import ProtocolRunner, ProtocolConfig
from foodspec.protocol_engine import EXAMPLE_PROTOCOL


def main():
    parser = argparse.ArgumentParser(description="FoodSpec automatic RQ analysis (CLI demo).")
    parser.add_argument("csv", help="CSV with peak intensities/ratios and metadata.")
    parser.add_argument("--oil_col", default="oil_type")
    parser.add_argument("--matrix_col", default="matrix")
    parser.add_argument("--heating_col", default="heating_stage")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    mappings = {
        "oil_col": args.oil_col,
        "matrix_col": args.matrix_col,
        "heating_col": args.heating_col,
    }
    res = run_foodspec_auto_analysis(
        df,
        mappings,
        preset=default_preset(),
        config=AutoAnalysisConfig(),
    )

    print("=== Executive summary ===")
    print(res.summary)
    print("\n=== Full report (first 40 lines) ===")
    print("\n".join(res.report.splitlines()[:40]))

    # Optionally save key figures
    out_dir = Path("foodspec_auto_outputs")
    out_dir.mkdir(exist_ok=True)
    for name, fig in res.figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=200)
    print(f"Saved {len(res.figures)} figures to {out_dir}")


if __name__ == "__main__":
    main()
