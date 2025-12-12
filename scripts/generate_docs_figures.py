#!/usr/bin/env python3
"""
Generate example figures for FoodSpec docs using bundled example data and protocols.
Outputs saved under docs/assets/figures/.

Run:
    python scripts/generate_docs_figures.py

This script is non-destructive: it reads existing example data/protocols and
writes figures for documentation illustration. Adapt paths/logic as needed
if your bundle structure differs.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from foodspec.protocol_engine import ProtocolRunner, ProtocolConfig
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Cannot import FoodSpec protocol engine: {exc}")


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
PROT = ROOT / "examples" / "protocols"
OUT = ROOT / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def run_protocol(input_paths, protocol_path, out_dir):
    cfg = ProtocolConfig.from_file(protocol_path)
    runner = ProtocolRunner(cfg)
    ctx = runner.run(input_datasets=input_paths, output_dir=out_dir)
    return ctx


def copy_existing_fig(bundle_dir: Path, pattern: str, target_name: str):
    figs = bundle_dir / "figures"
    for f in figs.glob(pattern):
        (OUT / target_name).write_bytes(f.read_bytes())
        return True
    return False


def plot_bar_from_csv(csv_path: Path, value_col: str, label_col: str, out_name: str, title: str, ascending=False, n=10):
    df = pd.read_csv(csv_path)
    df = df.sort_values(value_col, ascending=ascending).head(n)
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x=value_col, y=label_col, orient="h")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=300)
    plt.close()


def plot_minimal_panel(csv_path: Path, out_name: str):
    df = pd.read_csv(csv_path)
    if "k" not in df.columns or "mean_acc" not in df.columns:
        return
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=df, x="k", y="mean_acc")
    if "std_acc" in df.columns:
        plt.fill_between(df["k"], df["mean_acc"] - df["std_acc"], df["mean_acc"] + df["std_acc"], alpha=0.2)
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.title("Minimal panel accuracy")
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=300)
    plt.close()


def plot_cv_box(csv_path: Path, metric_col: str, out_name: str):
    df = pd.read_csv(csv_path)
    if metric_col not in df.columns:
        return
    plt.figure(figsize=(4, 4))
    sns.boxplot(y=df[metric_col])
    plt.title(f"Cross-validation {metric_col}")
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=300)
    plt.close()


def simple_architecture(out_name: str):
    plt.figure(figsize=(6, 4))
    plt.axis("off")
    y = 0.9
    step = 0.12
    labels = [
        "Input (CSV/HDF5/vendor)",
        "IO layer (SpectralDataset / HyperspectralDataset)",
        "Preprocess + Harmonize",
        "RQ Engine",
        "Protocol Engine",
        "Bundles (report/figures/tables/models)",
        "GUI / CLI / Web",
    ]
    for lbl in labels:
        plt.text(0.5, y, lbl, ha="center", va="center")
        y -= step
    plt.savefig(OUT / out_name, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Oil discrimination
    oil_run = ROOT / "runs" / "doc_oil_basic"
    run_protocol([DATA / "oils.csv"], PROT / "oil_basic.yaml", oil_run)
    copy_existing_fig(oil_run, "*confusion*.png", "oil_confusion.png")
    if (oil_run / "tables" / "discriminative_summary.csv").exists():
        plot_bar_from_csv(oil_run / "tables" / "discriminative_summary.csv", "rf_importance", "feature", "oil_discriminative.png", "Top discriminative ratios", ascending=False)
    if (oil_run / "tables" / "stability_summary.csv").exists():
        plot_bar_from_csv(oil_run / "tables" / "stability_summary.csv", "cv", "feature", "oil_stability.png", "Top stable ratios", ascending=True)
    if (oil_run / "tables" / "minimal_panel.csv").exists():
        plot_minimal_panel(oil_run / "tables" / "minimal_panel.csv", "oil_minimal_panel.png")

    # Heating trends
    heat_run = ROOT / "runs" / "doc_oil_heating"
    run_protocol([DATA / "oils.csv"], PROT / "oil_heating.yaml", heat_run)
    copy_existing_fig(heat_run, "trend*.png", "heating_trend.png")

    # Oil vs chips
    ovsc_run = ROOT / "runs" / "doc_oil_vs_chips"
    run_protocol([DATA / "oils.csv", DATA / "chips.csv"], PROT / "oil_vs_chips.yaml", ovsc_run)
    if (ovsc_run / "tables" / "oil_vs_chips_summary.csv").exists():
        plot_bar_from_csv(ovsc_run / "tables" / "oil_vs_chips_summary.csv", "effect_size", "feature", "oil_vs_chips_divergence.png", "Top matrix divergences", ascending=False)

    # HSI
    hsi_run = ROOT / "runs" / "doc_hsi"
    run_protocol([DATA / "hsi_cube.h5"], PROT / "hsi_segment_roi.yaml", hsi_run)
    copy_existing_fig(hsi_run, "hsi_label_map*.png", "hsi_label_map.png")
    copy_existing_fig(hsi_run, "roi_spectra*.png", "roi_spectra.png")

    # Validation example (if available)
    # plot_cv_box(Path("path/to/cv_metrics.csv"), "balanced_accuracy", "cv_boxplot.png")

    # Architecture schematic
    simple_architecture("architecture_flow.png")


if __name__ == "__main__":  # pragma: no cover
    main()
