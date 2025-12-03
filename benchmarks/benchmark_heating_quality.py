"""
Benchmark script for heating/quality monitoring.

Uses a small synthetic dataset to exercise the heating workflow, saves metrics,
plots, and run metadata for reproducibility.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foodspec.apps.heating import run_heating_quality_workflow
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.viz.heating import plot_ratio_vs_time


def _synthetic_heating_dataset():
    rng = np.random.default_rng(0)
    wavenumbers = np.linspace(600, 1800, 200)
    n = 20
    times = np.linspace(0, 30, n)
    spectra = []
    for t in times:
        peak1 = np.exp(-0.5 * ((wavenumbers - 1655) / 8) ** 2) * (1.0 - 0.01 * t)
        peak2 = np.exp(-0.5 * ((wavenumbers - 1742) / 8) ** 2) * (1.0 + 0.005 * t)
        baseline = 0.02 + 0.00001 * (wavenumbers - 1200) ** 2
        noise = rng.normal(0, 0.01, size=wavenumbers.shape)
        spectra.append(peak1 + peak2 + baseline + noise)
    x = np.vstack(spectra)
    metadata = pd.DataFrame({"sample_id": [f"s{i:03d}" for i in range(n)], "heating_time": times})
    return FoodSpectrumSet(x=x, wavenumbers=wavenumbers, metadata=metadata, modality="raman")


def main():
    out_dir = Path("benchmarks_output/heating_quality")
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = _synthetic_heating_dataset()
    res = run_heating_quality_workflow(fs, time_column="heating_time")
    ratio_name = res.key_ratios.columns[0]
    model = res.trend_models.get(ratio_name)

    # Save ratios and model coefficients
    res.key_ratios.to_csv(out_dir / "ratios.csv", index=False)
    if model is not None:
        coeffs = {"slope": float(model.coef_[0]), "intercept": float(model.intercept_)}
        (out_dir / "trend_model.json").write_text(json.dumps(coeffs, indent=2))

    # Plot ratio vs time
    fig, ax = plt.subplots()
    plot_ratio_vs_time(fs.metadata["heating_time"], res.key_ratios[ratio_name], model=model, ax=ax)
    fig.savefig(out_dir / "ratio_vs_time.png", dpi=150)
    plt.close(fig)

    # Run metadata
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "foodspec_version": getattr(sys.modules.get("foodspec"), "__version__", "unknown"),
        "time_column": "heating_time",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Benchmark artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
