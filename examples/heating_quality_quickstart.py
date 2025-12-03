"""
Quickstart script for heating/quality monitoring using foodspec.
Run with: python examples/heating_quality_quickstart.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.apps.heating import run_heating_quality_workflow
from foodspec.viz.heating import plot_ratio_vs_time


def _synthetic_heating_dataset():
    rng = np.random.default_rng(0)
    wavenumbers = np.linspace(600, 1800, 200)
    n = 20
    # synthetic baseline + two peaks changing with time
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
    fs = _synthetic_heating_dataset()
    result = run_heating_quality_workflow(fs, time_column="heating_time")
    print(result.key_ratios.head())

    ratio_name = result.key_ratios.columns[0]
    fig, ax = plt.subplots()
    plot_ratio_vs_time(
        fs.metadata["heating_time"],
        result.key_ratios[ratio_name],
        model=result.trend_models.get(ratio_name),
        ax=ax,
    )
    fig.savefig("heating_ratio_vs_time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
