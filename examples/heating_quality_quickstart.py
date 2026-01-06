"""
Quickstart script for heating/quality monitoring using foodspec.
Run with: python examples/heating_quality_quickstart.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foodspec.apps.heating import run_heating_quality_workflow
from foodspec.demo import synthetic_heating_dataset
from foodspec.viz.heating import plot_ratio_vs_time


def main():
    fs = synthetic_heating_dataset()
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
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "heating_ratio_vs_time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
