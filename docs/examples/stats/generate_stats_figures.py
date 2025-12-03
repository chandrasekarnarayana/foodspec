"""
Generate simple statistical figures for documentation.

Outputs:
- docs/assets/boxplot_anova.png: Boxplot showing clear group differences.
- docs/assets/corr_scatter.png: Scatter plot with Pearson correlation.
- docs/assets/corr_heatmap.png: Correlation heatmap for two variables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parents[2] / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def fig_boxplot_anova():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    rng.normal(1.0, 0.1, size=20),
                    rng.normal(1.5, 0.1, size=20),
                    rng.normal(2.0, 0.1, size=20),
                ]
            ),
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        }
    )
    plt.figure(figsize=(4, 3))
    groups = ["A", "B", "C"]
    data_by_group = [data.loc[data["group"] == g, "value"] for g in groups]
    plt.boxplot(data_by_group, labels=groups, patch_artist=True)
    plt.title("Example ANOVA: clear group separation")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(ASSETS / "boxplot_anova.png", dpi=150)
    plt.close()


def fig_corr_scatter():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 40)
    y = 2 * x + rng.normal(0, 3, size=len(x))
    plt.figure(figsize=(4, 3))
    plt.scatter(x, y, s=25, alpha=0.7, label="data")
    # simple least-squares line
    coef = np.polyfit(x, y, deg=1)
    y_pred = np.polyval(coef, x)
    plt.plot(x, y_pred, color="red", label="fit")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Example Pearson correlation (positive trend)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "corr_scatter.png", dpi=150)
    plt.close()


def fig_corr_heatmap():
    df = pd.DataFrame({"ratio": [1, 2, 3, 4], "peroxide": [2, 4, 6, 8]})
    corr = df.corr(method="spearman")
    plt.figure(figsize=(3, 3))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(im, label="Spearman r")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(ASSETS / "corr_heatmap.png", dpi=150)
    plt.close()


def main():
    fig_boxplot_anova()
    fig_corr_scatter()
    fig_corr_heatmap()
    print(f"Figures saved to {ASSETS}")


if __name__ == "__main__":
    main()
