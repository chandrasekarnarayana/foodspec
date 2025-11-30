# %% [markdown]
# # Validation: Peak Ratios (1655 / 1742)
#
# Synthetic spectra with controlled peak height ratios; compare true vs measured ratios.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from foodspec.features.peaks import PeakFeatureExtractor
from foodspec.features.ratios import RatioFeatureGenerator


def generate_spectra(n_samples=20):
    rng = np.random.default_rng(0)
    wn = np.linspace(1500, 1800, 400)
    heights_1655 = rng.uniform(0.8, 1.2, size=n_samples)
    ratios_true = rng.uniform(0.5, 1.5, size=n_samples)
    heights_1742 = heights_1655 / ratios_true
    spectra = []
    for h1, h2 in zip(heights_1655, heights_1742):
        spec = (
            h1 * np.exp(-0.5 * ((wn - 1655) / 10) ** 2)
            + h2 * np.exp(-0.5 * ((wn - 1742) / 10) ** 2)
        )
        spec += rng.normal(0, 0.01, size=wn.shape)
        spectra.append(spec)
    return wn, np.vstack(spectra), ratios_true


def main():
    wn, X, ratios_true = generate_spectra()
    extractor = PeakFeatureExtractor(expected_peaks=[1655.0, 1742.0], tolerance=6.0, features=("height",))
    feats = extractor.fit_transform(X, wavenumbers=wn)
    feat_df = {
        "peak_1655_height": feats[:, 0],
        "peak_1742_height": feats[:, 1],
    }
    ratio_gen = RatioFeatureGenerator({"ratio_1655_1742": ("peak_1655_height", "peak_1742_height")})
    import pandas as pd

    feat_df = pd.DataFrame(feat_df)
    ratios_meas = ratio_gen.transform(feat_df)["ratio_1655_1742"].to_numpy()

    corr = np.corrcoef(ratios_true, ratios_meas)[0, 1]
    rmse = np.sqrt(np.mean((ratios_true - ratios_meas) ** 2))

    plt.figure(figsize=(5, 5))
    plt.scatter(ratios_true, ratios_meas, alpha=0.7)
    lims = [min(ratios_true.min(), ratios_meas.min()), max(ratios_true.max(), ratios_meas.max())]
    plt.plot(lims, lims, "k--", label="ideal")
    plt.xlabel("True ratio (1655/1742)")
    plt.ylabel("Measured ratio")
    plt.title("Peak ratio validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("validation_peak_ratios.png", dpi=150)
    plt.show()

    print(f"Correlation true vs measured: {corr:.3f}")
    print(f"RMSE true vs measured: {rmse:.4f}")


if __name__ == "__main__":
    main()

