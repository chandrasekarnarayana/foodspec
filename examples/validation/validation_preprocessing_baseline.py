# %% [markdown]
# # Validation: Baseline Correction Methods
#
# Synthetic Raman-like spectra with polynomial baseline and Gaussian peaks.
# This script compares ALSBaseline, RubberbandBaseline, and PolynomialBaseline.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from foodspec.preprocess.baseline import ALSBaseline, PolynomialBaseline, RubberbandBaseline


def generate_synthetic_spectrum(n_points=750):
    wn = np.linspace(500, 2000, n_points)
    baseline = 0.00008 * (wn - 500) ** 2 + 0.05 * (wn - 500)
    peaks = (
        1.0 * np.exp(-0.5 * ((wn - 800) / 20) ** 2)
        + 0.8 * np.exp(-0.5 * ((wn - 1200) / 25) ** 2)
        + 1.2 * np.exp(-0.5 * ((wn - 1700) / 18) ** 2)
    )
    noise = np.random.default_rng(0).normal(0, 0.02, size=wn.shape)
    spectrum = baseline + peaks + noise
    return wn, spectrum, baseline


def compute_metrics(wn, spectrum, baseline, corrected, estimated_baseline):
    peak_centers = [800, 1200, 1700]
    window = 40
    mask = np.ones_like(wn, dtype=bool)
    for c in peak_centers:
        mask &= (wn < c - window) | (wn > c + window)
    pre_mean = spectrum[mask].mean()
    post_mean = corrected[mask].mean()
    rmse = np.sqrt(np.mean((estimated_baseline - baseline) ** 2))
    return pre_mean, post_mean, rmse


def main():
    wn, spectrum, baseline = generate_synthetic_spectrum()
    methods = {
        "ALS": ALSBaseline(lambda_=1e5, p=0.01, max_iter=15),
        "Rubberband": RubberbandBaseline(),
        "Polynomial": PolynomialBaseline(degree=2),
    }
    results = []

    fig, axes = plt.subplots(2, len(methods), figsize=(14, 6), sharex=True)

    for idx, (name, transformer) in enumerate(methods.items()):
        # estimate baseline by subtracting corrected from original
        corrected = transformer.fit_transform(spectrum[None, :])[0]
        estimated = spectrum - corrected
        pre_mean, post_mean, rmse = compute_metrics(wn, spectrum, baseline, corrected, estimated)
        results.append((name, pre_mean, post_mean, rmse))

        axes[0, idx].plot(wn, spectrum, label="original", alpha=0.6)
        axes[0, idx].plot(wn, baseline, label="true baseline", linestyle="--")
        axes[0, idx].plot(wn, estimated, label="estimated baseline")
        axes[0, idx].set_title(f"{name} baseline")
        axes[0, idx].invert_xaxis()

        axes[1, idx].plot(wn, corrected, label="corrected")
        axes[1, idx].set_title(f"{name} corrected")
        axes[1, idx].invert_xaxis()

    axes[0, 0].legend()
    axes[1, 0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1, 0].set_ylabel("Intensity")
    fig.tight_layout()
    plt.savefig("validation_baseline.png", dpi=150)
    plt.show()

    print("=== Baseline correction summary (non-peak mean, RMSE) ===")
    for name, pre_mean, post_mean, rmse in results:
        print(
            f"{name}: pre-mean={pre_mean:.4f}, post-mean={post_mean:.4f}, "
            f"improvement={(abs(pre_mean)/max(abs(post_mean),1e-9)):.1f}x, baseline RMSE={rmse:.4f}"
        )


if __name__ == "__main__":
    main()

