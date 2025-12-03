"""
Generate simple illustrative spectra showing ideal, noisy, and baseline-drift cases.

Outputs:
- docs/assets/spectra_artifacts.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def main():
    wn = np.linspace(600, 1800, 400)
    peak1 = np.exp(-0.5 * ((wn - 1000) / 20) ** 2)
    peak2 = 0.7 * np.exp(-0.5 * ((wn - 1400) / 25) ** 2)
    ideal = peak1 + peak2

    rng = np.random.default_rng(0)
    noisy = ideal + rng.normal(0, 0.05, size=wn.shape)
    baseline = 0.00002 * (wn - 1200) ** 2 + 0.02
    drifted = ideal + baseline + rng.normal(0, 0.02, size=wn.shape)

    plt.figure(figsize=(6, 3))
    plt.plot(wn, ideal, label="Ideal", linewidth=1.5)
    plt.plot(wn, noisy, label="Noisy", linewidth=1.0, alpha=0.8)
    plt.plot(wn, drifted, label="Baseline drift", linewidth=1.0, alpha=0.8)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Illustrative spectra: ideal vs noise vs baseline drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "spectra_artifacts.png", dpi=150)
    plt.close()
    print(f"Saved figure to {ASSETS/'spectra_artifacts.png'}")


if __name__ == "__main__":
    main()
