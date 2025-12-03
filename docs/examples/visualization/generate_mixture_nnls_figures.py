\"\"\"Generate NNLS overlay/residual figures for docs/assets.

Uses a simple synthetic demo: two reference spectra mixed at known fractions,
then solved with nnls_mixture. Saves overlay and residual plots.
\"\"\"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from foodspec.chemometrics.mixture import nnls_mixture


def synthetic_refs(n_points=400, noise=0.0):
    wn = np.linspace(1400, 1800, n_points)
    ref1 = (
        np.exp(-0.5 * ((wn - 1655) / 8) ** 2)
        + 0.4 * np.exp(-0.5 * ((wn - 1745) / 10) ** 2)
    )
    ref2 = (
        0.6 * np.exp(-0.5 * ((wn - 1600) / 7) ** 2)
        + np.exp(-0.5 * ((wn - 1710) / 9) ** 2)
    )
    if noise > 0:
        ref1 += noise * np.random.randn(*ref1.shape)
        ref2 += noise * np.random.randn(*ref2.shape)
    A = np.vstack([ref1, ref2]).T  # shape (n_points, 2)
    return wn, A


def main():
    wn, A = synthetic_refs()
    true_frac = np.array([0.7, 0.3])
    y = A @ true_frac

    coeffs, resid = nnls_mixture(y, A)
    y_hat = A @ coeffs
    fractions = coeffs / coeffs.sum()

    out_dir = Path(__file__).parent.parent.parent / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overlay
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(wn, y, label="Observed mixture")
    ax.plot(wn, y_hat, "--", label="Reconstructed")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend()
    fig.tight_layout()
    overlay_path = out_dir / "nnls_overlay.png"
    fig.savefig(overlay_path, dpi=200)

    # Residual
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(wn, y - y_hat, label="Residual")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Residual")
    ax.legend()
    fig.tight_layout()
    resid_path = out_dir / "nnls_residual.png"
    fig.savefig(resid_path, dpi=200)

    print(f"Fractions (true vs est): {true_frac} vs {fractions}")
    print(f"Saved {overlay_path} and {resid_path}")


if __name__ == "__main__":
    main()
