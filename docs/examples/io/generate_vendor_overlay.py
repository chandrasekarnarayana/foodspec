"""Generate a synthetic vendor overlay plot to illustrate vendor import normalization.

This does not require proprietary data. It mimics two spectra that could have been
loaded via vendor loaders (SPC/OPUS) after normalization to wavenumber/intensity
arrays. The figure is saved to docs/assets/vendor_overlay.png.
"""
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
wavenumbers = np.linspace(600, 1800, 400)

# Two synthetic spectra with slightly shifted peaks
spec_a = (
    0.5 * np.exp(-0.5 * ((wavenumbers - 1000) / 30) ** 2)
    + 0.8 * np.exp(-0.5 * ((wavenumbers - 1450) / 25) ** 2)
    + 0.02 * rng.normal(size=wavenumbers.size)
)
spec_b = (
    0.55 * np.exp(-0.5 * ((wavenumbers - 1020) / 35) ** 2)
    + 0.7 * np.exp(-0.5 * ((wavenumbers - 1470) / 30) ** 2)
    + 0.02 * rng.normal(size=wavenumbers.size)
)

plt.figure(figsize=(6, 4))
plt.plot(wavenumbers, spec_a, label="Vendor spectrum A (normalized)")
plt.plot(wavenumbers, spec_b, label="Vendor spectrum B (normalized)", alpha=0.8)
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity (a.u.)")
plt.title("Synthetic SPC/OPUS-style overlay after normalization")
plt.legend()
plt.tight_layout()
plt.savefig("docs/assets/vendor_overlay.png", dpi=150)
plt.close()
