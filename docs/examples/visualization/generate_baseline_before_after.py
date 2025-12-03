"""Generate a baseline before/after illustration.
Saves docs/assets/baseline_before_after.png.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)
wavenumbers = np.linspace(600, 1800, 400)
true_signal = (
    0.7 * np.exp(-0.5 * ((wavenumbers - 1100) / 30) ** 2)
    + 0.5 * np.exp(-0.5 * ((wavenumbers - 1500) / 40) ** 2)
)
baseline = 0.00005 * (wavenumbers - 1200) ** 2 + 0.1
noise = 0.02 * rng.normal(size=wavenumbers.size)
raw = true_signal + baseline + noise

# "Corrected" simply subtracts the known baseline for illustration
corrected = raw - baseline

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(wavenumbers, raw, label="Raw (with baseline)", color="C0")
ax.plot(wavenumbers, baseline, label="Baseline", color="C2", linestyle="--")
ax.plot(wavenumbers, corrected, label="After correction", color="C1")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Baseline before/after (synthetic)")
ax.legend()
fig.tight_layout()
fig.savefig("docs/assets/baseline_before_after.png", dpi=150)
plt.close(fig)
