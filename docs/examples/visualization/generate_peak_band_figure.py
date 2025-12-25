"""Generate an annotated peak/band spectrum figure.
Saves docs/assets/peak_band_annotation.png.
"""
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(1)
wavenumbers = np.linspace(600, 1800, 400)
true = (
    0.8 * np.exp(-0.5 * ((wavenumbers - 1000) / 25) ** 2)
    + 0.6 * np.exp(-0.5 * ((wavenumbers - 1450) / 35) ** 2)
)
noise = 0.02 * rng.normal(size=wavenumbers.size)
spectrum = true + noise

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(wavenumbers, spectrum, label="Spectrum", color="C0")

# Annotate peaks
peak_positions = [1000, 1450]
peak_labels = ["Peak A", "Peak B"]
for pos, lab in zip(peak_positions, peak_labels):
    ax.axvline(pos, color="C1", linestyle="--", alpha=0.7)
    ax.text(pos + 5, spectrum.max() * 0.9, lab, rotation=90, color="C1", va="top")

# Annotate a band region
band_min, band_max = 980, 1040
ax.fill_between(
    wavenumbers,
    0,
    spectrum,
    where=(wavenumbers >= band_min) & (wavenumbers <= band_max),
    color="C2",
    alpha=0.2,
    label="Band area (integration)",
)

ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Annotated peaks and band integration")
ax.legend()
fig.tight_layout()
fig.savefig("docs/assets/peak_band_annotation.png", dpi=150)
plt.close(fig)
