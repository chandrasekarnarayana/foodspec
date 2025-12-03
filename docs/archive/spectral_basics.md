# Spectroscopy basics (Raman/FTIR)

## What is a spectrum?
A spectrum is a curve showing how light is absorbed or scattered as a function of the **wavenumber** (x‑axis) with corresponding **intensity** (y‑axis). Peaks and shoulders reveal vibrational modes of chemical bonds in the sample.

## Wavenumber (cm⁻¹)
- Wavenumber is the reciprocal of wavelength (1/λ) and is reported in cm⁻¹.
- It is the standard unit in vibrational spectroscopy because it aligns directly with energy levels and vibrational transitions.
- Axes must be monotonic (increasing or decreasing consistently) for proper interpolation, preprocessing, and modeling.

## Raman vs FTIR (intuitive differences)
- **Raman**: measures inelastic scattering; good for aqueous systems, often affected by fluorescence background; cosmic-ray spikes may appear.
- **FTIR**: measures absorbance; sensitive to water and CO₂ interference; ATR accessories introduce depth-dependent effects.
- Both probe molecular vibrations but with different selection rules and sensitivities; preprocessing choices differ accordingly.

## Typical spectral ranges for food
- **Fingerprint region**: ~600–1800 cm⁻¹ (rich in C–C, C–O, C=O, C=C, and CH bending).
- **High wavenumber (CH stretch)**: ~2800–3100 cm⁻¹ (CH₂/CH₃ stretching).
- Some instruments provide wider ranges; crop to informative regions for stability.

## How to read a spectral plot
- **Axes**: x = wavenumber (cm⁻¹), y = intensity (a.u.). Ensure units are noted.
- **Peaks**: sharp or broad maxima; indicate specific vibrational modes (e.g., 1655 cm⁻¹ for C=C, 1742 cm⁻¹ for C=O in oils).
- **Shoulders**: subtle features adjacent to main peaks; can indicate overlapping bands.
- **Baseline**: background level; may slope (FTIR) or rise (Raman fluorescence). Proper baseline correction makes peaks interpretable.
- **Noise**: random fluctuations; smoothing reduces noise but should preserve peak shape.
