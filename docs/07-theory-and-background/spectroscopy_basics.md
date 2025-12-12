# Theory – Spectroscopy Basics (Raman/FTIR)

This page provides a concise background for FoodSpec users. For deeper coverage, see `docs/foundations/spectroscopy_basics.md`.

## Fundamentals
- Raman: inelastic scattering; sensitive to molecular vibrations. FTIR: absorption of IR radiation; both yield fingerprint spectra.
- Key regions for edible oils: carbonyl (∼1740 cm⁻¹), unsaturation (∼1650 cm⁻¹), CH₂ bending/twisting (∼1430/1290 cm⁻¹), CH stretch (∼2720–3000 cm⁻¹).
- For chips/matrices: similar bands plus matrix-specific contributions (starch, proteins) that can shift intensity ratios.

## Instrument factors
- Laser wavelength, grating, objective, integration time affect signal intensity and baseline.
- Fluorescence, cosmic rays, and baseline drift are common artifacts addressed by preprocessing (ALS, smoothing, spike removal).

## Why it matters in FoodSpec
- Band/ratio behavior underlies discrimination, stability, and trend analyses in FoodSpec protocols.
- For practical steps, see [cookbook_preprocessing.md](../03-cookbook/cookbook_preprocessing.md) and [oil_discrimination_basic.md](../02-tutorials/oil_discrimination_basic.md).
