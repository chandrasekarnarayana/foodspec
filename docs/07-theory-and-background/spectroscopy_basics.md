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

---

## When Results Cannot Be Trusted

⚠️ **Red flags for spectroscopy data interpretation:**

1. **Wavenumber calibration not verified (assuming wavenumber axis accurate without polystyrene/neon reference)**
   - Laser frequency drift or grating misalignment shifts wavenumber axis
   - Band assignments off; peak isolation wrong
   - **Fix:** Include polystyrene or neon reference with every measurement; verify wavenumber accuracy <±2 cm⁻¹

2. **Baseline artifacts mistaken for real spectral features (spectral envelope from baseline follows sample curves)**
   - Poor baseline correction can create false peaks
   - Ratio calculations affected
   - **Fix:** Visualize baseline alone; remove baseline, ensure smooth baseline with no sharp residuals

3. **Peak assignment without fine structure inspection (assuming 1740 cm⁻¹ is ester C=O without checking full carbonyl region)**
   - Overlapping peaks in same region; single assignment oversimplifies
   - Deconvolution or second-derivative may reveal true structure
   - **Fix:** Examine full region ±10 cm⁻¹ around assignment; use second-derivative or deconvolution

4. **Fluorescence or autofluorescence contaminating signal (strong background under Raman, especially in visible Raman)**
   - Fluorescence dominates weak Raman peaks
   - Obscures spectral features
   - **Fix:** Use NIR or 785/1064 nm Raman to reduce fluorescence; subtract fluorescence baseline

5. **Cosmic rays or spikes not removed (single outlier pixel in spectra affects average/ratio)**
   - Cosmic rays create false peaks
   - Averaging contaminated data spreads artifact
   - **Fix:** Inspect raw spectra; remove cosmic rays before averaging; use robust statistical methods

6. **Non-reproducible spectra (same sample gives different peaks/intensities across measurements, attributed to real variation)**
   - Instrument drift, laser power variation, or sample positioning issues
   - Lack of reproducibility indicates experimental problem, not chemical variation
   - **Fix:** Check laser power stability; verify sample positioning; measure blanks/standards frequently

7. **Matrix effects unrecognized (comparing spectra of oil in different solvents, assuming differences are real)**
   - Solvent, cuvette, temperature profoundly affect spectra
   - Differences attributable to matrix, not analyte chemistry
   - **Fix:** Control matrix; use same solvent, cuvette, temperature for comparable samples

8. **Intensity ratio affected by concentrations not accounted for (comparing peak A/B at different sample concentrations)**
   - Beer-Lambert law: intensity proportional to concentration; comparing spectra at different concentrations misleading
   - Ratios change with sample amount, not just chemistry
   - **Fix:** Normalize by total intensity or concentration; use absorbance if available
