# Foundations: Spectroscopy Basics

This chapter introduces vibrational spectroscopy for food science: what spectra are, how wavenumbers are used, and how Raman, FTIR, and NIR differ. It anchors the physics so later preprocessing and ML chapters have a common starting point.

## 1. What is a spectrum?
- A spectrum plots **intensity** vs **wavenumber** (cm⁻¹). Wavenumber \( \tilde{\nu} = 1/\lambda \) is preferred because it scales linearly with energy.
- Peaks correspond to vibrational modes of molecules (stretching, bending). Food matrices contain lipids, proteins, carbohydrates, water—each with characteristic bands.
- Always store axes in ascending cm⁻¹ for computational pipelines.

## 2. Raman vs FTIR vs NIR
- **Raman (inelastic scattering):** Measures shifts relative to laser line (Stokes/anti-Stokes). Good for aqueous samples; sensitive to symmetric stretches (e.g., C=C, CH).
- **FTIR (absorption):** Measures molecular absorption; ATR-FTIR is common in food labs. Strong for polar bonds (C=O, O–H).
- **NIR (overtones/combination bands):** Broader, weaker features; useful for bulk composition and rapid QC.

### Typical food spectral regions (examples)
- **Fingerprint (600–1800 cm⁻¹):** C–C, C–O, C=O; unsaturation bands (≈1655–1745 cm⁻¹) in oils; amide bands (protein) around 1650/1550 cm⁻¹.
- **CH stretching (2800–3100 cm⁻¹):** Lipid/protein CH2/CH3 bands.
- **OH/NH (3200–3600 cm⁻¹):** Water/protein hydrogen bonding (FTIR).

## 3. Peak shapes, baselines, and artifacts
- **Peaks/bands:** Can be sharp (Raman) or broad (NIR). Shoulders often encode overlapping modes.
- **Baseline & fluorescence:** Raman often has fluorescence backgrounds; FTIR can show sloping baselines due to ATR contact or scattering.
- **Atmospheric lines:** Water/CO₂ in FTIR; remove or account for them in preprocessing.
- **Noise & scatter:** Instrument noise, cosmic rays (Raman spikes), path-length/contact variation.

## 3a. Vibrational modes and spectral signatures
- **Stretching vs bending:** stretching changes bond length; bending changes bond angles. Raman favors polarizability changes (e.g., C=C), FTIR favors dipole changes (e.g., O–H).
- **Food-relevant bands (cm⁻¹, illustrative):**
  - **FTIR synthetic example (generated via `generate_synthetic_ftir_spectrum`):** O–H stretch (~3300), C–H stretches (2800–3000), ester C=O (~1740), CH₂ bend (~1450), C–O stretch (~1050), fingerprint 800–1500. Plot wavenumber vs absorbance and label each band with the mode and a food interpretation (e.g., ester C=O in lipids).
  - **Raman synthetic example (generated via `generate_synthetic_raman_spectrum`):** discrete peaks at ~717 (C–C stretch), 1265 (cis =C–H bend), 1440 (CH₂ bend), 1655 (C=C stretch). Annotate peaks and note how intensity shifts relate to unsaturation/saturation.
- **Interpretation:** shifts or intensity changes in these bands map to composition (unsaturation, ester content, moisture). Synthetic plots (see plotting helpers) mirror real bands observed in oils/fats.
- For notation/abbreviations, see the [Glossary](../glossary.md). For a practical bands/ratios guide, see [Feature extraction](../preprocessing/feature_extraction.md#how-to-choose-bands-and-ratios-decision-mini-guide).

## 4. Sampling and instrument notes
- Laser wavelength (Raman) affects fluorescence and penetration; ATR crystal choice (FTIR) affects depth of penetration.
- Resolution: finer spacing yields more data points but may increase noise.
- Export formats: vendor-specific to TXT/CSV. FoodSpec standardizes via CSV → HDF5; see [CSV → HDF5 pipeline](../csv_to_library.md).

## 5. Choosing a modality for food tasks
- **Authentication/adulteration:** Raman/FTIR fingerprint region for oils, spices; NIR for rapid screening.
- **Heating/oxidation studies:** Track unsaturation bands (1650–1750 cm⁻¹) and CH stretches.
- **Protein-rich samples (dairy/meat):** Amide bands (FTIR/Raman); CH stretches.
- **Water-dominated matrices:** Raman often preferred to avoid strong water absorption in FTIR.

## 6. Links to computation
- Baseline drift and fluorescence → [Baseline correction](../preprocessing/baseline_correction.md).
- Scatter/contact effects → [Normalization & smoothing](../preprocessing/normalization_smoothing.md) and [Scatter & cosmic-ray handling](../preprocessing/scatter_correction_cosmic_ray_removal.md).
- High dimensionality → [PCA](../ml/pca_and_dimensionality_reduction.md).

## Summary
- Wavenumber in cm⁻¹ is the standard axis; keep spectra monotonic.
- Raman, FTIR, and NIR emphasize different vibrational modes; choose modality by matrix and question.
- Baselines, fluorescence, atmospheric lines, and scatter are common artifacts to mitigate in preprocessing.

## Further reading
- [Baseline correction](../preprocessing/baseline_correction.md)
- [Normalization & smoothing](../preprocessing/normalization_smoothing.md)
- [PCA and dimensionality reduction](../ml/pca_and_dimensionality_reduction.md)
- [CSV → HDF5 pipeline](../csv_to_library.md)
