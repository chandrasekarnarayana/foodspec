# Foundations: Food Spectroscopy Applications

This chapter surveys common applications of Raman, FTIR, and NIR spectroscopy across food matrices. It provides concrete examples that motivate later preprocessing, feature extraction, and modeling choices.

## 1. Edible oils and fats
- **Questions:** What oil type is present? Is it adulterated (e.g., cheap oil mixed into EVOO)? Has heating/oxidation altered quality?
- **Signatures:** C=C stretch (~1655 cm⁻¹), ester C=O (~1740–1750 cm⁻¹), CH2/CH3 stretches (2800–3000 cm⁻¹), trans vs cis markers.
- **Typical tasks:** Classification (oil_type), ratio monitoring (unsaturation), regression (mixture fractions).
- **See also:** [Oil authentication workflow](../workflows/oil_authentication.md), [Mixture analysis](../workflows/mixture_analysis.md).

## 2. Dairy, meat, and protein-rich products
- **Questions:** Species/source verification, spoilage/freshness, protein/lipid ratios.
- **Signatures:** Amide I/II (≈1650/1550 cm⁻¹), CH stretches, lipid bands; water OH (FTIR).
- **Typical tasks:** Classification; QC via one-class models; trend analysis during processing.
- **See also:** [Domain templates](../workflows/domain_templates.md).

## 3. Microbial identification and contamination
- **Questions:** Which species/strain is present? Are there out-of-distribution signals?
- **Signatures:** Nucleic acids, proteins (amide), specific polysaccharide/cell-wall features; often subtle and require careful preprocessing.
- **Typical tasks:** Multi-class classification; novelty detection; clustering for exploratory analysis.
- **See also:** [Batch quality control](../workflows/batch_quality_control.md).

## 4. Spices, grains, and plant materials
- **Questions:** Authenticity, adulteration, origin/varietal, moisture content.
- **Signatures:** Polysaccharide bands (C–O–C), phenolics/aromatics, carotenoids (Raman), moisture OH (NIR/FTIR).
- **Typical tasks:** Classification and regression (moisture/protein), fingerprint similarity.

## 5. Process and quality monitoring
- **Questions:** Real-time tracking of frying/heating, fermentation, drying, or storage effects.
- **Signatures:** Time-dependent changes in unsaturation bands, Maillard/oxidation indicators, water/protein shifts.
- **Typical tasks:** Ratio trends vs time/temperature; regression/ANOVA; QC alarms.
- **See also:** [Heating quality monitoring](../workflows/heating_quality_monitoring.md).

## 6. Hyperspectral imaging
- **Questions:** Spatial distribution of components/contaminants in surfaces or slices.
- **Signatures:** Same as above, but per pixel; enables mapping and segmentation.
- **Typical tasks:** Pixel-wise classification, ratio/intensity maps, cluster maps.
- **See also:** [Hyperspectral mapping](../workflows/hyperspectral_mapping.md).

## Summary
- Food spectroscopy spans authentication, adulteration, process monitoring, and spatial mapping.
- Raman/FTIR/NIR modalities highlight different bonds; choose by matrix and question.
- Later chapters show how preprocessing and models are adapted per application.

## Further reading
- [Spectroscopy basics](spectroscopy_basics.md)
- [Feature extraction](../../preprocessing/feature_extraction/)
- [Classification & regression](../ml/classification_regression.md)
- [Mixture models](../ml/mixture_models.md)

---

## When Results Cannot Be Trusted

⚠️ **Red flags for food spectroscopy applications:**

1. **Single-source validation (all "olive oil" from one producer; all "adulterated" from single batch)**
   - Intra-source variability unknown; model may learn source-specific patterns
   - Generalization to other sources unverified
   - **Fix:** Include multiple sources, varieties, origins; validate across independent suppliers

2. **Matrix effects ignored (oil spectra compared to dairy spectra without accounting for matrix differences)**
   - Different food matrices have different spectral baselines, absorption
   - Direct comparison invalid
   - **Fix:** Normalize by matrix; use matrix-matched standards; validate within-matrix first

3. **Aging/storage effects not controlled (samples of different ages compared as if equivalent)**
   - Oxidation, ripening, degradation change spectra over time
   - Age confounds with treatment
   - **Fix:** Control storage time; document sample age; include age as covariate

4. **Adulteration detection without testing at realistic levels (model validated on 50% adulteration, deployed at 1%)**
   - Detection limit unestablished; method sensitivity unknown
   - May miss realistic adulteration levels
   - **Fix:** Test at 0.5%, 1%, 2%, 5%, 10%, 20%; report limit of detection

5. **Spectroscopy method choice not justified (using NIR for C=C bands best detected by Raman)**
   - Different methods have different sensitivities
   - Mismatch reduces performance
   - **Fix:** Match method to analyte; justify choice based on band strengths; benchmark alternatives

6. **Reference method comparisons missing (spectroscopy results not validated by HPLC/GC-MS/wet chemistry)**
   - Spectroscopy is indirect; orthogonal validation essential
   - Can't confirm chemical interpretation without reference method
   - **Fix:** Validate key findings with orthogonal methods; report agreement
