# FoodSpec Protocol Library (edible oils & food)

**EdibleOil_Classification_v1**  
When to use: RQ1–RQ3 discrimination between oils (VO/PO/OO/CO, etc.).  
What it does: Baseline + smoothing + reference normalization, extracts core peaks, computes ratios, runs discrimination (p-values, RF importance), minimal panel, clustering.  
Inputs: oil_type, matrix, heating_stage + Raman peaks or raw spectra.

**EdibleOil_Heating_Stability_v1**  
When to use: RQ2–RQ4 thermal stability and degradation markers.  
What it does: Similar preprocessing, focuses on heating trends, stability (CV/MAD), trend p-values with FDR.  
Inputs: oil_type, heating_stage, matrix.

**Chips_vs_Oil_MatrixEffects_v1**  
When to use: Compare chips vs pure oil behaviors; find divergent markers.  
What it does: Same peak set; looks at mean/Trend/CV differences, effect sizes, FDR-adjusted divergences.  
Inputs: matrix (oil/chips), oil_type, heating_stage.

**MinimalMarkerPanel_Screening_v1**  
When to use: Identify compact QC marker panels (RQ5).  
What it does: L1-logreg-based selection, RF CV accuracy, reports smallest panel near best accuracy.  
Inputs: oil_type, matrix, heating_stage.

All protocols share baseline/smoothing/reference normalization defaults (ALS λ=1e5, p=0.01, SG window=9, poly=3) and a common peak set (1742, 1652, 1434, 1296, 1259, 2720). Ratios are primarily vs 2720. Adjust parameters directly in the YAMLs if needed.
