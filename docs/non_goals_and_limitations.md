# Non-Goals and Limitations

**Who should read this:**
Regulators, auditors, food safety professionals, and researchers evaluating FoodSpec for production or compliance use.

**What this page covers:**
Explicit scope boundaries: what FoodSpec is NOT designed to do, and fundamental scientific/operational limitations.

**When to review:**
- Before deploying FoodSpec to make high-stakes decisions (regulatory, safety, product release)
- If considering FoodSpec as a substitute for other validation methods
- When troubleshooting unexpected results

---

## Non-Goals (What FoodSpec Does NOT Do)

### 🚫 Regulatory Certification & Legal Claims

FoodSpec is **not** designed for and **must not** be used for:

- ❌ Regulatory certification (ISO, FSSC 22000, FDA clearance, etc.)
- ❌ Legal/contractual claims of authenticity, purity, or safety
- ❌ Root cause determination in food incidents or recalls
- ❌ Compliance substitutes for mandated reference methods
- ❌ Pass/fail decisions in food border control or customs

**Why:** FoodSpec supports exploratory and screening analysis. Regulatory/legal decisions require full method validation per ISO/regulatory guidelines, audited chains of custody, and institutional liability structures FoodSpec cannot provide.

---

### 🚫 Real-Time Process Control Without Human Oversight

FoodSpec is **not** a plug-and-play in-line sensor for:

- ❌ Autonomous production line shutdowns or ingredient rejections
- ❌ Closed-loop control without human review and approval
- ❌ Unattended decision-making in high-throughput operations

**Why:** FoodSpec results reflect the quality of input data, instrument calibration, and model assumptions. Production systems require human-in-the-loop review, environmental monitoring, and feedback mechanisms.

---

### 🚫 Absolute Purity/Safety Determination

FoodSpec supports:
- ✅ Detection of *likely* adulterants or anomalies
- ❌ Absolute proof of "purity" or "safety"
- ❌ Detection of compounds below limit-of-detection
- ❌ Pathogen/microbiological screening

**Why:** Spectroscopy cannot detect what is not spectrally active. A "clean" spectrum does not guarantee absence of odorless, colorless, or spectrally silent contaminants.

---

## Scientific Limitations

### Sample-Dependent Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Heterogeneity** | Bulk spectra average over 1 mm³–cm³; local inhomogeneity lost | Use replicate sampling, document texture/phase state |
| **Liquid vs. Solid** | Solid samples require careful baseline; liquids risk evaporation/settling | Standardize sample prep; verify reproducibility |
| **Particle size** | Scattering increases with particle size; baseline instability | Pre-specify grinding/sieving; validate on reference materials |
| **Optical path length** | Varies with sample geometry (powders, paste, films); affects intensities | Use fixed-geometry cuvettes or standardize mounting |
| **Temperature sensitivity** | Raman/FTIR band positions shift ~0.1–0.5 cm⁻¹/°C; affects discrimination models | Control temperature; document thermal history |

### Instrument-Related Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Calibration drift** | Unmanaged laser power, detector gain, or grating shifts degrade model performance | Routine (daily/weekly) reference material checks |
| **Baseline instability** | Cosmic rays, fluorescence, detector noise create spurious features | Use robust baseline correction; exclude high-noise wavenumbers |
| **Saturation/detector clipping** | Overexposed samples lose spectral detail; underexposed samples have poor SNR | Optimize integration times per sample type |
| **Spectral resolution** | Low resolution blurs nearby peaks; obscures subtle adulterants | Document instrument specifications; test on validation set |

### Statistical & Model Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Small sample sizes** | Models overfit; validation estimates unreliable (<n=30) | Plan studies with statistical power; use nested CV or external test set |
| **Class imbalance** | Rare classes underrepresented; model biased toward majority | Use stratified sampling, reweighting, or synthetic sampling if justified |
| **Batch effects** | Instrument/time/operator variations confound biological signal | Use batch-aware CV folds; include batch controls in study design |
| **Confounding variables** | Unobserved factors (cultivar, harvest time, storage) correlate with adulterant | Design orthogonal experiments; document metadata thoroughly |
| **Limited feature interpretability** | High-dimensional models (PLS, neural networks) can fit noise; band assignments ambiguous | Use SHAP/permutation importance; validate on held-out test set; compare across models |

---

## Operational Limitations

### Data Requirements

- **Minimum replicate count:** Recommend ≥3 replicates per sample/condition (more for high-variability matrices)
- **Training set size:** Models with <30 samples per class are prone to overfitting; cross-validation estimates unreliable
- **Holdout test set:** FoodSpec's validation metrics assume independent test set; if unavailable, use nested CV or permutation tests
- **Missing data:** FoodSpec preprocessing assumes complete spectra; missing wavenumber regions require case-by-case handling

### Preprocessing Irreversibility

- Once preprocessing is applied (baseline correction, normalization, feature extraction), the original spectrum is lost
- Model predictions depend on preprocessing choices; changing preprocessing may require model retraining
- Preprocessing parameter choices are often empirical; optimal values dataset-dependent

### Model Generalization

- Models trained on oils may not generalize to other lipids (fats, shortenings) or non-lipid matrices
- Spectral baselines, scaling factors, and optimal preprocessing differ by instrument and sample type
- Deployment to a different instrument, lab, or time period requires validation (at minimum, test set evaluation)

---

## Known Misuse Patterns & How to Avoid Them

### ❌ "Golden Run" Mindset
**Problem:** Training a model on one "perfect" experiment, then expecting it to work on real production samples.

**Reality:** Production data are noisier, more variable, and may have confounders absent from controlled runs.

**Mitigation:** Explicitly reserve a diverse, independent test set. Include production samples in training or use domain adaptation techniques.

---

### ❌ Ignoring Batch Effects
**Problem:** Fitting a single global model across different instruments, dates, or operators without accounting for shifts.

**Reality:** Batch effects can be as large as or larger than biological signal.

**Mitigation:** Use batch-aware CV (fold by batch). Include batch as a covariate or use batch correction (ComBat, SVA) before modeling. See [Study Design](stats/study_design_and_data_requirements.md).

---

### ❌ Feature Overinterpretation
**Problem:** Assuming that a feature's importance in a black-box model has direct chemical meaning.

**Reality:** Feature importance reflects correlation with the target in the training set, not causation. High importance can reflect confounding or noise.

**Mitigation:** Validate features on independent data. Use interpretability tools (SHAP, permutation importance). Cross-validate model structure. See [Interpretability](ml/model_interpretability.md).

---

### ❌ Trusting "Too Good" Accuracy
**Problem:** Celebrating 99% accuracy or R² = 0.99 without investigating how.

**Reality:** Such results often indicate data leakage, batch confounding, or overfitting.

**Mitigation:** Check for leakage (same sample in train and test). Examine residual distributions and feature importance. Use external test set or nested CV.

---

### ❌ Single-Replicate Predictions
**Problem:** Using a model to predict a single spectrum without replicates.

**Reality:** A single spectrum is noisy; natural variability may exceed model discrimination ability.

**Mitigation:** Always take ≥3 replicates. Report confidence intervals or error bounds. See [Study Design](stats/study_design_and_data_requirements.md).

---

### ❌ Model as Ground Truth
**Problem:** Treating FoodSpec model predictions as more reliable than reference methods.

**Reality:** FoodSpec is an indirect, correlative method. Spectral features may be confounded or unstable.

**Mitigation:** FoodSpec should screen, guide, or support decisions—not replace reference methods. Combine with orthogonal evidence.

---

## When to Contact FoodSpec Developers or Domain Experts

Consider seeking expert review if:

1. **Unusual accuracy:** Validation metrics (accuracy, R², AUC) exceed 95% without clear explanation.
2. **Negative results or instability:** Severe class imbalance, batch effects, or confounding are suspected.
3. **New application domain:** Shifting from oils to fats, non-lipids, or novel matrix types.
4. **Regulatory or legal context:** Any decision affecting product safety, regulatory claims, or litigation.
5. **Model interpretation:** Questions about which spectral features or preprocessing steps drive predictions.

---

## Summary

| Aspect | What FoodSpec Does | What FoodSpec Does NOT Do |
|--------|-------------------|--------------------------|
| **Screening & exploration** | ✅ Identify likely adulterants, anomalies, or quality trends | ❌ Prove absolute purity or safety |
| **Decision support** | ✅ Provide rapid preliminary results to guide further testing | ❌ Replace regulatory reference methods or human review |
| **Research** | ✅ Correlate spectral patterns with chemical/biological properties | ❌ Guarantee causation or mechanistic insight |
| **Reproducibility** | ✅ Reproducible within same instrument/operator/batch | ❌ Guaranteed transfer across instruments without validation |
| **Automation** | ✅ Speed up routine analysis or high-throughput screening | ❌ Enable autonomous critical decisions without human oversight |

---

## See Also

- [Study Design and Data Requirements](stats/study_design_and_data_requirements.md) — How to plan robust FoodSpec studies
- [Model Evaluation and Validation](ml/model_evaluation_and_validation.md) — How to assess and validate models
- [Reporting Guidelines](troubleshooting/reporting_guidelines.md) — How to communicate FoodSpec results responsibly

