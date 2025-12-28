# Methods Text Generator — Journal-Ready Methods Section

**Who:**
Authors preparing FoodSpec-based analyses for publication; reviewers evaluating methodology transparency.

**What:**
A structured template for generating a journal-ready "Methods" section from a FoodSpec analysis, with guidance on what to include, what to hide, and what to emphasize for different audiences.

**When:**
Use before submitting manuscripts or regulatory documentation.

**When NOT:**
Do not use as a substitute for domain-specific method validation or ISO/regulatory requirements.

**Key Assumptions:**
- Analysis followed the [Reference Protocol](reference_protocol.md) (or equivalent rigor)
- Complete metadata and code/config are available
- Data can be shared (or appropriately anonymized/restricted)

**What can go wrong:**
- Omitting crucial preprocessing details → irreproducibility
- Overstating generalizability → overfitting in readers' minds
- Underreporting validation → unrealistic confidence in results

---

## Template: FoodSpec Methods Section (Raman/FTIR Classification)

---

### [SECTION I] Samples and Data Collection

#### Sample Selection
[FILL IN: How many samples? From which sources? What classes?]

> *Example:* "We analyzed 120 edible oils: 60 authentic olive oils (sourced from certified producers across three harvest years) and 60 adulterated samples (prepared by mixing authentic olive oil with refined seed oil at 10%, 25%, and 50% v/v). Ground truth was established by gas chromatography (GC-FID, ...see Appendix)."

**Metadata to include:**
- Sample ID scheme
- Source/supplier
- Authentication method (if not obvious)
- Storage conditions and duration
- Lot/batch information

#### Spectral Acquisition
[FILL IN: Instrument model, settings, number of replicates]

> *Example for Raman:*
> "Raman spectra were acquired on a [Instrument Model, Vendor] equipped with a 532 nm laser, 20× objective, 300 grooves/mm grating, spectral resolution ~1 cm⁻¹. Each sample was measured in triplicate, with 10 s integration time per scan. Laser power was [X mW], attenuated to minimize sample damage."

> *Example for FTIR:*
> "FTIR spectra were acquired on a [Instrument Model] with attenuated total reflectance (ATR) sampling, 4 cm⁻¹ resolution, 400–4000 cm⁻¹ range, 32 scans per sample, with air background every 10 samples. Each oil sample was measured in triplicate (10 µL per replicate, fresh vial each time)."

**Minimum details to report:**
- Instrument make/model
- Laser wavelength or IR range
- Spectral resolution
- Sampling geometry (ATR, transmission, diffuse reflection)
- Integration time / number of scans
- Temperature control (if relevant)
- Replication scheme (technical, intra-day, inter-day)

---

### [SECTION II] Data Preprocessing and Feature Engineering

[FILL IN: Preprocessing steps, order, parameters, rationale]

#### Preprocessing Pipeline

> *Example:*
> "Spectra were preprocessed in the following order:
> 1. Cosmic ray removal (automated spike detection, threshold = 3σ above local median)
> 2. Baseline correction (asymmetric least squares, λ = 100, p = 0.01)
> 3. Smoothing (Savitzky–Golay filter, 7-point window, 2nd-order polynomial)
> 4. Normalization (L2 unit vector normalization)
> 5. Feature extraction (peak heights at [list specific bands, cm⁻¹], see Table 1)
> 
> Preprocessing was applied identically to training and test data, with preprocessing parameters determined on training data only (to avoid leakage). All preprocessing was implemented in [Python/R package, version]."

**Parameters to report:**
- Baseline correction algorithm and hyperparameters (λ, p, etc.)
- Smoothing window size and polynomial order
- Normalization method
- Wavenumber regions excluded (if any) and why
- Software/version used

#### Feature Engineering (if applicable)

If using hand-crafted features instead of full-spectrum ML:

> *Example:*
> "We extracted 12 spectral features: peak heights at 1740 cm⁻¹ (C=O), 1655 cm⁻¹ (C=C), 1600 cm⁻¹ (aromatic C=C), and 1450 cm⁻¹ (CH₂ bending); ratios of intensities (1740/1450, 1655/1450); and full-width at half-maximum (FWHM) of selected peaks. Feature extraction was performed on preprocessed spectra using custom Python scripts (available at [repo URL])."

**Include:**
- Which features (band assignments, cm⁻¹)
- Rationale (chemistry, prior knowledge)
- Computation method

---

### [SECTION III] Model Development and Training

[FILL IN: Model type, hyperparameters, training procedure, validation strategy]

#### Model Selection and Hyperparameter Tuning

> *Example:*
> "We evaluated five model types: partial least squares (PLS) discrimination analysis, random forest, support vector machine (SVM) with RBF kernel, logistic regression, and gradient boosting. Model selection and hyperparameter optimization were performed via nested 5-fold stratified cross-validation on the training set (n=96; 24 samples per class withheld for final test set). The inner loop (5-fold CV) evaluated hyperparameter combinations (PLS: 2–15 components; RF: max_depth 5–20, n_estimators 100–500; SVM: C ∈ {0.01, 0.1, 1, 10}, γ ∈ {0.001, 0.01, 0.1, 'scale'}). The best hyperparameters from the inner loop were used to train the final model on the full training set."

**Required details:**
- Model types evaluated
- Hyperparameter ranges
- Cross-validation scheme (k-fold, stratified?, nested?)
- Optimization metric (accuracy, AUC, etc.)
- Software/version (e.g., scikit-learn 1.2.0)

#### Class Weighting and Imbalance Handling

If classes imbalanced:

> *Example:*
> "To account for class imbalance ([ratio]), we applied sample weighting inversely proportional to class frequency: weight = 1 / (n_samples_per_class / n_samples_total)."

---

### [SECTION IV] Model Validation and Performance

[FILL IN: Validation approach, test set, metrics, confidence intervals]

#### Validation Strategy

> *Example:*
> "The final model was evaluated on a held-out test set (n=24; 8 samples per class) that was not used during hyperparameter tuning. Test set samples were acquired on different dates and by a different operator than training data to assess robustness to temporal and procedural drift. Additionally, we computed 95% bootstrap confidence intervals by resampling the test set with replacement (1000 iterations) and recomputing metrics."

**Include:**
- Test set size and composition
- How test set was held out (stratified, random, by batch, etc.)
- Whether test set differs in time/operator/batch (generalization)
- Confidence interval computation method

#### Performance Metrics

> *Example for classification:*
> "Performance on the test set was quantified as follows:
> - Overall accuracy: 92.1% (95% CI: 86.3%–96.4%)
> - Class-specific recall: Authentic 95.8%, Adulterated 88.3%
> - Precision: Authentic 90.2%, Adulterated 94.7%
> - AUC-ROC: 0.968 (95% CI: 0.924–0.992)
> 
> See Table 2 for confusion matrix and per-fold cross-validation metrics."

> *Example for regression:*
> "On the test set, the model achieved:
> - RMSE: 2.1 mg/kg (95% CI: 1.6–2.8)
> - MAE: 1.6 mg/kg
> - R²: 0.891 (95% CI: 0.834–0.938)
> 
> Residual distribution was approximately normal (Shapiro–Wilk p = 0.14)."

**Must report:**
- Point estimates + 95% CI for all metrics
- Confusion matrix or per-class metrics
- Test set size and composition
- Cross-validation folds and metrics (to demonstrate consistency)

---

### [SECTION V] Feature Importance and Interpretability

[FILL IN: Which spectral bands/features are important? Why?]

> *Example for PLS:*
> "The first two PLS components explained 68% and 12% of variance in class membership, respectively (total 80%). Feature (wavenumber) loadings on PC1 highlighted the carbonyl band (1740 cm⁻¹), conjugated C=C (1655 cm⁻¹), and aromatic CH (1600 cm⁻¹), with positive loadings indicative of unsaturated/oxidized species. This pattern aligns with expected compositional differences between authentic and adulterated oils (Fig. 3)."

> *Example for black-box model:*
> "Feature importance was estimated via permutation importance: for each feature, we shuffled its values on the test set and recomputed accuracy, with importance = drop in accuracy. Top 5 features were [band A, band B, ...]. To validate that these features have interpretable chemical meaning, we performed post-hoc analysis: [correlation with known adulterant indicator, e.g., GC peak areas]. See Appendix Fig. A2."

**Include if applicable:**
- Loadings, feature importance, or SHAP values
- Which spectral regions are most informative
- Chemical/biological interpretation
- Caveats (e.g., "feature importance may reflect confounding; independent validation needed")

---

### [SECTION VI] Limitations and Failure Modes

[FILL IN: What could go wrong? When should readers be skeptical?]

> *Example:*
> "Several limitations should be noted:
> 1. *Sample diversity:* Oils were sourced from [X regions, Y suppliers]; generalization to other cultivars or harvest years is unclear.
> 2. *Adulterant scope:* We tested [specific adulterants]; other adulterants or blends may not be detected.
> 3. *Instrument transfer:* Models were trained on a single Raman instrument; transfer to different instruments requires re-validation.
> 4. *No batch effect correction:* While data were acquired across [N days, M operators], we did not explicitly correct for batch effects using methods such as ComBat. Performance may degrade if batch drift exceeds training variability.
> 5. *Small test set:* The test set (n=24) provides limited statistical power to detect small differences in model performance; external validation on an independent cohort is recommended."

**Mandatory sections:**
- Sample scope (which populations/conditions were tested?)
- Preprocessing assumptions (will preprocessing generalize?)
- Instrument/batch limitations
- Confounding or confounding variables
- Statistical caveats (small test set, imbalance, etc.)
- See [When Results Cannot Be Trusted](#when-results-cannot-be-trusted) below

---

### [SECTION VII] Data and Code Availability

> *Example:*
> "Preprocessed spectral data, trained models, and analysis code are available at [GitHub/OSF/institutional repository], with DOI [XXX], under [open/restricted access with data use agreement]. Raw spectra are [available/restricted due to proprietary instrument format] from [author name] upon request. A fully reproducible analysis script (Python 3.9, scikit-learn 1.2.0, pandas 1.5.2) is provided in the repository; running it generates all figures and tables from preprocessed data."

**Best practice:**
- Share preprocessed spectra (with privacy/IP caveats)
- Share trained model (pickle/SavedModel/ONNX)
- Share reproducible analysis code (Jupyter notebook, Python script, or Snakemake workflow)
- Specify software versions exactly

---

## When Results Cannot Be Trusted

⚠️ **Red flags for Methods section — if authors report these, be skeptical:**

1. **Perfect or near-perfect metrics without domain explanation**
   - Reported accuracy >98% or AUC >0.99 on test set
   - **Investigation needed:** Look for data leakage, batch confounding, or artificial separation
   
2. **Missing or vague validation strategy**
   - "We validated using cross-validation" but no mention of test set, CV scheme, or metrics
   - **Investigation needed:** Can't assess generalization or overfitting risk

3. **No confidence intervals or error bounds**
   - Point estimates only (e.g., "accuracy 92%"); no uncertainty
   - **Investigation needed:** Can't assess precision or compare models

4. **Training and test data from same batch/date/operator**
   - Temporal, instrument, or batch separation not mentioned
   - **Investigation needed:** Model may not generalize to new data

5. **Preprocessing details omitted or vague**
   - "Spectra were preprocessed" with no parameters or software cited
   - **Investigation needed:** Results irreproducible; unclear if preprocessing is optimal

6. **Model selection methodology unclear**
   - No hyperparameter tuning described; unclear how model was chosen
   - **Investigation needed:** May reflect cherry-picking; nested CV not used

7. **Single feature drives predictions**
   - Feature importance shows one band >>others; no investigation of confounding
   - **Investigation needed:** Likely batch effect or confounding variable

8. **No mention of batch effects or confounding**
   - Especially problematic if data from multiple instruments/dates/operators
   - **Investigation needed:** Batch effects can mimic biological signal

9. **Small sample size**
   - n < 30 samples per class; unclear if power analysis was done
   - **Investigation needed:** Results likely overfit; validation estimates unreliable

10. **Generalization claims without evidence**
    - "Model should work on any oil" but validation limited to one source/region
    - **Investigation needed:** Actual generalization scope unknown

---

## Template: Quick Checklist for Reviewers

- [ ] Methods describe sample collection, source, and authentication
- [ ] Spectral acquisition instrument, settings, and replication scheme clearly stated
- [ ] Preprocessing pipeline and parameters fully specified
- [ ] Model type, hyperparameters, and tuning procedure described
- [ ] Validation strategy uses held-out test set (not just CV)
- [ ] Metrics include point estimates + 95% CI
- [ ] Feature importance reported and interpreted
- [ ] Limitations and failure modes honestly discussed
- [ ] Code and data availability or access path provided
- [ ] No red flags from the list above

---

## See Also

- [Reference Protocol](reference_protocol.md) — The underlying methodology
- [Model Evaluation and Validation](../methods/chemometrics/model_evaluation_and_validation.md) — Metrics and interpretation
- [Non-Goals and Limitations](../non_goals_and_limitations.md) — What FoodSpec cannot do
- [Reporting Guidelines](../troubleshooting/reporting_guidelines.md) — Best practices for communicating results

