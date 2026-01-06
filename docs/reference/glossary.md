# Glossary

**Purpose:** Define key terminology for FoodSpec documentation at multiple levels (plain English, technical, mathematical).

**Audience:** Anyone encountering unfamiliar terms—from absolute beginners to domain experts.

**Time:** 2–5 minutes per term lookup.

**Prerequisites:** None (designed for all audiences).

---

## Starter Terms (Plain English / Layer 1)

### Spectrum (plural: Spectra)
**Layman:** A "fingerprint" showing how a sample reflects or absorbs different colors of light. Each food has a unique pattern.

**Food Scientist:** An array of intensity measurements across wavelengths/wavenumbers, capturing molecular vibrations (Raman/FTIR) or electronic transitions (UV-Vis).

**Physicist:** Intensity $I(\tilde{\nu})$ as a function of wavenumber $\tilde{\nu}$ (cm⁻¹), encoding vibrational transitions via $I \propto \frac{\partial\alpha}{\partial Q}$.

---

### Peak / Band
**Layman:** A bump in the fingerprint graph. Each bump represents a specific type of molecule (like a signature ingredient).

**Food Scientist:** A local maximum in a spectrum corresponding to a specific molecular vibration (e.g., carbonyl C=O at 1740 cm⁻¹).

**Physicist:** Spectral feature arising from vibrational mode $\nu_i$ with frequency $\omega_i = \sqrt{k/\mu}$ (spring constant $k$, reduced mass $\mu$).

---

### Ratio
**Layman:** Comparing two bumps in the fingerprint. Example: If bump A is twice as tall as bump B, the ratio is 2:1. This cancels out lighting differences.

**Food Scientist:** Intensity or area of one peak divided by another (e.g., $I_{1742} / I_{2720}$) to normalize illumination and focus on chemical composition changes.

**Physicist:** $R = \frac{I(\tilde{\nu}_1)}{I(\tilde{\nu}_2)}$ where $I(\tilde{\nu}_i) = I_0 \cdot \sigma(\tilde{\nu}_i) \cdot N_i$. Ratio cancels common factors ($I_0$, path length).

---

### Authentication
**Layman:** Proving food is what it claims to be (e.g., verifying olive oil isn't mixed with cheaper sunflower oil).

**Food Scientist:** Classification workflow distinguishing genuine samples from adulterants using spectral fingerprints and statistical models.

**Physicist:** Pattern recognition via supervised learning ($\mathbf{y} = f(\mathbf{X})$) where $\mathbf{X}$ are spectral features and $\mathbf{y}$ are class labels, validated with cross-validation.

---

### Preprocessing
**Layman:** Cleaning the data to remove noise and background interference—like adjusting a blurry photo before analyzing it.

**Food Scientist:** Baseline correction (removing background fluorescence/drift), smoothing (noise reduction), and normalization (standardizing intensity scales).

**Physicist:** Operations including ALS baseline ($\min_{\mathbf{b}} ||\mathbf{y} - \mathbf{b}||^2 + \lambda \sum |\Delta^2 b_i|$), Savitzky-Golay filtering, and reference peak normalization.

---

### Wavenumber Axis
**Layman:** The x-axis of your spectrum graph, showing different "colors" of infrared light from low to high energy.

**Food Scientist:** The spectral dimension measured in wavenumbers (cm⁻¹), inversely proportional to wavelength. Higher wavenumbers = higher energy vibrations.

**Physicist:** $\tilde{\nu} = \frac{1}{\lambda} = \frac{\omega}{2\pi c}$ where $\lambda$ is wavelength, $\omega$ is angular frequency, and $c$ is speed of light. For FTIR: typically 4000-400 cm⁻¹; Raman: varies by laser.

**Conventions:** Always ascending order (low → high wavenumbers). Resolution typically 2-8 cm⁻¹ for FTIR, 1-4 cm⁻¹ for Raman.

---

### Resolution (Spectral)
**Layman:** How detailed your measurement is—like the difference between regular and high-definition TV. Higher resolution shows finer details.

**Food Scientist:** The smallest wavenumber interval that can be distinguished. Lower values (e.g., 2 cm⁻¹) = better resolution = sharper peaks.

**Physicist:** $\Delta\tilde{\nu} = \frac{1}{2L}$ where $L$ is the optical path difference in Fourier-transform instruments. Determines peak width via convolution with instrumental line shape.

**Typical ranges:** FTIR: 2-8 cm⁻¹, Raman: 1-4 cm⁻¹, NIR: 4-16 cm⁻¹.

---

### Interpolation
**Layman:** Filling in the gaps when your data points don't line up—like drawing a smooth curve through dots on graph paper.

**Food Scientist:** Resampling spectra to a common wavenumber grid before analysis. Required when combining data from different instruments with different sampling points.

**Physicist:** Computing $I(\tilde{\nu}_{\text{target}})$ from measured $I(\tilde{\nu}_{\text{measured}})$ via linear, cubic spline, or sinc interpolation. Preserves peak positions but may introduce artifacts if over-sampled.

**Best practice:** Use cubic spline for smooth spectra, linear for noisy data. Never interpolate to resolution finer than instrument capability.

---

### Baseline
**Layman:** The "background hum" in your spectrum—like removing static noise from a radio signal before listening to music.

**Food Scientist:** Slowly varying background signal from fluorescence, instrument drift, or sample matrix. Must be removed to isolate true peaks.

**Physicist:** Additive offset $B(\tilde{\nu})$ where observed $I_{\text{obs}}(\tilde{\nu}) = I_{\text{true}}(\tilde{\nu}) + B(\tilde{\nu})$. Estimated via polynomial fitting, rubber-band methods, or ALS algorithm.

**Warning:** Over-correction removes real broad peaks; under-correction biases peak heights.

---

### Normalization
**Layman:** Adjusting all spectra to the same "volume level" so you can fairly compare them—like setting all songs to the same loudness.

**Food Scientist:** Scaling intensity values to account for variations in sample thickness, concentration, or illumination. Enables quantitative comparison across samples.

**Physicist:** Transformation $I'(\tilde{\nu}) = f(I(\tilde{\nu}))$ where $f$ can be:
- **SNV**: $(I - \mu_I) / \sigma_I$ (Standard Normal Variate)
- **Vector norm**: $I / ||I||_2$ or $I / ||I||_1$
- **Reference peak**: $I / I(\tilde{\nu}_{\text{ref}})$
- **MSC**: Multiplicative Scatter Correction

**Best practice:** Choose based on physics (reference peak for internal standards, SNV for scatter correction, vector norm for concentration independence).

---

### Label Encoding
**Layman:** Converting category names into numbers the computer can understand—like assigning "olive oil" = 1, "sunflower oil" = 2.

**Food Scientist:** Mapping categorical labels (variety, batch, treatment) to integers for machine learning models.

**Physicist:** Bijection $\phi: \{\text{class}_1, \ldots, \text{class}_C\} \to \{0, 1, \ldots, C-1\}$ or one-hot encoding $y_i \to [0, 0, 1, 0, \ldots]$ for $C$ classes.

**Warning:** Use one-hot encoding for tree-based models; integer encoding for neural networks with embedding layers. Never use arbitrary integer ordering (e.g., 1=low, 2=medium, 3=high) unless ordinal relationship exists.

---

### Sample vs. Replicate
**Layman:** A **sample** is one bottle of oil; a **replicate** is measuring that same bottle multiple times. Samples vary naturally (biology); replicates vary from measurement error.

**Food Scientist:** **Sample** = independent biological/physical unit (different batches, sources). **Replicate** = repeated measurement of same sample (technical variation). Important for proper CV strategy.

**Physicist:** Variance decomposition: $\sigma_{\text{total}}^2 = \sigma_{\text{biological}}^2 + \sigma_{\text{technical}}^2$. Replicates estimate $\sigma_{\text{technical}}$; samples estimate $\sigma_{\text{biological}}$.

**Best practice:** Keep replicates together in same fold during CV to avoid **leakage** (see below). Report both technical and biological variability separately.

---

### Matrix Effect
**Layman:** When the "packaging" affects the measurement—like how coffee tastes different in a plastic cup vs. ceramic mug, even though it's the same coffee.

**Food Scientist:** The food substrate (e.g., potato chips, meat) altering the spectral signature of the analyte of interest (e.g., frying oil). Complicates quantification.

**Physicist:** Interaction between analyte and matrix modifying absorption coefficients, scattering properties, or chemical environment. Described by Beer-Lambert deviations: $A \neq \epsilon c l$.

**Mitigation strategies:** Matrix-matched calibration, standard addition methods, or chemometric models trained on diverse matrices.

---

### Leakage (Data Leakage)
**Layman:** Accidentally giving the computer the answers during training—like studying for a test using the actual test questions. Makes results look better than they really are.

**Food Scientist:** Occurs when test data information "leaks" into training. Common sources: (1) replicates split across folds, (2) preprocessing on full dataset before CV, (3) feature selection using test labels.

**Physicist:** Correlation between training set $\mathcal{D}_{\text{train}}$ and test set $\mathcal{D}_{\text{test}}$ where $I(\mathcal{D}_{\text{train}}; \mathcal{D}_{\text{test}}) > 0$ (mutual information).

**How to prevent:**
- Keep replicates together in same fold
- Perform preprocessing within each CV fold separately
- Use nested CV for hyperparameter tuning
- Never look at test labels before final evaluation

**Warning:** Leakage inflates performance metrics (accuracy, R²) by 10-50%, leading to models that fail in production.

---

### CV Strategy (Cross-Validation Strategy)
**Layman:** The rules for how you split your data into training and testing groups—like deciding how to divide teams for a practice game.

**Food Scientist:** The splitting scheme for estimating model generalization. Must match deployment scenario (random = similar samples, batch-aware = new instruments, temporal = future predictions).

**Physicist:** Partitioning function $\pi: \mathcal{D} \to \{\mathcal{D}_{\text{train}}^{(k)}, \mathcal{D}_{\text{test}}^{(k)}\}_{k=1}^K$ with constraints based on metadata (batch, time, replicate groups).

**Common strategies:**
- **Random K-fold**: Shuffle and split randomly (valid only if samples truly independent)
- **Stratified K-fold**: Preserves class proportions in each fold
- **Group K-fold**: Keeps sample groups (replicates, batches) together (prevents leakage)
- **Time-series split**: Training on past, testing on future (for temporal data)
- **Leave-one-batch-out**: Train on N-1 batches, test on held-out batch (harshest test)

**Best practice for FoodSpec:** Use Group K-fold with sample_id groups to keep replicates together. For multi-instrument studies, use Leave-one-batch-out.

---

### Cross-Validation (CV)
**Layman:** Testing the computer's learning by hiding some data, training on the rest, then checking if it predicts the hidden data correctly. Prevents cheating.

**Food Scientist:** Splitting data into training/test folds multiple times to estimate model generalization. Batch-aware CV keeps batches intact to avoid leakage.

**Physicist:** Partitioning $\mathcal{D}$ into $K$ folds $\{\mathcal{D}_k\}$, computing $\text{Acc} = \frac{1}{K} \sum_{k=1}^K \text{Acc}(\hat{f}_{-k}, \mathcal{D}_k)$ where $\hat{f}_{-k}$ is trained without fold $k$.

---

### Batch / Instrument Drift
**Layman:** Different machines or measurement days produce slightly different readings for the same sample—like two bathroom scales giving different weights.

**Food Scientist:** Systematic variation across instruments, operators, or time periods that must be accounted for in validation to ensure models generalize.

**Physicist:** Additive/multiplicative offsets $\mathbf{y}_{\text{batch}} = a \cdot \mathbf{y} + \mathbf{b}$ requiring harmonization (calibration transfer, multiplicative scatter correction).

---

### Balanced Accuracy
**Layman:** Accuracy that treats all categories fairly, even if you have way more samples of one type. Example: If you test 100 olive oils and only 10 palm oils, this metric doesn't overemphasize olive oil.

**Food Scientist:** Average of per-class recall scores: $\text{BA} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$. Corrects class imbalance.

**Physicist:** Macro-averaged sensitivity across $C$ classes, invariant to class priors unlike raw accuracy.

---

### FDR (False Discovery Rate)
**Layman:** When testing many things at once (e.g., 1000 peaks), some will look important by chance. FDR controls how many false alarms you accept.

**Food Scientist:** Multiple-testing correction ensuring expected proportion of false positives $\leq \alpha$ (e.g., 5%). Use Benjamini-Hochberg procedure.

**Physicist:** $\text{FDR} = \mathbb{E}[\frac{V}{R}]$ where $V$ = false positives, $R$ = total rejections. BH controls FDR at level $\alpha$.

---

### Protocol / Workflow
**Layman:** A recipe for analyzing food samples—step-by-step instructions the computer follows automatically.

**Food Scientist:** YAML file specifying preprocessing, harmonization, QC checks, RQ analysis, and output generation with validation strategy.

**Physicist:** Declarative pipeline $\mathcal{P} = \{s_1, s_2, \ldots, s_n\}$ where each step $s_i$ applies transformation $T_i: \mathcal{X}_{i-1} \to \mathcal{X}_i$ with versioned parameters.

---

## Technical Terms (Layer 2-3)

### RQ (Ratio-Quality) Engine
**Definition:** FoodSpec module computing stability (CV/MAD), discriminative power (ANOVA F, effect sizes), trends (regression slopes), divergence (two-group comparisons), minimal panels (greedy feature selection), and clustering metrics on peak areas and ratios.

**When to use:** Quality control workflows requiring identification of reproducible, discriminative markers.

**Assumptions:** Approximately normal distributions for parametric tests; sufficient sample size ($n \geq 20$ per group).

---

### ALS (Asymmetric Least Squares) Baseline
**Definition:** Baseline correction minimizing $||\mathbf{y} - \mathbf{b}||^2 + \lambda \sum |\Delta^2 b_i| + w_i(\mathbf{y}_i - \mathbf{b}_i)$ where $w_i = 0$ for $\mathbf{y}_i < \mathbf{b}_i$, $w_i = 1$ otherwise.

**When to use:** Removing background fluorescence/drift in Raman/FTIR spectra.

**Failure modes:** Over-smoothing ($\lambda$ too large) removes real peaks; under-smoothing ($\lambda$ too small) retains baseline.

---

### HSI (Hyperspectral Imaging)
**Definition:** 3D datacube $(x, y, \tilde{\nu})$ capturing spatially resolved spectra at each pixel. Enables mapping chemical composition across surfaces.

**When to use:** Surface contamination detection, heterogeneity analysis, ROI extraction.

**Assumptions:** Stable illumination across field of view; negligible spatial-spectral coupling.

---

### MOATS (Model Optimized by Accumulated Threshold Selection)
**Definition:** Feature selection algorithm maximizing classification accuracy while minimizing feature count. Iteratively adds features based on cumulative importance.

**When to use:** Building minimal marker panels for cost-effective QA/QC (fewer measurements = faster/cheaper).

**Failure modes:** Greedy selection may miss optimal combinations; requires validation on independent test set.

---

### Harmonization
**Definition:** Aligning spectra from different instruments/batches via wavenumber calibration, power normalization, or calibration transfer (e.g., Piecewise Direct Standardization).

**When to use:** Multi-instrument studies, longitudinal monitoring, pooling data across labs.

**Assumptions:** Linear/affine relationship between instruments; consistent sample preparation.

---

## Mathematical Notation

### Spectroscopy Symbols
- $I(\tilde{\nu})$: Intensity at wavenumber $\tilde{\nu}$ (cm⁻¹)
- $\lambda$: Wavelength (nm); $\tilde{\nu} = 10^7 / \lambda$
- $\frac{\partial\alpha}{\partial Q}$: Polarizability derivative (determines Raman activity)
- $\sigma$: Scattering cross-section

### Linear Algebra
- $\mathbf{X}$: Data matrix (rows = samples, columns = features)
- $\mathbf{y}$: Response vector (observed spectrum or labels)
- $\Sigma$: Covariance matrix ($\Sigma = \frac{1}{n-1} \mathbf{X}^\top \mathbf{X}$)
- $V, \Lambda$: Eigenvector and eigenvalue matrices in PCA ($\Sigma = V \Lambda V^\top$)
- $T, P$: PCA scores and loadings ($\mathbf{X} = TP^\top + E$)

### Statistics
- $\alpha$: Significance level (typically 0.05)
- $d$: Cohen's d effect size ($d = \frac{\mu_1 - \mu_2}{\sigma_{\text{pooled}}}$)
- $s^2$: Variance
- $f$: F-statistic (ANOVA, ratio of between-group to within-group variance)
- $p_{\text{perm}}$: Permutation p-value (fraction of permuted test statistics ≥ observed)

---

## Abbreviations (Alphabetical)

- **ALS**: Asymmetric Least Squares baseline correction
- **ANOVA**: Analysis of Variance (tests group mean differences)
- **AUC**: Area Under the Curve (ROC or PR curve)
- **CI**: Confidence Interval
- **CV**: Coefficient of Variation ($\text{CV} = \sigma/\mu \times 100\%$) OR Cross-Validation
- **DL**: Deep Learning
- **FDR**: False Discovery Rate (multiple testing correction)
- **F1**: Harmonic mean of precision and recall: $F1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$
- **FTIR**: Fourier Transform Infrared Spectroscopy
- **HSI**: Hyperspectral Imaging
- **IoU**: Intersection over Union (segmentation accuracy metric)
- **LOA**: Limits of Agreement (Bland–Altman analysis)
- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **MAD**: Median Absolute Deviation (robust dispersion measure)
- **MCC**: Matthews Correlation Coefficient
- **MCR-ALS**: Multivariate Curve Resolution – Alternating Least Squares
- **MOATS**: Model Optimized by Accumulated Threshold Selection
- **NIR**: Near-Infrared Spectroscopy
- **NNLS**: Non-Negative Least Squares (constrained regression for mixture fractions)
- **OC-SVM**: One-Class Support Vector Machine (novelty detection)
- **PCA**: Principal Component Analysis
- **PLS / PLS-DA**: Partial Least Squares (Regression / Discriminant Analysis)
- **PR curve**: Precision–Recall curve
- **QC**: Quality Control
- **RMSE**: Root Mean Square Error
- **ROC**: Receiver Operating Characteristic curve
- **RQ**: Ratio-Quality (FoodSpec analysis engine)
- **SNR**: Signal-to-Noise Ratio
- **t-SNE**: t-distributed Stochastic Neighbor Embedding (visualization only, not for inference)

---

## Domain-Specific Terms

### Edible Oils
- **OO**: Olive Oil
- **PO**: Palm Oil
- **VO**: Vegetable Oil (often sunflower or canola)
- **CO**: Coconut Oil
- **Carbonyl band**: ~1740 cm⁻¹ (C=O stretch in oxidized oils)
- **Unsaturation band**: ~1650 cm⁻¹ (C=C stretch in double bonds)
- **Reference peak**: ~2720 cm⁻¹ (used for normalization)

### Food Science
- **Adulteration**: Mixing with cheaper/undeclared ingredients
- **Thermal degradation**: Chemical changes during heating/frying (oxidation, polymerization)
- **Matrix effect**: Food substrate (e.g., chips) altering oil spectral signature
- **Shelf life**: Time until quality degrades below acceptable threshold

---

## When Terms Are Used Incorrectly

**Common Mistake:** Using "accuracy" for imbalanced datasets.  
**Fix:** Use balanced accuracy or F1 score.

**Common Mistake:** Calling t-SNE a "model."  
**Fix:** t-SNE is visualization only; use PCA for dimensionality reduction in modeling.

**Common Mistake:** "Cross-validation" without specifying batch-aware.  
**Fix:** Always state validation strategy (random, stratified, batch-aware, nested).

---

## What's Next?

- **See notation in context:** [RQ Engine Theory](../theory/rq_engine_detailed.md)
- **Understand validation terms:** [Validation Strategies](../05-advanced-topics/validation_strategies.md)
- **Learn preprocessing methods:** [Preprocessing Recipes](../methods/preprocessing/normalization_smoothing.md)

---

**Can't find a term?** Open an issue: https://github.com/chandrasekarnarayana/foodspec/issues
