# Reporting Standards for Validation

!!! abstract "Reproducibility & Transparency"
    Scientific validation is only valuable if it can be **reproduced** by others. This page provides comprehensive reporting standards for:
    
    1. **Minimum reporting checklist** for papers and internal reports
    2. **Methods text templates** for Materials & Methods sections
    3. **Supplementary information guidelines** (code, data, hyperparameters)
    4. **FAIR principles** (Findable, Accessible, Interoperable, Reusable)
    
    Following these standards ensures your work meets modern reproducibility requirements for publication in high-impact journals.

---

## Why Reporting Standards Matter

### The Reproducibility Crisis

Studies in chemometrics and machine learning face a reproducibility crisis:

- **2016 Nature Survey:** 70% of researchers failed to reproduce another scientist's experiments
- **2019 Chemometrics Review:** 60% of published models lacked sufficient detail for reproduction
- **2022 ML Audit:** 42% of papers showed signs of data leakage or inadequate validation

**Root Causes:**
1. **Incomplete methods:** Missing preprocessing parameters, CV strategy, hyperparameters
2. **Cherry-picking results:** Reporting best-case performance without uncertainty estimates
3. **Inaccessible code/data:** "Code available upon request" (rarely fulfilled)
4. **Ambiguous descriptions:** "Standard preprocessing was applied" (what does this mean?)

!!! danger "Consequences of Poor Reporting"
    - **Wasted resources:** Months spent trying to reproduce published results
    - **Publication retractions:** Non-reproducible findings discovered post-publication
    - **Regulatory rejection:** FDA/EFSA refuse to approve models without reproducible validation
    - **Loss of scientific credibility:** Entire fields viewed as unreliable

---

## Minimum Reporting Checklist

Use this checklist for **every** validation study (publication, thesis, internal report):

### ☑ Dataset Description

- [ ] **Sample size:** Number of samples (biological replicates, not technical replicates)
- [ ] **Class distribution:** Samples per class (e.g., 30 EVOO, 25 Lampante, 15 Pomace)
- [ ] **Replicates:** Number of technical replicates per sample (e.g., 3-5 spectra/sample)
- [ ] **Batches/days:** Measurement structure (e.g., 5 batches collected over 2 weeks)
- [ ] **Instruments:** Model, manufacturer, configuration (e.g., Bruker Alpha FTIR-ATR, diamond crystal)
- [ ] **Collection protocol:** Sample preparation, measurement parameters (resolution, scans, etc.)
- [ ] **Wavenumber range:** Spectral range used for modeling (e.g., 4000-650 cm⁻¹)
- [ ] **Data availability:** DOI, repository link, or explicit "available upon request" statement

**Example:**
> *Dataset consisted of 40 olive oil samples (20 extra virgin, 20 lampante grade) with 5 technical replicates each (200 spectra total). Samples were collected in 8 batches over 3 weeks (5 samples/batch). Spectra were measured using a Bruker Alpha FTIR-ATR spectrometer (diamond crystal, 4 cm⁻¹ resolution, 24 scans/spectrum, 4000-650 cm⁻¹ range). Data are available at [DOI:10.5281/zenodo.XXXXXXX].*

---

### ☑ Preprocessing Steps

- [ ] **Order of operations:** Exact sequence (e.g., baseline → smooth → normalize)
- [ ] **Baseline correction:** Method (ALS, polynomial, etc.) with parameters (λ, p, order)
- [ ] **Smoothing:** Method (Savitzky-Golay, Gaussian) with parameters (window, polynomial order)
- [ ] **Normalization:** Method (SNV, MSC, vector norm, reference peak) with parameters
- [ ] **Feature selection:** Method (if any) and criteria (e.g., VIP scores >1.0, top 50 variables)
- [ ] **Preprocessing leakage prevention:** Explicit statement that preprocessing was fit within CV folds

**Example:**
> *Spectra were preprocessed as follows: (1) Asymmetric Least Squares (ALS) baseline correction (λ=10⁶, p=0.01), (2) Savitzky-Golay smoothing (window=9, polyorder=2), (3) Standard Normal Variate (SNV) normalization. All preprocessing parameters were fit exclusively on training data within each CV fold to prevent data leakage.*

---

### ☑ Validation Strategy

- [ ] **CV type:** Random K-fold, Stratified, Grouped (by sample/batch/day), Leave-One-Group-Out, Time-Series
- [ ] **Grouping variable:** What samples were grouped (e.g., "Grouped by sample ID to prevent replicate leakage")
- [ ] **Number of folds:** K in K-fold (e.g., 5-fold, 10-fold)
- [ ] **Number of repeats:** How many times CV was repeated with different random seeds (e.g., 10 repeats, 20 repeats)
- [ ] **Total number of evaluations:** K × repeats (e.g., 5-fold × 10 repeats = 50 evaluations)
- [ ] **Train/test split ratio:** If using holdout validation (e.g., 80/20 split)
- [ ] **Stratification:** How classes were balanced across folds (if applicable)

**Example:**
> *Model performance was assessed using 10-fold grouped cross-validation (grouped by sample ID to prevent replicate leakage), repeated 20 times with different random seeds (200 total evaluations). Each fold maintained stratified class proportions (50% EVOO, 50% lampante).*

---

### ☑ Model & Hyperparameters

- [ ] **Model type:** Algorithm name (e.g., Random Forest, PLS-DA, SVM)
- [ ] **Hyperparameters:** All tunable parameters with final values (e.g., `n_estimators=100`, `max_depth=10`)
- [ ] **Hyperparameter tuning:** How parameters were selected (grid search, random search, Bayesian optimization)
- [ ] **Tuning CV:** Inner CV strategy for hyperparameter tuning (e.g., nested 5-fold CV)
- [ ] **Software versions:** Package versions (e.g., scikit-learn 1.3.0, FoodSpec 1.0.0, Python 3.10)
- [ ] **Random seed:** Fixed seed for reproducibility (e.g., `random_state=42`)

**Example:**
> *A Random Forest classifier (scikit-learn 1.3.0, Python 3.10) was trained with hyperparameters: n_estimators=100, max_depth=10, min_samples_split=5, random_state=42. Hyperparameters were selected via grid search with nested 5-fold cross-validation on the training set only.*

---

### ☑ Performance Metrics

- [ ] **Primary metric:** Main evaluation metric with uncertainty (e.g., accuracy: 87.3% ± 3.2% [95% CI])
- [ ] **Secondary metrics:** Additional metrics (F1, MCC, precision, recall, RMSE, R², etc.)
- [ ] **Per-class performance:** Class-specific precision/recall (for multiclass problems)
- [ ] **Confusion matrix:** Full confusion matrix (as table or heatmap)
- [ ] **Uncertainty quantification:** Confidence intervals (95% CI via repeated CV or bootstrapping)
- [ ] **Statistical significance:** p-values when comparing models (paired t-test, McNemar, etc.)

**Example:**
> *Classification accuracy was 87.3% ± 3.2% (mean ± 95% CI from 20 repeated 10-fold CV runs). Secondary metrics: F1-score = 0.865 ± 0.035, MCC = 0.821 ± 0.041. Per-class performance: EVOO (precision=0.91, recall=0.89), Lampante (precision=0.70, recall=0.75). See Supplementary Table S1 for confusion matrix.*

---

### ☑ Robustness Testing

- [ ] **Preprocessing sensitivity:** Performance stability across parameter ranges
- [ ] **Outlier robustness:** Performance on corrupted spectra (if tested)
- [ ] **Batch/day effects:** Leave-one-batch-out or temporal validation results
- [ ] **Adversarial testing:** Out-of-distribution performance (adulteration, degradation, matrix effects)

**Example:**
> *Preprocessing sensitivity analysis (Supplementary Fig. S2) showed accuracy varied by <4% across ALS λ ∈ [10⁵, 10⁷] and smoothing window ∈ [5, 13]. Leave-one-batch-out CV yielded 82.1% ± 5.2% accuracy (8 batches), indicating good batch robustness.*

---

### ☑ Code & Data Availability

- [ ] **Code repository:** GitHub/GitLab link with DOI (via Zenodo archival)
- [ ] **Data repository:** Zenodo, Figshare, Dryad, or domain-specific repository
- [ ] **Environment specification:** `requirements.txt`, `environment.yml`, or Docker container
- [ ] **Analysis scripts:** Exact scripts used to generate results (not just library code)
- [ ] **Reproducibility instructions:** README with step-by-step reproduction protocol

**Example:**
> *All code is available at GitHub ([https://github.com/username/repo](https://github.com/username/repo), archived at Zenodo DOI:10.5281/zenodo.XXXXXXX). Raw spectral data and preprocessed features are available at Zenodo ([DOI:10.5281/zenodo.YYYYYYY](https://doi.org/10.5281/zenodo.YYYYYYY)). Analysis was conducted using FoodSpec 1.0.0 (Python 3.10) in a conda environment (environment.yml provided).*

---

## Methods Text Templates

### Template 1: Classification Study (FTIR Olive Oil Authentication)

> **Materials and Methods**
>
> **Dataset.** Forty olive oil samples (20 extra virgin olive oil [EVOO], 20 lampante grade) were analyzed. Each sample was measured in quintuplicate (5 technical replicates) using a Bruker Alpha FTIR-ATR spectrometer (diamond crystal, 4 cm⁻¹ resolution, 24 scans per spectrum) over the 4000-650 cm⁻¹ range. Samples were collected in 8 batches over a 3-week period (5 samples per batch). Raw spectral data (200 spectra total) are available at Zenodo ([DOI:10.5281/zenodo.XXXXXXX]).
>
> **Preprocessing.** Spectra underwent the following preprocessing pipeline: (1) Asymmetric Least Squares (ALS) baseline correction¹ (λ=10⁶, p=0.01, 10 iterations), (2) Savitzky-Golay smoothing² (window length=9 points, polynomial order=2), and (3) Standard Normal Variate (SNV) normalization³. All preprocessing steps were applied within cross-validation folds (fit on training data only) to prevent data leakage.
>
> **Model Training.** A Random Forest classifier⁴ (scikit-learn 1.3.0, Python 3.10) was trained with the following hyperparameters: n_estimators=100, max_depth=10, min_samples_split=5, random_state=42. Hyperparameters were selected via grid search using nested 5-fold cross-validation on the training set.
>
> **Validation Strategy.** Model performance was assessed using 10-fold grouped cross-validation, where all technical replicates of a sample were assigned to the same fold to prevent replicate leakage⁵. Cross-validation was repeated 20 times with different random seeds, yielding 200 total performance evaluations. Folds were stratified to maintain equal class proportions (50% EVOO, 50% lampante).
>
> **Performance Metrics.** Classification accuracy (primary metric), F1-score, Matthews Correlation Coefficient (MCC), and per-class precision/recall were computed. Confidence intervals (95% CI) were calculated from the 200 repeated CV runs. Model comparison used paired t-tests (α=0.05).
>
> **Robustness Testing.** Preprocessing sensitivity was assessed by varying ALS λ ∈ [10⁵, 10⁷] and smoothing window ∈ [5, 13] (Supplementary Fig. S2). Batch robustness was evaluated via leave-one-batch-out cross-validation (8 folds, one per batch).
>
> **References:**  
> 1. Eilers & Boelens (2005). *Comput. Stat.*, 1:1.  
> 2. Savitzky & Golay (1964). *Anal. Chem.*, 36:1627.  
> 3. Barnes et al. (1989). *Appl. Spectrosc.*, 43:772.  
> 4. Breiman (2001). *Mach. Learn.*, 45:5.  
> 5. Brereton & Lloyd (2010). *J. Chemometrics*, 24:1.

---

### Template 2: Regression Study (Adulteration Quantification)

> **Materials and Methods**
>
> **Dataset.** Fifty olive oil samples with known hazelnut oil adulteration levels (0%, 1%, 2%, 5%, 10%, 20%) were analyzed (n=10 samples per level, 3 technical replicates each, 150 spectra total). Spectra were collected using a Perkin-Elmer Spectrum Two FTIR-ATR (diamond crystal, 4 cm⁻¹ resolution, 32 scans, 4000-650 cm⁻¹). Data are available at [DOI:10.5281/zenodo.XXXXXXX].
>
> **Preprocessing.** Spectra underwent (1) rubberband baseline correction (64 baseline points), (2) Savitzky-Golay first derivative (window=11, polyorder=2), and (3) mean centering. Preprocessing was applied within CV folds to prevent leakage.
>
> **Model Training.** A Partial Least Squares Regression (PLS-R) model¹ (scikit-learn 1.3.0) was trained with 8 latent variables (selected via cross-validation on training data). Training used the NIPALS algorithm² with random_state=42.
>
> **Validation Strategy.** Performance was assessed using 5-fold grouped cross-validation (grouped by sample ID, repeated 10 times, 50 total evaluations). An independent holdout test set (20% of samples, never used in training) provided final validation.
>
> **Performance Metrics.** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² were computed. Prediction intervals (90% coverage) were estimated via conformal prediction³ using calibration residuals from a validation set.
>
> **Robustness Testing.** Model performance was tested on simulated degraded samples (thermal oxidation: carbonyl peak enhancement at 1740 cm⁻¹) and matrix-contaminated samples (oil extracted from fried potato chips).
>
> **References:**  
> 1. Wold et al. (2001). *Chemom. Intell. Lab. Syst.*, 58:109.  
> 2. Geladi & Kowalski (1986). *Anal. Chim. Acta*, 185:1.  
> 3. Shafer & Vovk (2008). *JMLR*, 9:371.

---

## Supplementary Information Guidelines

### What to Include

Supplementary materials should enable **exact reproduction** of your analysis:

#### 1. Detailed Methods

- **Extended preprocessing description:** Formulas, algorithms, rationale for parameter choices
- **Hyperparameter tuning logs:** Full grid search results (not just best parameters)
- **Feature importance analysis:** VIP scores, SHAP values, or permutation importance (if applicable)

#### 2. Complete Results

- **Full confusion matrices:** For all CV folds (not just average)
- **Per-fold performance:** Table showing accuracy/RMSE for each CV fold
- **Learning curves:** Training set size vs. performance (assess data sufficiency)
- **Calibration curves:** Predicted vs. true values (regression) or reliability diagrams (classification)

#### 3. Robustness Analysis

- **Preprocessing sensitivity heatmaps:** Parameter grid search results
- **Leave-one-batch-out table:** Performance for each held-out batch
- **Outlier robustness curves:** Accuracy vs. outlier fraction (0%, 5%, 10%, 20%)
- **Batch effect PCA plots:** Visualize batch separation

#### 4. Code & Environment

- **Analysis scripts:** Jupyter notebooks or Python scripts with inline comments
- **Environment specification:**
  ```yaml
  # environment.yml
  name: foodspec-paper
  channels:
    - conda-forge
    - defaults
  dependencies:
    - python=3.10
    - scikit-learn=1.3.0
    - numpy=1.24.0
    - pandas=2.0.0
    - matplotlib=3.7.0
    - foodspec=1.0.0
  ```
- **Reproduction instructions:** Step-by-step README with expected output

#### 5. Raw Data

- **Spectral data:** HDF5 or CSV format with metadata (sample ID, batch, replicate, label)
- **Data dictionary:** Explain column names, units, allowed ranges
- **Quality control metrics:** SNR, baseline quality, outlier flags

---

## FAIR Principles Compliance

Ensure your work adheres to **FAIR data principles**:

### Findable

- [ ] Assign a **persistent identifier (DOI)** to dataset and code (via Zenodo, Figshare)
- [ ] Use descriptive **metadata** (keywords, abstract, authors, date)
- [ ] Register in domain-specific repositories (e.g., ChemSpider, PubChem, metabolomics repositories)

### Accessible

- [ ] Provide **open access** to data and code (CC BY 4.0 or MIT license)
- [ ] If data are restricted (proprietary, GDPR), provide **synthetic data** with same structure
- [ ] Ensure long-term preservation (institutional repositories, not personal websites)

### Interoperable

- [ ] Use **standard formats:** CSV for tabular data, HDF5 for arrays, JSON for metadata
- [ ] Include **ontology terms:** "FTIR-ATR spectroscopy" (not just "spectroscopy"), "Olea europaea" (not just "olive")
- [ ] Provide **machine-readable metadata:** JSON-LD, RDF, or DataCite schema

### Reusable

- [ ] Include **clear license** (CC BY 4.0 for data, MIT/Apache 2.0 for code)
- [ ] Document **provenance:** How data were collected, processed, quality-controlled
- [ ] Provide **usage examples:** Minimal working example in README
- [ ] Ensure **long-term compatibility:** Avoid deprecated libraries, document versions

---

## Journal-Specific Requirements

Many journals now mandate specific reporting standards:

| Journal | Reporting Standard | Key Requirements |
|---------|-------------------|------------------|
| **Nature/Science** | [ARRIVE, CONSORT] | Code/data availability statement, statistical power analysis |
| **Analytical Chemistry** | ACS Guidelines | Preprocessing details, CV strategy, uncertainty quantification |
| **Chemometrics & Intelligent Lab Systems** | Chemometrics Checklist | Replicate leakage prevention, per-fold results |
| **TrAC Trends in Analytical Chemistry** | Mishra et al. (2021) | Minimum reporting checklist (25 items) |
| **Food Chemistry** | Elsevier Data Policy | Raw data in repository, reproducible analysis scripts |

!!! tip "Check Early"
    Consult target journal's author guidelines **before** conducting experiments. Retrofitting reproducibility is harder than planning ahead.

---

## FoodSpec Protocol Logging

FoodSpec automatically logs validation details when using Protocols:

```python
from foodspec.protocols import Protocol

# Define protocol
protocol = Protocol.from_yaml('olive_oil_authentication.yaml')

# Run with automatic logging
results = protocol.run(data_path='olive_oils.csv', log_dir='validation_logs/')

# Generates:
# - validation_logs/20231215_143022_run.log (human-readable)
# - validation_logs/20231215_143022_metadata.json (machine-readable)
# - validation_logs/20231215_143022_performance.csv (metrics per fold)
# - validation_logs/20231215_143022_confusion_matrix.png
```

**Logged Information:**
- Preprocessing steps with parameters
- CV strategy (folds, repeats, grouping variable)
- Model hyperparameters
- Performance metrics with confidence intervals
- Software versions (FoodSpec, scikit-learn, Python)
- Random seeds for reproducibility
- Execution time and compute environment

**Export for Publication:**
```python
from foodspec.protocols.reporting import generate_methods_text

methods_text = generate_methods_text(
    log_file='validation_logs/20231215_143022_run.log',
    template='classification',  # or 'regression'
    journal='nature'  # Adapts to journal style
)

print(methods_text)  # Copy to manuscript
```

---

## Reproducibility Checklist (Before Submission)

Use this final checklist before submitting a paper:

### Data

- [ ] Raw spectral data uploaded to repository with DOI
- [ ] Metadata file includes sample IDs, batches, replicates, labels
- [ ] Data dictionary explains all columns and units
- [ ] Quality control flags documented

### Code

- [ ] All analysis scripts on GitHub/GitLab with DOI (Zenodo archive)
- [ ] Scripts run successfully in clean environment (tested on separate machine)
- [ ] README provides step-by-step reproduction instructions
- [ ] Environment specification file (`requirements.txt` or `environment.yml`)

### Methods

- [ ] Preprocessing steps described with exact parameters
- [ ] CV strategy explicitly stated (grouping variable, folds, repeats)
- [ ] Hyperparameter tuning procedure documented
- [ ] Leakage prevention explicitly mentioned

### Results

- [ ] Primary metric reported with 95% confidence interval
- [ ] Confusion matrix provided (classification) or calibration plot (regression)
- [ ] Statistical significance tests (if comparing models)
- [ ] Robustness tests documented (preprocessing sensitivity, batch effects)

### Supplementary Materials

- [ ] Extended methods with preprocessing rationale
- [ ] Full hyperparameter grid search results
- [ ] Per-fold performance table
- [ ] Preprocessing sensitivity heatmaps
- [ ] Leave-one-batch-out results

### Accessibility

- [ ] Code and data have persistent identifiers (DOIs)
- [ ] Open licenses specified (CC BY 4.0 for data, MIT for code)
- [ ] No broken links in manuscript or README
- [ ] Contact information provided for questions

---

## Example: Comprehensive Reporting (Olive Oil Study)

### Main Text

> **Abstract:** We developed a Random Forest classifier for FTIR-based authentication of extra virgin olive oil (EVOO) vs. lampante grade. Using 40 samples (200 spectra) with rigorous grouped cross-validation, the model achieved 87.3% ± 3.2% accuracy (95% CI). Leave-one-batch-out validation (82.1% ± 5.2%) confirmed batch robustness. All data and code are openly available (DOI:10.5281/zenodo.XXXXXXX).
>
> **Methods:** [See Template 1 above]
>
> **Results:** Classification accuracy was 87.3% ± 3.2% (mean ± 95% CI from 20 repeated 10-fold grouped CV runs, 200 total evaluations). Secondary metrics: F1-score = 0.865 ± 0.035, MCC = 0.821 ± 0.041. Per-class performance: EVOO (precision=0.91, recall=0.89), lampante (precision=0.70, recall=0.75). See Figure 2 for confusion matrix.
>
> Leave-one-batch-out cross-validation yielded 82.1% ± 5.2% accuracy across 8 batches, indicating good day-to-day robustness (see Supplementary Table S3). Preprocessing sensitivity analysis (Supplementary Fig. S2) showed accuracy varied by <4% across ALS λ ∈ [10⁵, 10⁷] and smoothing window ∈ [5, 13], confirming preprocessing robustness.

### Supplementary Information

**Supplementary Table S1:** Confusion matrix (aggregated from 200 CV runs)

|                | Predicted EVOO | Predicted Lampante |
|----------------|----------------|---------------------|
| **True EVOO**  | 1782 (89%)     | 218 (11%)           |
| **True Lampante** | 318 (16%)   | 1682 (84%)          |

**Supplementary Table S2:** Per-fold performance (10-fold grouped CV, 1 repeat shown)

| Fold | Test Samples | Accuracy | F1-Score | MCC   |
|------|--------------|----------|----------|-------|
| 1    | OO_01-04     | 0.850    | 0.842    | 0.789 |
| 2    | OO_05-08     | 0.900    | 0.895    | 0.845 |
| ...  | ...          | ...      | ...      | ...   |
| 10   | OO_37-40     | 0.875    | 0.868    | 0.820 |

**Supplementary Table S3:** Leave-one-batch-out performance

| Test Batch | Date       | Test Samples | Accuracy |
|------------|------------|--------------|----------|
| Batch 1    | 2023-11-01 | 5 (25 spectra) | 0.867  |
| Batch 2    | 2023-11-03 | 5 (25 spectra) | 0.840  |
| ...        | ...        | ...          | ...      |
| Batch 8    | 2023-11-20 | 5 (25 spectra) | 0.853  |

**Mean ± Std:** 0.852 ± 0.023 (range: [0.820, 0.880])

**Supplementary Figure S2:** Preprocessing sensitivity heatmap (accuracy vs. ALS λ and smoothing window)

**Supplementary Code:** Available at [https://github.com/username/olive-oil-auth](https://github.com/username/olive-oil-auth) (archived at Zenodo DOI:10.5281/zenodo.XXXXXXX)

---

## Further Reading

- **Mishra et al. (2021).** "New data preprocessing trends based on ensemble of multiple preprocessing techniques." *TrAC Trends in Analytical Chemistry*, 132:116045. [DOI](https://doi.org/10.1016/j.trac.2021.116045)
- **Wilkinson et al. (2016).** "The FAIR Guiding Principles for scientific data management and stewardship." *Sci. Data*, 3:160018. [DOI](https://doi.org/10.1038/sdata.2016.18)
- **Stodden et al. (2018).** "An empirical analysis of journal policy effectiveness for computational reproducibility." *PNAS*, 115:2584-2589. [DOI](https://doi.org/10.1073/pnas.1708290115)
- **Raschka (2018).** "Model evaluation, model selection, and algorithm selection in machine learning." *arXiv:1811.12808*. [Link](https://arxiv.org/abs/1811.12808)

---

## Related Pages

- [Cross-Validation & Leakage](cross_validation_and_leakage.md) – Prevent data leakage
- [Metrics & Uncertainty](metrics_and_uncertainty.md) – Quantify confidence intervals
- [Robustness Checks](robustness_checks.md) – Test preprocessing sensitivity
- [Protocols → Methods Text Generator](../../protocols/methods_text_generator.md) – Automated methods text
- [Developer Guide → Documentation](../../developer-guide/documentation_guidelines.md) – Internal docs standards

---

**Congratulations!** You've completed the Validation & Scientific Rigor guide. Apply these standards to ensure your work is reproducible, transparent, and publication-ready.
