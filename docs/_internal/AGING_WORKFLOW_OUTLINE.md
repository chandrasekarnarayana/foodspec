# Aging Workflow Page: Canonical Structure Outline

**Document Purpose:** Design specification for rewriting `docs/workflows/aging_workflows.md` to match canonical workflow structure established in heating_quality_monitoring.md, oil_authentication.md, mixture_analysis.md, and batch_quality_control.md.

**Target Audience:** Documentation maintainers, workflow implementers.

---

## Section 1: Standard Header (📋)

### Subsections

1. **Purpose** (1-2 sentences)
2. **When to Use** (5-7 bulleted scenarios)
3. **Inputs** (structured list)
4. **Outputs** (structured list)
5. **Assumptions** (4-5 bulleted statements)

### Acceptance Criteria

**Purpose:**
- ✅ Defines workflow in 1-2 sentences maximum
- ✅ Distinguishes aging (storage/passive) from heating (thermal/active)
- ✅ Mentions "shelf-life prediction" and "degradation trajectories"
- ❌ Does NOT mention thermal stress, frying, or elevated temperature studies

**When to Use:**
- ✅ Lists 5-7 concrete use cases for aging workflow
- ✅ Includes: shelf-life estimation, storage stability monitoring, packaging evaluation, oxidation kinetics at storage temps
- ✅ Each bullet starts with action verb (Monitor, Estimate, Compare, Predict, etc.)
- ✅ Emphasizes days-months timescale (not hours)
- ❌ Does NOT overlap with heating workflow scenarios (no frying, accelerated thermal tests)

**Inputs:**
- ✅ Format: HDF5/CSV with time-series spectra
- ✅ Required metadata: `time_col` (days/months), `entity_col` (sample ID), value column (degradation metric)
- ✅ Optional metadata: `storage_temp`, `packaging`, `treatment`, `batch`
- ✅ Wavenumber range: 600-1800 cm⁻¹ (or specify if different)
- ✅ Min samples: States minimum time points (10-30 typical) and replicates (3+ per entity)
- ✅ Temporal spacing: Recommends spacing (e.g., weekly for 12 weeks, monthly for 12 months)

**Outputs:**
- ✅ Lists 5 primary outputs:
  - `trajectories.csv` — Trajectory fit parameters per entity (slope, intercept, R², acceleration)
  - `shelf_life_estimates.csv` — Time-to-threshold (t_star) with 95% confidence intervals
  - `stage_labels.csv` — Classification (early/mid/late) per entity-timepoint
  - `degradation_plots.png` — Time-series with fitted trajectories
  - `report.md` — Narrative summary with recommendations
- ✅ Each output has 1-line description
- ✅ Format clearly specified (CSV, PNG, JSON, MD)

**Assumptions:**
- ✅ Lists 4-5 key assumptions:
  - Time measurements accurate (date/age recorded consistently)
  - Storage conditions controlled (temp, humidity, light)
  - Degradation monotonic (not reversible)
  - Threshold defined based on sensory/chemical criteria
  - Independent samples (not repeated scans of same unit)
- ✅ Each assumption testable or verifiable from data/experimental design
- ❌ Does NOT include trivial assumptions (e.g., "spectra are numeric")

---

## Section 2: Minimal Reproducible Example (🔬)

### Subsections

1. **Option A: Bundled Synthetic Data** (complete working code)
2. **Option B: Custom Synthetic Generator** (code template for user data simulation)
3. **Expected Output** (YAML-formatted console output + figure caption)

### Acceptance Criteria

**Option A (Bundled Synthetic Data):**
- ✅ Imports only from `foodspec` modules (no external dependencies except numpy/pandas/matplotlib)
- ✅ Uses synthetic data generator: `from foodspec.demo import synthetic_aging_dataset` or equivalent
- ✅ Calls high-level workflow function: `from foodspec.workflows.aging import compute_degradation_trajectories`
- ✅ Calls shelf-life estimator: `from foodspec.workflows.shelf_life import estimate_remaining_shelf_life`
- ✅ Prints summary statistics: number of entities, time range, R² values, shelf-life estimates
- ✅ Generates 1-2 plots: trajectory fits + shelf-life estimates with CI
- ✅ Saves outputs to `outputs/` directory with clear filenames
- ✅ Includes print statements showing key results (slope, R², t_star, confidence intervals)
- ✅ Runs in < 5 seconds on typical hardware
- ✅ Length: 30-50 lines (not counting comments)

**Option B (Custom Synthetic Generator):**
- ✅ Provides `generate_synthetic_aging()` function signature
- ✅ Parameters: `n_entities=5`, `n_timepoints=20`, `time_range=(0, 180)` (days), `random_state=42`
- ✅ Creates realistic degradation patterns: linear + noise, or quadratic acceleration
- ✅ Includes metadata columns: `entity_id`, `time_days`, `degradation_value`
- ✅ Returns `SpectralDataset` object with synthetic spectra + metadata
- ✅ Shows example usage: `fs = generate_synthetic_aging()` → `compute_degradation_trajectories(fs)`
- ✅ Length: 20-40 lines

**Expected Output:**
- ✅ YAML code block showing console output
- ✅ Includes: number of entities, time range, example trajectory fit (R², slope, p-value)
- ✅ Shows shelf-life estimate: t_star with 95% CI
- ✅ References saved figure: `![Aging trajectories](../../assets/workflows/aging/degradation_trajectories.png)`
- ✅ Output realistic (R² > 0.80, p < 0.01, narrow CI < 30% of t_star)

---

## Section 3: Step-by-Step Pipeline (Numbered)

### Subsections

1. **Problem and Dataset** (Why labs care; input description)
2. **Pipeline (Default)** (Preprocessing → Feature Extraction → Modeling → Outputs)
3. **Python Example (Synthetic)** (Code snippet with comments)
4. **CLI Example (with Config)** (YAML config + command)
5. **Interpretation** (Qualitative + Quantitative; Reviewer phrasing template)
6. **Summary** (3-5 bullet points)
7. **Statistical Analysis** (Code example + interpretation template)

### Acceptance Criteria

**1. Problem and Dataset:**
- ✅ States why labs use aging workflow (shelf-life, regulatory compliance, storage optimization)
- ✅ Describes typical input: time-series spectra with storage metadata
- ✅ Specifies typical dataset size: 5-20 entities × 10-30 timepoints = 50-600 spectra
- ✅ Notes temporal resolution: weekly (short studies) or monthly (long studies)

**2. Pipeline (Default):**
- ✅ 4-step pipeline clearly outlined:
  1. **Preprocessing:** ALS baseline → Savitzky-Golay smoothing → SNV normalization
  2. **Feature Extraction:** Degradation marker (e.g., oxidation ratio, peak height, integrated area)
  3. **Trajectory Fitting:** Linear or spline regression per entity; compute_degradation_trajectories()
  4. **Shelf-Life Estimation:** Time-to-threshold with CI; estimate_remaining_shelf_life()
- ✅ Each step 1-2 sentences, no code
- ✅ States default methods (e.g., "linear method" for trajectories, "95% CI via delta method" for shelf-life)

**3. Python Example (Synthetic):**
- ✅ 10-20 lines of annotated code
- ✅ Imports workflow functions
- ✅ Loads synthetic data
- ✅ Runs trajectory fitting
- ✅ Runs shelf-life estimation
- ✅ Saves outputs to file
- ✅ Comments explain each step

**4. CLI Example (with Config):**
- ✅ Shows YAML config file structure (10-15 lines)
- ✅ Config includes: `input_hdf5`, `time_col`, `entity_col`, `value_col`, `threshold`, `output_dir`
- ✅ Shows command: `foodspec aging --config examples/configs/aging_demo.yml`
- ✅ States expected outputs: trajectories.csv, shelf_life_estimates.csv, plots, report.md

**5. Interpretation:**
- ✅ **Qualitative:** Explains trajectory plot interpretation (slope = degradation rate, R² = fit quality)
- ✅ **Quantitative:** Provides template for reporting: "Degradation trajectory: slope = X ± Y per day, R² = Z, p < 0.001"
- ✅ **Reviewer phrasing:** Copy-paste template for Methods/Results sections
- ✅ Includes shelf-life reporting: "Estimated shelf-life: t_star = X days (95% CI: [Y, Z])"

**6. Summary:**
- ✅ 3-5 bullet points
- ✅ Reiterates: time-series fitting, shelf-life estimation, validation checks
- ✅ No new information (summary only)

**7. Statistical Analysis:**
- ✅ Explains why: test monotonic degradation, quantify slope significance
- ✅ Code example: 10-15 lines showing linear regression with statsmodels or scipy
- ✅ Interpretation template: "Significant degradation over time (slope = X, p = Y); t_star = Z days"
- ✅ Mentions alternative tests: Spearman correlation (non-linear trends), Mann-Kendall (monotonic trends)

---

## Section 4: Validation & Sanity Checks (✅)

### Subsections

1. **Success Indicators** (3 categories × 3-5 checks each)
2. **Failure Indicators** (6-10 warning signs with Problem/Fix pairs)
3. **Quality Thresholds** (table with Minimum/Good/Excellent criteria)

### Acceptance Criteria

**Success Indicators:**
- ✅ 3 categories:
  1. **Trajectory Fit Quality:** R² > 0.70, p < 0.05, residuals normally distributed
  2. **Chemical Plausibility:** Slope direction matches expectation (oxidation increases, freshness decreases)
  3. **Shelf-Life Estimates:** CI width < 30% of t_star, positive t_star, consistent across replicates
- ✅ Each indicator has ✅ symbol
- ✅ Each check 1 sentence or metric threshold
- ✅ Includes 9-15 total checks across categories

**Failure Indicators:**
- ✅ Lists 6-10 warning signs with **⚠️** symbol
- ✅ Each warning structured:
  - **Problem statement** (1 sentence)
  - **Cause** (why it happens)
  - **Fix** (actionable remedy)
- ✅ Examples:
  1. **Non-monotonic trajectory:** Degradation increases then decreases → Check storage conditions, verify preprocessing
  2. **Low R² (< 0.60):** High biological variability → Increase replicates, check for batch effects
  3. **Negative slope (freshness increases):** Ratio inverted or wrong metric → Verify degradation marker definition
  4. **Wide CI (> 50% of t_star):** Insufficient data or high noise → Increase timepoints, check instrument drift
  5. **All entities same shelf-life:** No variability captured → Check entity_col definition, verify distinct samples
- ✅ Each fix actionable (not "check data" but "add 5 more timepoints" or "verify time_col units")

**Quality Thresholds Table:**
- ✅ Table format with 4 columns: Metric | Minimum | Good | Excellent
- ✅ 5-8 metrics:
  - Trajectory R²: 0.60 | 0.80 | 0.95
  - Trajectory p-value: < 0.05 | < 0.01 | < 0.001
  - Shelf-life CI width: < 50% | < 30% | < 15%
  - Timepoints per entity: 10 | 20 | 30+
  - Replicates per timepoint: 2 | 3 | 5+
  - Residuals normality (Shapiro p): > 0.05 | > 0.10 | > 0.20
  - Within-Entity CV: < 25% | < 15% | < 10%
- ✅ All thresholds quantitative (no "adequate" or "moderate")

---

## Section 5: Parameters You Must Justify (⚙️)

### Subsections

1. **Critical Parameters** (4-6 parameters with structured justification templates)

### Acceptance Criteria

**Each Parameter Entry:**
- ✅ **Parameter name** (e.g., "Time Column (`time_col`)")
- ✅ **Default value** (e.g., `time_col="storage_days"`)
- ✅ **When to adjust** (2-3 scenarios with specific conditions)
- ✅ **Justification template** (copy-paste text for Methods section)

**Required Parameters:**

1. **Time Column (`time_col`)**
   - Default: `"storage_days"`
   - When to adjust: Use `"storage_weeks"` for short studies, `"storage_months"` for long studies
   - Justification: "Time measured in days from production date, recorded via batch metadata."

2. **Entity Column (`entity_col`)**
   - Default: `"sample_id"`
   - When to adjust: Use `"batch"` if tracking batch-level degradation, `"replicate_id"` if individual units
   - Justification: "Each entity represents an independent storage unit (n = X units tracked)."

3. **Value Column (`value_col`)**
   - Default: `"oxidation_ratio"` or `"peak_1742"` (carbonyl peak)
   - When to adjust: Use `"freshness_score"` for composite metrics, `"ratio_1655_1742"` for unsaturation/carbonyl
   - Justification: "Degradation quantified via carbonyl peak height (1742 cm⁻¹), normalized to CH2 stretch."

4. **Fitting Method (`method`)**
   - Default: `"linear"`
   - When to adjust: Use `"spline"` if acceleration observed (non-linear degradation), `"quadratic"` for explicit acceleration modeling
   - Justification: "Linear trajectory assumed for constant degradation rate; validated via residual analysis (R² > 0.80)."

5. **Shelf-Life Threshold (`threshold`)**
   - Default: Application-specific (e.g., 0.5 for ratio, 2.0 for peak height)
   - When to adjust: Use sensory threshold (trained panel), regulatory limit, or literature value
   - Justification: "Shelf-life threshold (X units) based on sensory panel acceptance limit (Ref: ISO 5492)."

6. **Confidence Level (`alpha`)**
   - Default: `0.05` (95% CI)
   - When to adjust: Use 0.10 (90% CI) for conservative estimates, 0.01 (99% CI) for regulatory submissions
   - Justification: "95% confidence intervals computed via delta method for time-to-threshold estimation."

---

## Section 6: When Results Cannot Be Trusted (⚠️)

### Subsections

1. **Red Flags for Aging Workflow** (8-10 numbered scenarios)

### Acceptance Criteria

**Each Red Flag Entry:**
- ✅ Numbered scenario (1-10)
- ✅ **Bold problem statement** (1 sentence)
- ✅ 2-3 bullet sub-points explaining consequences
- ✅ **Fix:** actionable remedy (1-2 sentences)

**Required Scenarios:**

1. **Single time-point or too few timepoints (< 5)**
   - Cannot fit trajectory; insufficient degrees of freedom
   - Shelf-life estimate unstable (wide CI)
   - **Fix:** Collect minimum 10 timepoints spanning expected shelf-life; use weekly/monthly sampling

2. **Thermal studies analyzed as aging (heating at 60°C treated as "storage")**
   - Accelerated degradation kinetics don't extrapolate to storage temps
   - Arrhenius activation energy differs for thermal vs oxidative processes
   - **Fix:** Use heating_quality_workflow for thermal studies; aging_workflow only for ≤ 25°C storage

3. **Undefined or arbitrary threshold (no chemical/sensory basis)**
   - Shelf-life estimate meaningless without defensible limit
   - Reviewer questions: "Why this threshold?"
   - **Fix:** Define threshold from sensory panel, regulatory standard, or literature precedent; document in Methods

4. **High variability, no replication (CV > 30% per timepoint, n=1 per time)**
   - Trajectory fit unreliable (low R²); CI too wide
   - Cannot distinguish degradation from measurement noise
   - **Fix:** Include 3+ replicates per timepoint; stratify by batch if needed

5. **Repeated scans of same sample (pseudo-replication)**
   - Technical replicates inflate significance (autocorrelation)
   - p-values misleadingly low
   - **Fix:** Use independent samples; average technical replicates before trajectory fitting

6. **Batch confounded with time (batch A at t=0, batch B at t=30)**
   - Cannot distinguish batch effects from degradation
   - Trajectory reflects batch differences, not aging
   - **Fix:** Include multiple batches at each timepoint; model batch as random effect

7. **No storage condition control (temperature varies 10-30°C)**
   - Temperature fluctuations dominate degradation kinetics
   - Arrhenius acceleration confounds time effect
   - **Fix:** Control storage temp (±2°C); document deviations; model temperature as covariate

8. **Extrapolation beyond measured range (fit 0-60 days, predict 365-day shelf-life)**
   - Linear fit may not hold over extended time (plateau, acceleration)
   - Prediction uncertainty increases quadratically with extrapolation distance
   - **Fix:** Only predict within 1.5× measured time range; collect long-term validation data

9. **Reversible changes treated as degradation (moisture loss, then gain)**
   - Non-monotonic trajectory (R² low, residuals non-random)
   - "Degradation" reversed by environmental change (not true aging)
   - **Fix:** Control moisture; verify monotonic trend; exclude reversible factors

10. **Statistical significance mistaken for practical shelf-life (p < 0.001 but t_star = 1000 days)**
    - Statistically significant trend may have negligible slope
    - Shelf-life far exceeds typical product lifecycle
    - **Fix:** Report effect sizes (slope magnitude); define minimum practical degradation rate

---

## Section 7: Recommended Defaults (📝)

### Subsections

1. **Default Parameter Set** (table)
2. **When to Deviate** (3-5 scenarios)

### Acceptance Criteria

**Default Parameter Table:**
- ✅ Table format with 3 columns: Parameter | Default Value | Rationale
- ✅ 6-8 parameters:
  - `time_col`: `"storage_days"` | Standard unit for shelf-life studies
  - `entity_col`: `"sample_id"` | Tracks independent storage units
  - `value_col`: `"oxidation_ratio"` | Common degradation marker
  - `method`: `"linear"` | Simplest model; adequate for constant-rate degradation
  - `threshold`: `0.5` (ratio) or `2.0` (peak) | Application-specific; must justify
  - `alpha`: `0.05` | Standard 95% confidence level
  - `preprocessing`: `["als_baseline", "savgol_smooth", "snv_normalize"]` | Removes baseline/scatter artifacts
- ✅ Each rationale 1 sentence

**When to Deviate:**
- ✅ Lists 3-5 scenarios requiring non-default parameters
- ✅ Each scenario: condition → parameter adjustment → reason
- ✅ Examples:
  1. **Accelerated degradation observed:** Use `method="spline"` or `"quadratic"` to capture non-linearity
  2. **Short shelf-life (< 30 days):** Increase sampling frequency (weekly instead of monthly); use `time_col="storage_days"`
  3. **Multi-batch study:** Add `batch` as grouping factor; fit separate trajectories per batch
  4. **High instrument noise:** Increase smoothing (larger Savgol window); check SNR before fitting
  5. **Regulatory submission:** Use `alpha=0.01` (99% CI) for conservative estimates

---

## Section 8: See Also (🔗)

### Subsections

1. **Methods** (5-8 relevant method pages)
2. **Examples** (2-3 example scripts)
3. **API** (3-5 key functions/classes)

### Acceptance Criteria

**Methods Links:**
- ✅ Lists 5-8 method pages as markdown links:
  - [Baseline Correction](../../methods/preprocessing/baseline_correction.md)
  - [Normalization & Smoothing](../../methods/preprocessing/normalization_smoothing.md)
  - [Linear Regression](../../methods/statistics/linear_regression.md)
  - [Confidence Intervals](../../methods/statistics/confidence_intervals.md)
  - [Time-Series Analysis](../../methods/statistics/time_series_analysis.md)
  - [Model Evaluation](../../methods/chemometrics/model_evaluation_and_validation.md)
  - [Peak Ratios](../../methods/features/peak_ratios.md)
  - [Oxidation Markers](../../methods/features/oxidation_markers.md)
- ✅ Each link valid (page exists)
- ✅ Links relevant to aging workflow (no classification or clustering methods)

**Examples Links:**
- ✅ Lists 2-3 example scripts:
  - [Aging Quickstart](../../examples/aging_quickstart.md)
  - [Shelf-Life Estimation Demo](../../examples/shelf_life_demo.md)
  - [MOATS Demo (includes aging)](../../examples/moats_demo.md)
- ✅ Each link points to example with aging workflow code

**API Links:**
- ✅ Lists 3-5 key API references:
  - [`foodspec.workflows.aging.compute_degradation_trajectories`](../../api/workflows/aging.md)
  - [`foodspec.workflows.shelf_life.estimate_remaining_shelf_life`](../../api/workflows/shelf_life.md)
  - [`foodspec.stats.time_metrics.linear_slope`](../../api/stats/time_metrics.md)
  - [`foodspec.stats.time_metrics.quadratic_acceleration`](../../api/stats/time_metrics.md)
  - [`foodspec.viz.time_series.plot_trajectories`](../../api/viz/time_series.md)
- ✅ Each link points to API docstring page (mkdocstrings)

---

## Cross-Cutting Requirements

### Consistency with Other Workflows

- ✅ Uses same section headers (📋 Standard Header, 🔬 MRE, ✅ Validation, ⚙️ Parameters, ⚠️ When Cannot Trust, 🔗 See Also)
- ✅ Follows same structure within each section (subsection order matches heating_quality_monitoring.md)
- ✅ Uses same emoji conventions (✅ for success, ⚠️ for warnings, ❌ for anti-patterns)
- ✅ Code examples formatted identically (triple-backticks, language specified, output in YAML blocks)

### Evidence-Based Content

- ✅ All parameter defaults match source code (`aging.py`, `shelf_life.py`)
- ✅ Quality thresholds informed by existing validation examples (moats_demo.py)
- ✅ Chemical plausibility checks reference real degradation markers (carbonyl, unsaturation)
- ✅ Failure modes drawn from common user errors (documented in issue tracker or support channels)

### Accessibility & Usability

- ✅ MRE code runs without modification (copy-paste ready)
- ✅ All figures referenced exist in `docs/assets/workflows/aging/` or are generated by MRE
- ✅ Jargon defined on first use (e.g., "delta method" linked to stats glossary)
- ✅ No broken links (all method/example/API pages exist)
- ✅ Estimated reading time: 15-20 minutes (1500-2000 words excluding code)

---

## Summary Table: Section Lengths

| Section | Target Length | Acceptance Check |
|---------|---------------|------------------|
| Standard Header | 150-200 words | Purpose (20 words), When to Use (70 words), Inputs (40 words), Outputs (40 words), Assumptions (30 words) |
| MRE | 300-400 words + 50-80 lines code | Option A (40 lines), Option B (30 lines), Expected Output (30 words + figure) |
| Step-by-Step Pipeline | 400-600 words + 30-50 lines code | 7 subsections, each 50-100 words except code examples |
| Validation & Sanity Checks | 400-600 words | Success (150 words), Failure (250 words), Thresholds (50 words table) |
| Parameters You Must Justify | 300-400 words | 6 parameters × 50-70 words each |
| When Cannot Be Trusted | 300-400 words | 8-10 red flags × 30-40 words each |
| Recommended Defaults | 150-200 words | Table (100 words), Deviations (50-100 words) |
| See Also | 50-100 words | 15-20 links with brief annotations |

**Total Target Length:** 2,000-3,000 words + 100-150 lines code

---

## Validation Checklist for Completed Page

Before considering the page complete, verify:

- [ ] All 8 sections present with correct headers and emojis
- [ ] MRE code runs successfully (tested on clean environment)
- [ ] All internal links resolve (no 404s)
- [ ] All figures referenced exist or are generated by MRE
- [ ] Quality thresholds align with source code defaults
- [ ] Parameter justification templates are copy-paste ready
- [ ] Red flags cover common user errors (verified against support tickets)
- [ ] Structure matches heating_quality_monitoring.md (diff < 10% section lengths)
- [ ] No heating workflow terminology used (thermal, frying, elevated temp)
- [ ] Aging vs heating distinction clear (storage/passive vs thermal/active)
- [ ] Estimated reading time: 15-20 minutes
- [ ] Tone matches other workflow pages (imperative, concise, evidence-based)

---

## Notes for Implementation

**Figures to Create:**
- `docs/assets/workflows/aging/degradation_trajectories.png` — Time-series plot with fitted linear/spline trajectories
- `docs/assets/workflows/aging/shelf_life_estimates.png` — Bar chart with t_star ± CI per entity
- `docs/assets/workflows/aging/residual_plot.png` — Residuals vs time to validate linear fit

**Synthetic Data Generator:**
- Create `foodspec.demo.synthetic_aging_dataset()` function if not exists
- Generate 5 entities × 20 timepoints × 3 replicates = 300 spectra
- Degradation: linear slope 0.01-0.03 per day + Gaussian noise (σ=0.05)
- Time range: 0-180 days (6 months)
- Metadata: entity_id, time_days, oxidation_ratio, storage_temp (fixed 20°C)

**Config File Template:**
- Create `examples/configs/aging_demo.yml` with standard parameters
- Document in CLI example section
- Include comments explaining each parameter

**Comparison Table (Aging vs Heating):**
- Add dedicated subsection in "When to Use" or "Standard Header"
- 2-column table: Aging Workflow | Heating Workflow
- 5-7 rows: Timescale, Temperature, Purpose, Inputs, Typical Outputs, Statistical Model, Example Use Case
- Clarifies distinction; prevents workflow misapplication
