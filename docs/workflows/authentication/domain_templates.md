# Workflow: Domain Templates (Meat, Microbial, and Beyond)

## 📋 Standard Header

**Purpose:** Provide pre-configured workflow templates for specific food domains (meat, dairy, grains, microbial ID) with sensible defaults.

**When to Use:**
- Quick-start authentication for common food matrices (meat species, microbial strains)
- Leverage domain-specific preprocessing and feature defaults
- Compare results to published domain benchmarks
- Adapt existing validated workflows to similar matrices
- Training users on standardized approaches for specific domains

**Inputs:**
- Format: Same as oil authentication (HDF5/CSV with spectra)
- Required metadata: Domain-specific labels (e.g., `meat_type`, `microbial_species`, `grain_variety`)
- Optional metadata: `source`, `treatment`, `preparation_method`
- Wavenumber range: Domain-dependent (meat: 600–1800 cm⁻¹; microbial: 800–1800 cm⁻¹)
- Min samples: 50+ per class (same as general authentication)

**Outputs:**
- Same as oil authentication workflow (confusion matrix, PCA, metrics, report)
- Domain-specific interpretation notes in report

**Assumptions:**
- Domain template validated on representative samples from target domain
- Sample preparation matches template specifications (powder, liquid, cuvette type)
- Spectroscopy method matches template (Raman vs FTIR)
- Within-domain variability captured in training data (varieties, sources, seasons)

---

## What this chapter covers
- How domain templates map to the oil-auth style pipeline (preprocessing + classifier).
- Typical metadata/label expectations per domain (meat_type, species/strain, etc.).
- When to use a domain template vs configure your own workflow.
- Links to meat/microbial tutorial pages for runnable examples.

## Outline
- **Template concept:** Thin wrappers around preprocessing + classification; default features/models.
- **Meat:** Raman/FTIR use cases; label expectations; adapting oil defaults.
- **Microbial:** Spectral IDs; class imbalance considerations; QC steps.
- **Dairy/adulteration (future):** Apply the same preprocessing/ratios/PCA + classifier pattern; record instrument (FTIR/NIR), matrix (milk powders/liquids), target labels (adulterant level/type); reuse reproducibility fields for plots/reports.
- **Spices/grains (future):** Heterogeneous matrices; emphasize preprocessing choices (baseline, normalization), feature selection (key bands), and QC/statistics similar to oil workflows.
- **Extensibility:** Adding new domain templates; using CLI `domains` command (if applicable).
- **Pointers:** See `../meat_tutorial.md` and `../microbial_tutorial.md` for code/CLI recipes.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for domain-specific templates:**

1. **Domain template applied without verifying it covers sample diversity**
   - Template trained on limited subset of domain; real samples more variable
   - Boundary cases (organic vs conventional, rare varieties) not represented
   - **Fix:** Include diverse sources, varieties, and processing methods in template validation

2. **Spectroscopy method mismatch (template for Raman applied to FTIR data)**
   - Different methods give different spectra; models don't transfer without retraining
   - Spectral ranges, baseline, and peak positions different
   - **Fix:** Use method-specific template; validate transfer before cross-method deployment

3. **Sample preparation not matching template assumptions (template assumes dried powder, new samples are liquid)**
   - Preparation dramatically affects spectra; cuvette, path length, temperature critical
   - Template model won't work if prep fundamentally different
   - **Fix:** Match sample prep to template specifications; retrain if prep changes

4. **Seasonal or temporal variation not addressed (template trained in summer, deployed in winter)**
   - Ambient temperature, storage time, ripeness, and harvest effects not captured
   - Spectra may shift seasonally, violating template assumptions
   - **Fix:** Include samples from different seasons/harvest times; validate temporal generalization

5. **Reference database for domain template incomplete (missing adulterant types, new varieties)**
   - Template can only detect adulterants in training set
   - Novel adulterant or variety will be misclassified
   - **Fix:** Continuously update reference database; validate on new adulterants before deployment

6. **Domain template assumes homogeneous matrix (oil domain applied to olive oils, but different cultivars have very different compositions)**
   - Intra-domain variability can exceed inter-domain differences
   - Single template insufficient
   - **Fix:** Stratify by important factors (variety, origin, processing); use multiple templates or include factors in model

7. **No cross-validation across domain subsets (all training on one supplier/region)**
   - Supplier-specific patterns learned; won't generalize
   - Cross-source/region validation reveals true robustness
   - **Fix:** Include multiple suppliers/regions in training; validate on held-out sources

8. **Domain boundaries unclear (when is sample "in domain" vs "out of domain"?)**
   - No objective rule for when template applies; user confusion
   - Can lead to inappropriate use
   - **Fix:** Define domain explicitly (e.g., "olive oils from Mediterranean, post-harvest, stored <2 years"); flag out-of-domain samples

## Next steps
- Use a template for rapid prototyping; switch to custom pipelines for specialized datasets.
- Explore **Protocols & reproducibility** to document template use in studies.
