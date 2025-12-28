# Design: Statistics & Data Analysis Layer for FoodSpec

This document outlines the planned architecture, APIs, and documentation for a comprehensive statistics and data-analysis layer in FoodSpec. It is written to guide future implementation and to align with the book-style documentation.

## 1. Goals
- Provide statistically sound, chemometrics-aligned tools for common food spectroscopy tasks.
- Integrate smoothly with `FoodSpectrumSet`/`HyperSpectralCube` and existing workflows (oil auth, heating, mixtures, QC).
- Offer clear, documented APIs with reproducibility in mind (configurable, metadata-aware).
- Map methods to documentation chapters and figures for a textbook + protocol presentation.

### Current implementation status
- Implemented: t-tests (one/two-sample, paired), one-way ANOVA, MANOVA (with optional statsmodels), Tukey HSD, nonparametric (Kruskal–Wallis, Mann–Whitney U, Wilcoxon, Friedman), correlations (Pearson/Spearman, cross-correlation), effect sizes (Cohen’s d, eta/partial eta), robustness helpers (bootstrap/permutation), study design summaries.
- Documented: stats theory chapters, nonparametric/robustness, metrics integration, workflow examples calling stats functions.
- Future work: two-way ANOVA support and additional post-hoc corrections (Bonferroni/FDR) beyond Tukey; richer MANOVA diagnostics; more real-data examples for vendor-specific datasets.

## 2. Proposed package architecture
- New subpackage: `foodspec.stats` with submodules:
  - `hypothesis_tests`: t-tests (one-sample, two-sample, paired), ANOVA (one-way; two-way remains future work), MANOVA (via statsmodels if available), post-hoc (Tukey HSD). Nonparametric (Kruskal–Wallis, Mann–Whitney U, Wilcoxon, Friedman) implemented.
  - `correlations`: Pearson, Spearman, Kendall (optional), cross-correlation for sequences/time series (synchronous/asynchronous mapping), correlation heatmaps.
  - `effects`: Effect sizes (Cohen’s d, eta-squared, partial eta-squared), simple confidence intervals.
  - `design`: Study design summaries (group sizes, balance check, missingness), utility to warn when design is unsuitable for ANOVA/MANOVA.
- Dependencies:
  - Prefer SciPy for basic tests (ttest, anova/KW).
  - Statsmodels for MANOVA/Tukey HSD if available; otherwise emit clear ImportErrors explaining the optional dependency.
  - pandas/NumPy for data handling. Avoid heavy new deps.
- API style:
  - Functions accept either raw arrays/Series/DataFrames **and** `FoodSpectrumSet` (using metadata columns for grouping).
  - Return typed results (dataclasses or dicts) with statistics, p-values, effect sizes, and assumptions summaries.

## 3. Core statistical methods (to implement)
### Hypothesis testing
- One-sample t-test
- Two-sample t-test (independent)
- Paired t-test
- One-way ANOVA
- Two-way ANOVA (future work; design info needed)
- MANOVA (via statsmodels MANOVA; optional if dependency present)
- Post-hoc: Tukey HSD (via statsmodels); Bonferroni/FDR (future work)
- Nonparametric: Kruskal–Wallis, Mann–Whitney U, Wilcoxon, Friedman implemented

### Correlations & mapping
- Pearson, Spearman; Kendall optional.
- Cross-correlation for sequences/time series (e.g., heating stages).
- “Synchronous mapping” (same-condition correlation between spectral features and quality metrics).
- “Asynchronous mapping” (cross-correlation for lagged relationships).

### Effect size & summary
- Cohen’s d (two-group).
- Eta-squared / partial eta-squared for ANOVA.
- Simple CI helpers where meaningful.

## 4. Mapping to workflows
- **Oil authentication:** ANOVA/Tukey on peak ratios or PC scores across oil types; correlation between ratios and quality labels if present.
- **Heating quality:** ANOVA across stages; correlation of ratios with time; cross-correlation for time series; effect sizes for differences between stages.
- **Mixtures:** MANOVA or correlation between proportions and PCs/ratios; regression diagnostics.
- **Batch QC:** Correlation matrices of reference vs suspect sets; tests of mean differences on key ratios/PCs.

## 5. Documentation plan (new stats chapters)
- `docs/stats/introduction_to_statistical_analysis.md`: role of stats in spectroscopy; link to chemometrics; when to test vs model.
- `docs/stats/hypothesis_testing_in_food_spectroscopy.md`: t-tests, ANOVA, MANOVA, assumptions, examples on ratios/PCs.
- `docs/stats/anova_and_manova.md`: deeper dive; assumptions; design requirements; post-hoc (Tukey); effect sizes.
- `docs/stats/t_tests_effect_sizes_and_power.md`: t-tests, Cohen’s d, power considerations (outline).
- `docs/stats/correlation_and_mapping.md`: Pearson/Spearman, cross-correlation for time series, interpretation in food contexts.
- `docs/stats/study_design_and_data_requirements.md`: replication, balance vs unbalance, randomization, link to instrument noise/drift.
- `docs/stats/nonparametric_methods_and_robustness.md`: covers Kruskal–Wallis, Wilcoxon, Mann–Whitney U, Friedman, and robustness via bootstrap/permutation.
- Add a short “How to explore the API” section (API hub + keyword index) referencing `foodspec.stats`.

## 6. Data acquisition & study design chapter
- Cover sample size, replication, randomization, balance vs unbalance, handling drift, and implications for ANOVA/MANOVA power.
- Placement: under `docs/stats/study_design_and_data_requirements.md` (stats section) and cross-link from Foundations and Protocols.

## 7. Figures & diagrams
- Python-generated: boxplots, ANOVA mean plots with CIs, Tukey HSD plot (if available), correlation heatmaps, residual plots.
- Mermaid decision trees:
  - “Which test should I use?” (numeric, #groups, pairing, normality assumption).
  - “Is my design sufficient for ANOVA/MANOVA?” (balance, replication, missingness).

## 8. Minimal scaffolding (code)
- Ensure `src/foodspec/stats/__init__.py` exposes the public API (no placeholders).
- Core modules implemented (`hypothesis_tests.py`, `correlations.py`, `effects.py`, `design.py`, robustness helpers) with documented APIs; future extensions limited to two-way ANOVA and additional post-hoc corrections.

## 9. Integration points
- Functions should accept `FoodSpectrumSet` and group columns; use metadata to subset/group.
- Return structures compatible with reporting (e.g., to JSON) and plotting (pandas-friendly).
- Link to protocol/benchmarking: enable logging of test parameters and results for standardized reports.

## 10. Next steps (future implementation)
- Implement t-tests/ANOVA/MANOVA wrappers using SciPy/statsmodels; add effect size calculators.
- Add correlation utilities and plotting helpers (heatmaps) leveraging matplotlib/seaborn (if already allowed).
- Extend workflow docs with statistical tests examples (oil ratios, heating trends).
- Wire mkdocstrings for `foodspec.stats` into API pages once implemented.

---

This design keeps implementation light for now and prioritizes a clear API surface, documentation plan, and integration with existing workflows and reproducibility practices.
