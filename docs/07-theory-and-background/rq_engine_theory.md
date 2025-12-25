# Theory – Ratio-Quality (RQ) Engine Rationale

Why these metrics?
- **Peaks vs ratios**: ratiometric analysis reduces sensitivity to illumination/collection differences and focuses on relative chemical changes.
- **Stability (CV, MAD)**: captures reproducibility; low CV/MAD indicates robust markers; MAD is robust to outliers.
- **Discriminative power**: ANOVA/Kruskal with FDR controls false discoveries across many features; effect sizes quantify practical significance; model-based importance (RF/LR) complements tests.
- **Heating trends**: linear slopes capture directional change; Spearman ρ tests monotonicity; both corrected for multiple testing (FDR) to avoid false positives across many ratios.
- **Oil vs chips divergence**: matrix effects can alter mean/CV/trend; divergence metrics + effect sizes highlight whether a marker is matrix-robust or matrix-sensitive.
- **Normalization comparisons**: reference-peak vs vector/area/max normalization shows whether conclusions are robust to scaling choices.
- **Clustering metrics**: silhouette/ARI quantify unsupervised structure; helpful to see if oils/chips form natural groups without labels.

How outputs inform conclusions:
- High discrimination/importance → strong markers for identity/QA.
- Low CV/MAD → reproducible markers; good for monitoring.
- Significant trends/monotonicity → heating/processing markers.
- Divergence + effect sizes → matrix robustness or sensitivity.
- Normalization/clustering robustness → confidence that findings are not artifacts of scaling or labels.

See also: [cookbook_rq_questions.md](../03-cookbook/cookbook_rq_questions.md) and tutorials under `02-tutorials/` for applied examples.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for RQ (Ratio-Quality) engine results:**

1. **High discrimination/importance without independent validation (R²_discrimination = 0.95 on training data)**
   - Training metrics optimistic; test set performance is ground truth
   - Cross-validation needed to assess generalization
   - **Fix:** Use cross-validation; validate on held-out data; report test performance

2. **Low CV/MAD without checking what's being replicated (3 spectra of 1 sample reported as 3 "replicates")**
   - Technical replicates are autocorrelated; not independent biological replicates
   - Apparent reproducibility artificial
   - **Fix:** Use distinct samples; document replication structure; analyze at sample level

3. **Significant trends/monotonicity claimed without statistical test (visual trend in scattered data)**
   - Outliers or noise can create visual trends
   - Statistics required for defensible monotonicity
   - **Fix:** Run monotonicity test (Spearman ρ, Mann–Kendall tau); report p-value and effect size

4. **Divergence/matrix effect interpreted without control (oils and chips differ, attributed to matrix; not checked against other factors)**
   - Confounding variables (source, preprocessing) may explain divergence
   - Attribution premature without controls
   - **Fix:** Include controls (same source oils vs chips, same processing); perform ANOVA with batch/source factors

5. **Normalization/scaling robustness not tested (RQ output unchanged with different scaling, but not verified)**
   - Some metrics sensitive to scaling; sensitivity unknown unless tested
   - Results may depend on arbitrary normalization choice
   - **Fix:** Test robustness to alternative normalizations; report metric range across normalization strategies

6. **RQ metrics interpreted as ground truth without external validation (discrimination ratio high, so authentication is validated)**
   - RQ engine produces ratios; chemical/biological validity requires independent confirmation
   - High metric doesn't guarantee real-world utility
   - **Fix:** Validate RQ outputs against orthogonal methods (HPLC, sensory, certified standards)

7. **Batch effects or confounders not visualized in RQ outputs (no batch-colored PCA or QC charts)**
   - Important confounders can be hidden in aggregate metrics
   - Batch or temporal patterns undetected
   - **Fix:** Visualize RQ outputs colored by batch/time/source; include QC time-series; report confounding effects

8. **Interpretation of effect sizes without context (effect size = 0.5, interpretation unclear)**
   - Cohen's d = 0.5 is "medium"; practical significance depends on domain
   - Same effect size in authentication (critical) vs trend (informational) may have different meaning
   - **Fix:** Contextualize effect sizes; compare to operational specifications; discuss practical significance
