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
