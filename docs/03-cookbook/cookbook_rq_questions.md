# Cookbook: RQ Questions (RQ1–RQ6)

- RQ1: Discrimination (oil types) – use classification protocol, ANOVA/importances.
- RQ2: Stability – CV tables/plots.
- RQ3: Discriminative ratios – ANOVA/importance tables.
- RQ4: Thermal degradation markers – heating trend analyses.
- RQ5: Minimal panel – compact marker set.
- RQ6: Clustering – silhouette/ARI metrics.
# Cookbook – RQ questions (RQ1–RQ6)

Use these recipes to answer common Ratio-Quality questions. Each entry points to protocols, bundle outputs, and deeper references.

## RQ1 – Can I tell oils apart?
- **What:** Classification + discriminative features.  
- **Protocol:** `examples/protocols/oil_basic.yaml`.  
- **CLI:** `foodspec-run-protocol --input examples/data/oils.csv --protocol examples/protocols/oil_basic.yaml`  
- **Bundle outputs:** `tables/discriminative_summary.csv`, confusion matrix/ROC plots, minimal panel.  
- **More:** [oil_discrimination_basic.md](../02-tutorials/oil_discrimination_basic.md)

## RQ2 – Which peaks/ratios are stable?
- **What:** Stability metrics (CV, MAD).  
- **Protocol:** same as RQ1 or any with RQ enabled.  
- **Bundle outputs:** `tables/stability_summary.csv`, stability barplots.  
- **More:** [rq_engine_theory.md](../07-theory-and-background/rq_engine_theory.md)

## RQ3 – Which ratios matter most?
- **What:** Feature importance + ANOVA/Kruskal with FDR.  
- **Protocol:** oil discrimination; minimal panel enabled.  
- **Bundle outputs:** `tables/discriminative_summary.csv`, importance plots, `tables/minimal_panel.csv`.  
- **More:** [cookbook_preprocessing.md](cookbook_preprocessing.md)

## RQ4 – Do markers track heating?
- **What:** Trends vs heating_stage (slopes, Spearman ρ, monotonicity, FDR).  
- **Protocol:** `examples/protocols/oil_heating.yaml`.  
- **Bundle outputs:** `tables/heating_trend_summary.csv`, trend plots.  
- **More:** [thermal_stability_tracking.md](../02-tutorials/thermal_stability_tracking.md)

## RQ5 – Minimal marker set?
- **What:** Minimal panel meeting target accuracy (L1-logreg/greedy).  
- **Protocol:** enable minimal panel in RQ step.  
- **Bundle outputs:** `tables/minimal_panel.csv`, accuracy vs feature count.  
- **More:** [oil_discrimination_basic.md](../02-tutorials/oil_discrimination_basic.md)

## RQ6 – Do oils cluster naturally?
- **What:** Unsupervised clustering (k-means/hierarchical) with silhouette/ARI.  
- **Protocol:** enable clustering in RQ step.  
- **Bundle outputs:** clustering metrics table, PCA/MDS plots (`figures/pca_*`, `mds_*`).  
- **More:** [chemometrics_and_ml_basics.md](../07-theory-and-background/chemometrics_and_ml_basics.md)
