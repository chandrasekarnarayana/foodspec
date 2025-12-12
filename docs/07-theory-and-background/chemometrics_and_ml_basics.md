# Theory – Chemometrics & ML Basics

This page summarizes core concepts underpinning FoodSpec analyses. For worked examples, see Tutorials (02) and Cookbook (03).

## Core methods
- **PCA**: unsupervised dimensionality reduction; reveals structure/clusters and supports clustering metrics (silhouette/ARI).
- **PLS/PLS-DA**: regression/classification with latent variables (not always needed for simple ratio sets but common in spectroscopy).
- **Classification**: logistic regression (often with L1 for minimal panels), random forests for nonlinear importance; balanced accuracy/confusion matrices for evaluation.

## Why cross-validation matters
- Prevents optimistic bias; estimates generalization performance.
- Batch-aware or group-aware splits avoid leakage across instruments/batches.
- Nested CV supports feature selection/hyperparameter tuning without reusing test folds.

## Scaling/normalization
- Standardization and ratiometric features stabilize intensity variations; see preprocessing recipes and RQ theory for why specific ratios are used.

See also: [cookbook_validation.md](../03-cookbook/cookbook_validation.md) and [oil_discrimination_basic.md](../02-tutorials/oil_discrimination_basic.md) for applied examples.

How FoodSpec uses these:
- PCA/MDS visualizations and clustering metrics in RQ outputs.
- Classification (LR/RF) for discrimination and minimal panels.
- Cross-validation strategies (batch-aware/nested) for honest performance estimates.
