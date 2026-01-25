"""Agreement and consistency metrics for multi-modal fusion."""
from __future__ import annotations


from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def modality_agreement_kappa(
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute Cohen's kappa between all pairs of modality predictions.

    Args:
        predictions: Mapping from modality name to 1D array of class predictions.

    Returns:
        Pairwise kappa matrix (symmetric DataFrame).

    Raises:
        ValueError: If fewer than two modalities are provided.
    """
    if len(predictions) < 2:
        raise ValueError("Need at least two modalities for agreement.")
    modalities = list(predictions.keys())
    n = len(modalities)
    kappa_matrix = np.zeros((n, n), dtype=float)
    for i, m1 in enumerate(modalities):
        for j, m2 in enumerate(modalities):
            if i == j:
                kappa_matrix[i, j] = 1.0
            elif i < j:
                kappa = cohen_kappa_score(predictions[m1], predictions[m2])
                kappa_matrix[i, j] = kappa
                kappa_matrix[j, i] = kappa
    return pd.DataFrame(kappa_matrix, index=modalities, columns=modalities)


def modality_consistency_rate(
    predictions: Dict[str, np.ndarray],
) -> float:
    """Fraction of samples where all modalities agree.

    Args:
        predictions: Mapping from modality name to 1D array of class predictions.

    Returns:
        Proportion of samples with unanimous predictions across modalities (0.0 to 1.0).

    Raises:
        ValueError: If fewer than two modalities are provided.
    """
    if len(predictions) < 2:
        raise ValueError("Need at least two modalities for consistency.")
    pred_arrays = list(predictions.values())
    pred_arrays[0].shape[0]
    stacked = np.vstack(pred_arrays).T  # (n_samples, n_modalities)
    unanimous = np.all(stacked == stacked[:, [0]], axis=1)
    return float(unanimous.mean())


def cross_modality_correlation(
    feature_dict: Dict[str, np.ndarray],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation between feature matrices of different modalities.

    For each pair of modalities, compute average correlation between their features.

    Parameters
    ----------
    feature_dict : Dict[str, np.ndarray]
        Mapping from modality to feature matrix (n_samples, n_features).
    method : str
        Correlation method: "pearson" or "spearman".

    Returns
    -------
    pd.DataFrame
        Pairwise average correlation matrix.
    """
    if len(feature_dict) < 2:
        raise ValueError("Need at least two modalities for cross-correlation.")
    modalities = list(feature_dict.keys())
    n = len(modalities)
    corr_matrix = np.zeros((n, n), dtype=float)
    for i, m1 in enumerate(modalities):
        for j, m2 in enumerate(modalities):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                X1 = feature_dict[m1]
                X2 = feature_dict[m2]
                if X1.shape[0] != X2.shape[0]:
                    raise ValueError("Feature matrices must have same number of samples.")
                # Compute correlation between all pairs of features
                from scipy.stats import pearsonr, spearmanr

                corr_func = pearsonr if method == "pearson" else spearmanr
                corrs = []
                for f1 in range(X1.shape[1]):
                    for f2 in range(X2.shape[1]):
                        r, _ = corr_func(X1[:, f1], X2[:, f2])
                        if np.isfinite(r):
                            corrs.append(r)
                avg_corr = float(np.mean(corrs)) if corrs else 0.0
                corr_matrix[i, j] = avg_corr
                corr_matrix[j, i] = avg_corr
    return pd.DataFrame(corr_matrix, index=modalities, columns=modalities)


__all__ = [
    "modality_agreement_kappa",
    "modality_consistency_rate",
    "cross_modality_correlation",
]
