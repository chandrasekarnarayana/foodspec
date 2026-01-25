"""
Interpretability utilities for model explanation and feature importance.

Provides methods to extract and visualize model coefficients, identify
top important features, and generate interpretability reports.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def extract_linear_coefficients(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract coefficients from linear models.
    
    Extracts coefficients from logistic regression, linear SVM, or other
    linear models. Returns DataFrame sorted by absolute coefficient magnitude.
    For multiclass models, includes per-class coefficients.
    
    Parameters
    ----------
    model : fitted model
        Fitted model supporting .coef_ attribute (e.g., LogisticRegression,
        LinearSVC, Ridge, Lasso, etc.)
    feature_names : list[str]
        Feature names corresponding to coefficient indices
    
    Returns
    -------
    DataFrame with columns:
        - feature : str
            Feature name
        - coefficient : float (univariate) or coef_{class} (multiclass)
            Coefficient value per feature
        - abs_coefficient : float
            Absolute value for sorting
        - [coef_class_0, coef_class_1, ...] : multiclass only
            Per-class coefficients if model is multiclass
    
    Raises
    ------
    AttributeError
        If model doesn't have .coef_ attribute
    ValueError
        If feature_names length doesn't match number of coefficients
    
    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> 
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> features = ['f0', 'f1', 'f2', 'f3', 'f4']
    >>> 
    >>> model = LogisticRegression(random_state=42)
    >>> model.fit(X, y)
    >>> 
    >>> coef_df = extract_linear_coefficients(model, features)
    >>> print(coef_df.head())
    """
    if not hasattr(model, "coef_"):
        raise AttributeError(
            f"Model {type(model).__name__} does not have .coef_ attribute"
        )
    
    coef = np.asarray(model.coef_)
    feature_names = list(feature_names)
    
    # Handle shape: (n_features,) for binary or (n_classes, n_features)
    # Note: Some models (e.g., LogisticRegression for binary) have shape (1, n_features)
    is_binary = (coef.ndim == 2 and coef.shape[0] == 1) or coef.ndim == 1
    
    if is_binary:
        # Binary classification: flatten to 1D
        if coef.ndim == 2:
            coef = coef[0]
        
        if len(feature_names) != len(coef):
            raise ValueError(
                f"feature_names length ({len(feature_names)}) doesn't match "
                f"coefficients ({len(coef)})"
            )
        
        df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        })
    else:
        # Multiclass
        n_classes, n_features = coef.shape
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) doesn't match "
                f"coefficients ({n_features})"
            )
        
        # Compute mean absolute coefficient across classes
        mean_abs_coef = np.abs(coef).mean(axis=0)
        
        df = pd.DataFrame({
            "feature": feature_names,
            "abs_coefficient": mean_abs_coef,
        })
        
        # Add per-class coefficients
        for c in range(n_classes):
            df[f"coef_class_{c}"] = coef[c]
        
        # Add mean coefficient
        df["mean_coefficient"] = coef.mean(axis=0)
    
    # Sort by absolute coefficient magnitude (descending)
    df = df.sort_values("abs_coefficient", ascending=False, ignore_index=True)
    
    return df


def top_k_features(
    coef_df: pd.DataFrame,
    k: int = 20,
) -> pd.DataFrame:
    """
    Select top-k most important features from coefficient DataFrame.
    
    Parameters
    ----------
    coef_df : pd.DataFrame
        Output from extract_linear_coefficients()
    k : int, default=20
        Number of top features to return
    
    Returns
    -------
    pd.DataFrame
        Top k rows (already sorted by importance)
    
    Raises
    ------
    ValueError
        If k is negative or if coef_df doesn't have required columns
    
    Examples
    --------
    >>> coef_df = extract_linear_coefficients(model, feature_names)
    >>> top_20 = top_k_features(coef_df, k=20)
    >>> print(top_20)
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    
    if "abs_coefficient" not in coef_df.columns:
        raise ValueError(
            "coef_df must have 'abs_coefficient' column from extract_linear_coefficients"
        )
    
    # Already sorted by extract_linear_coefficients
    return coef_df.head(k).reset_index(drop=True)


def coefficient_summary(
    coef_df: pd.DataFrame,
    decimals: int = 4,
) -> str:
    """
    Generate human-readable summary of coefficients.
    
    Parameters
    ----------
    coef_df : pd.DataFrame
        Output from extract_linear_coefficients()
    decimals : int, default=4
        Number of decimal places for rounding
    
    Returns
    -------
    str
        Formatted summary text
    
    Examples
    --------
    >>> summary = coefficient_summary(coef_df)
    >>> print(summary)
    """
    top_10 = top_k_features(coef_df, k=10)
    
    lines = ["Top 10 Important Features (by |coefficient|):\n"]
    lines.append("Feature | Coefficient")
    lines.append("-" * 40)
    
    for idx, row in top_10.iterrows():
        coef_val = row.get("coefficient", row.get("mean_coefficient", 0))
        lines.append(f"{row['feature']:30s} | {coef_val:8.{decimals}f}")
    
    return "\n".join(lines)


def to_markdown_coefficients(
    coef_df: pd.DataFrame,
    k: int = 20,
    caption: str = "Top Model Coefficients",
) -> str:
    """
    Export coefficients as markdown table.
    
    Parameters
    ----------
    coef_df : pd.DataFrame
        Output from extract_linear_coefficients()
    k : int, default=20
        Number of features to include
    caption : str
        Table caption
    
    Returns
    -------
    str
        Markdown table
    
    Examples
    --------
    >>> md = to_markdown_coefficients(coef_df, k=15)
    >>> print(md)
    """
    top_k = top_k_features(coef_df, k=k)
    
    lines = [f"\n**{caption}**\n"]
    
    # Build column headers
    if "coef_class_0" in top_k.columns:
        # Multiclass
        n_classes = sum(1 for c in top_k.columns if c.startswith("coef_class_"))
        cols = ["feature", "mean_coefficient", "abs_coefficient"] + \
               [f"coef_class_{i}" for i in range(n_classes)]
    else:
        # Binary
        cols = ["feature", "coefficient", "abs_coefficient"]
    
    # Filter to available columns
    cols = [c for c in cols if c in top_k.columns]
    
    # Header
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["-" * 15 for _ in cols]) + "|")
    
    # Rows
    for idx, row in top_k.iterrows():
        values = [str(row[c])[:14] for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def compare_coefficients(
    coef_df1: pd.DataFrame,
    coef_df2: pd.DataFrame,
    model_names: tuple[str, str] = ("Model 1", "Model 2"),
    k: int = 20,
) -> pd.DataFrame:
    """
    Compare coefficients from two models.
    
    Parameters
    ----------
    coef_df1, coef_df2 : pd.DataFrame
        Coefficient DataFrames from two models
    model_names : tuple[str, str]
        Names for the two models
    k : int
        Number of features to compare
    
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame with features from both models
    
    Examples
    --------
    >>> coef1 = extract_linear_coefficients(model1, features)
    >>> coef2 = extract_linear_coefficients(model2, features)
    >>> comparison = compare_coefficients(coef1, coef2, ("baseline", "improved"))
    >>> print(comparison)
    """
    top_1 = top_k_features(coef_df1, k=k).set_index("feature")
    top_2 = top_k_features(coef_df2, k=k).set_index("feature")
    
    # Get union of features
    all_features = sorted(set(top_1.index) | set(top_2.index))
    
    comparison = pd.DataFrame(index=all_features)
    comparison.index.name = "feature"
    
    # Coefficients
    coef_col1 = "coefficient" if "coefficient" in top_1.columns else "mean_coefficient"
    coef_col2 = "coefficient" if "coefficient" in top_2.columns else "mean_coefficient"
    
    comparison[f"{model_names[0]}_coef"] = top_1[coef_col1]
    comparison[f"{model_names[1]}_coef"] = top_2[coef_col2]
    
    # Absolute values
    comparison[f"{model_names[0]}_abs"] = top_1["abs_coefficient"]
    comparison[f"{model_names[1]}_abs"] = top_2["abs_coefficient"]
    
    # Difference
    comparison["abs_diff"] = abs(
        comparison[f"{model_names[0]}_abs"].fillna(0) -
        comparison[f"{model_names[1]}_abs"].fillna(0)
    )
    
    # Sort by difference
    comparison = comparison.sort_values("abs_diff", ascending=False)
    
    return comparison
