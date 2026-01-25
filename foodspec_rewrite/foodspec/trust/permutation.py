"""
Permutation importance for model-agnostic feature importance estimation.

Computes importance by measuring performance decrease when feature values
are randomly shuffled. Works with any model and any metric function.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_repeats: int = 10,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Compute permutation importance for model features.
    
    Importance is computed as the decrease in performance when each feature
    is randomly shuffled. Larger decrease = more important feature.
    
    **Important**: X and y must come from a test/evaluation fold, NOT from
    the training data. Using training data will overestimate importance due
    to overfitting.
    
    Parameters
    ----------
    model : fitted estimator
        Fitted model with .predict() or .predict_proba() method
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (must be from test/eval fold, not training!)
    y : np.ndarray, shape (n_samples,)
        Target labels for computing metric
    metric_fn : callable
        Metric function with signature: metric_fn(y_true, y_pred) -> float
        Higher values should be better (will be inverted if needed).
        Examples: sklearn.metrics.accuracy_score, custom_metric
    n_repeats : int, default=10
        Number of times to shuffle each feature (for variance estimation)
    seed : int or None, default=0
        Random seed for reproducibility. If None, non-deterministic.
    
    Returns
    -------
    pd.DataFrame with columns:
        - feature : str or int
            Feature index or name
        - importance_mean : float
            Mean decrease in metric across repeats
        - importance_std : float
            Standard deviation of decrease across repeats
        - baseline_metric : float
            Baseline metric on unshuffled data (same for all features)
    
    Raises
    ------
    ValueError
        If X and y have mismatched lengths
    
    Notes
    -----
    - Assumes higher metric values are better. Adjust metric_fn if minimizing.
    - Baseline is computed once; each feature is shuffled n_repeats times.
    - Shuffling uses random sampling without replacement (permutation).
    - Can be slow for large datasets; subsample if needed.
    
    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import accuracy_score
    >>> import numpy as np
    >>> 
    >>> # Generate test data from eval fold (not training!)
    >>> X_test = np.random.randn(100, 5)
    >>> y_test = np.random.randint(0, 2, 100)
    >>> 
    >>> # Train model (on separate training data)
    >>> X_train = np.random.randn(200, 5)
    >>> y_train = np.random.randint(0, 2, 200)
    >>> model = LogisticRegression(random_state=42)
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Compute permutation importance
    >>> importance = permutation_importance(
    ...     model, X_test, y_test,
    ...     metric_fn=accuracy_score,
    ...     n_repeats=10,
    ...     seed=42,
    ... )
    >>> print(importance.sort_values("importance_mean", ascending=False))
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have mismatched lengths: {X.shape[0]} vs {y.shape[0]}"
        )
    
    n_samples, n_features = X.shape
    rng = np.random.RandomState(seed)
    
    # Compute baseline metric on unshuffled data
    y_pred_baseline = model.predict(X)
    baseline_metric = metric_fn(y, y_pred_baseline)
    
    # For each feature, compute importance
    importances = []
    feature_indices = range(n_features)
    
    for feature_idx in feature_indices:
        # Shuffle this feature n_repeats times
        decreases = []
        
        for _ in range(n_repeats):
            # Create copy and shuffle feature
            X_shuffled = X.copy()
            X_shuffled[:, feature_idx] = rng.permutation(X_shuffled[:, feature_idx])
            
            # Get predictions on shuffled data
            y_pred_shuffled = model.predict(X_shuffled)
            
            # Compute metric and importance
            metric_shuffled = metric_fn(y, y_pred_shuffled)
            decrease = baseline_metric - metric_shuffled
            decreases.append(decrease)
        
        # Store mean and std
        importances.append({
            "feature": feature_idx,
            "importance_mean": np.mean(decreases),
            "importance_std": np.std(decreases),
            "baseline_metric": baseline_metric,
        })
    
    df = pd.DataFrame(importances)
    # Sort by importance (descending)
    df = df.sort_values("importance_mean", ascending=False, ignore_index=True)
    
    return df


def permutation_importance_with_names(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_repeats: int = 10,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Permutation importance with feature names.
    
    Wrapper around permutation_importance that replaces feature indices with names.
    
    Parameters
    ----------
    model : fitted estimator
        Fitted model
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    feature_names : list[str]
        Feature names corresponding to columns
    metric_fn : callable
        Metric function
    n_repeats : int
        Number of repeats per feature
    seed : int or None
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Importance DataFrame with feature names instead of indices
    
    Raises
    ------
    ValueError
        If feature_names length doesn't match X.shape[1]
    
    Examples
    --------
    >>> importance = permutation_importance_with_names(
    ...     model, X_test, y_test,
    ...     feature_names=["f0", "f1", "f2", "f3", "f4"],
    ...     metric_fn=accuracy_score,
    ...     n_repeats=10,
    ... )
    """
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) doesn't match "
            f"X.shape[1] ({X.shape[1]})"
        )
    
    df = permutation_importance(
        model, X, y, metric_fn, n_repeats=n_repeats, seed=seed
    )
    
    # Replace feature indices with names
    df["feature"] = df["feature"].map(lambda idx: feature_names[idx])
    
    return df


def top_k_important_features(
    importance_df: pd.DataFrame,
    k: int = 20,
) -> pd.DataFrame:
    """
    Select top-k most important features from importance DataFrame.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Output from permutation_importance() or similar
    k : int, default=20
        Number of top features to return
    
    Returns
    -------
    pd.DataFrame
        Top k rows
    
    Examples
    --------
    >>> importance = permutation_importance(model, X_test, y_test, metric_fn)
    >>> top_10 = top_k_important_features(importance, k=10)
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    
    return importance_df.head(k).reset_index(drop=True)


def compare_importances(
    importance_df1: pd.DataFrame,
    importance_df2: pd.DataFrame,
    model_names: tuple[str, str] = ("Model 1", "Model 2"),
) -> pd.DataFrame:
    """
    Compare permutation importances from two models.
    
    Parameters
    ----------
    importance_df1, importance_df2 : pd.DataFrame
        Importance DataFrames from two models
    model_names : tuple[str, str]
        Names for the two models
    
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame
    
    Examples
    --------
    >>> imp1 = permutation_importance(model1, X_test, y_test, metric_fn)
    >>> imp2 = permutation_importance(model2, X_test, y_test, metric_fn)
    >>> comparison = compare_importances(imp1, imp2, ("baseline", "improved"))
    """
    # Create DataFrames indexed by feature
    df1 = importance_df1.set_index("feature")
    df2 = importance_df2.set_index("feature")
    
    # Get union of features
    all_features = sorted(set(df1.index) | set(df2.index))
    
    comparison = pd.DataFrame(index=all_features)
    comparison.index.name = "feature"
    
    comparison[f"{model_names[0]}_mean"] = df1["importance_mean"]
    comparison[f"{model_names[1]}_mean"] = df2["importance_mean"]
    
    comparison[f"{model_names[0]}_std"] = df1["importance_std"]
    comparison[f"{model_names[1]}_std"] = df2["importance_std"]
    
    # Difference
    comparison["diff_mean"] = (
        comparison[f"{model_names[1]}_mean"].fillna(0) -
        comparison[f"{model_names[0]}_mean"].fillna(0)
    )
    
    # Sort by absolute difference
    comparison["abs_diff"] = abs(comparison["diff_mean"])
    comparison = comparison.sort_values("abs_diff", ascending=False)
    
    return comparison
