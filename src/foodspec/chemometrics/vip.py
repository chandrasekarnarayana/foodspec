"""Variable Importance in Projection (VIP) for PLS models.

This module implements VIP scores for PLS regression and PLS-DA models,
providing interpretability for multivariate calibration models.

References
----------
Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic tool
of chemometrics. Chemometrics and intelligent laboratory systems, 58(2), 109-130.

Mehmood, T., Liland, K. H., Snipen, L., & Sæbø, S. (2012). A review of variable
selection methods in partial least squares regression. Chemometrics and Intelligent
Laboratory Systems, 118, 62-69.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline


def calculate_vip(
    pls_model: Union[PLSRegression, Pipeline],
    X: ArrayLike,
    y: ArrayLike,
) -> np.ndarray:
    """Calculate Variable Importance in Projection (VIP) scores for PLS model.

    VIP scores indicate the importance of each variable in the PLS model.
    Variables with VIP scores > 1 are considered highly influential.
    Variables with VIP scores > 0.8 are considered moderately important.

    Parameters
    ----------
    pls_model : PLSRegression or Pipeline
        Fitted PLS regression model. If Pipeline, the last step must be a PLSRegression.
    X : array-like of shape (n_samples, n_features)
        Training data used to fit the model.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Training targets used to fit the model.

    Returns
    -------
    vip_scores : ndarray of shape (n_features,)
        VIP score for each feature. Scores > 1 indicate high importance.

    Raises
    ------
    ValueError
        If model is not fitted or is not a PLS model.
    TypeError
        If pls_model is not PLSRegression or Pipeline containing PLSRegression.

    Notes
    -----
    VIP is calculated as:

    .. math::
        VIP_j = \\sqrt{p \\cdot \\sum_{a=1}^{A} (SS_a \\cdot w_{aj}^2) / \\sum_{a=1}^{A} SS_a}

    where:
    - p is the number of variables
    - A is the number of PLS components
    - SS_a is the sum of squares explained by component a
    - w_{aj} is the weight of variable j in component a

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from foodspec.chemometrics.vip import calculate_vip
    >>> import numpy as np
    >>>
    >>> # Generate synthetic data
    >>> X = np.random.randn(100, 10)
    >>> y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    >>>
    >>> # Fit PLS model
    >>> pls = PLSRegression(n_components=3)
    >>> pls.fit(X, y)
    >>>
    >>> # Calculate VIP scores
    >>> vip_scores = calculate_vip(pls, X, y)
    >>> print(f"VIP scores shape: {vip_scores.shape}")
    >>> print(f"Important features (VIP > 1): {np.where(vip_scores > 1)[0]}")
    """
    # Extract PLS model from pipeline if necessary
    if isinstance(pls_model, Pipeline):
        # Get the last step (should be PLS)
        pls = pls_model.steps[-1][1]
        if not isinstance(pls, PLSRegression):
            raise TypeError(
                f"Expected Pipeline's last step to be PLSRegression, got {type(pls).__name__}"
            )
    elif isinstance(pls_model, PLSRegression):
        pls = pls_model
    else:
        raise TypeError(
            f"pls_model must be PLSRegression or Pipeline, got {type(pls_model).__name__}"
        )

    # Check if model is fitted
    if not hasattr(pls, "x_weights_"):
        raise ValueError("PLS model must be fitted before calculating VIP scores")

    # Convert inputs to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # Get PLS components and weights
    n_features = X.shape[1]

    # Get the weight matrix (W*)
    # x_weights_ shape: (n_features, n_components)
    W = pls.x_weights_

    # Get the explained variance for each component
    # This is approximated by the sum of squares of the PLS scores (T)
    T = pls.x_scores_  # Shape: (n_samples, n_components)

    # Calculate sum of squares for each component
    # SS_a = sum of squared scores for component a
    SS = np.sum(T ** 2, axis=0)  # Shape: (n_components,)

    # Calculate VIP scores for each variable
    # VIP_j = sqrt(p * sum_a(SS_a * w_ja^2) / sum_a(SS_a))
    W_squared = W ** 2  # Shape: (n_features, n_components)

    # Weighted sum across components
    numerator = np.sum(SS * W_squared, axis=1)  # Shape: (n_features,)
    denominator = np.sum(SS)

    vip_scores = np.sqrt(n_features * numerator / denominator)

    return vip_scores


def calculate_vip_da(
    pls_da_model: Union[Pipeline],
    X: ArrayLike,
    y: ArrayLike,
) -> np.ndarray:
    """Calculate VIP scores for PLS-DA (classification) model.

    For PLS-DA, the PLS model is typically the first step in a pipeline
    followed by a classifier. This function extracts VIP from the PLS component.

    Parameters
    ----------
    pls_da_model : Pipeline
        Fitted PLS-DA pipeline. Must contain a PLSRegression step.
    X : array-like of shape (n_samples, n_features)
        Training data used to fit the model.
    y : array-like of shape (n_samples,)
        Training class labels used to fit the model.

    Returns
    -------
    vip_scores : ndarray of shape (n_features,)
        VIP score for each feature.

    Raises
    ------
    ValueError
        If pipeline doesn't contain PLSRegression step.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> from foodspec.chemometrics.vip import calculate_vip_da
    >>> import numpy as np
    >>>
    >>> # Generate synthetic classification data
    >>> X = np.random.randn(100, 10)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>>
    >>> # Create PLS-DA pipeline
    >>> pls_da = Pipeline([
    ...     ('pls', PLSRegression(n_components=3)),
    ...     ('clf', LogisticRegression())
    ... ])
    >>> pls_da.fit(X, y)
    >>>
    >>> # Calculate VIP scores
    >>> vip_scores = calculate_vip_da(pls_da, X, y)
    """
    # Find PLS step in pipeline
    pls_step = None
    for name, step in pls_da_model.steps:
        if isinstance(step, PLSRegression):
            pls_step = step
            break

    if pls_step is None:
        raise ValueError("Pipeline must contain a PLSRegression step for VIP calculation")

    # For PLS-DA, encode y as dummy variables for the PLS step
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)

    if y_encoded.shape[1] == 1:
        # Binary classification - sklearn returns (n_samples, 1)
        # But PLS expects (n_samples, n_classes)
        y_encoded = np.hstack([1 - y_encoded, y_encoded])

    # Calculate VIP using the PLS step
    return calculate_vip(pls_step, X, y_encoded)


def interpret_vip(vip_scores: np.ndarray, feature_names: list[str] = None) -> dict:
    """Interpret VIP scores and return importance categories.

    Parameters
    ----------
    vip_scores : ndarray of shape (n_features,)
        VIP scores from calculate_vip().
    feature_names : list of str, optional
        Names of features. If None, uses indices.

    Returns
    -------
    interpretation : dict
        Dictionary with keys:
        - 'highly_important': Features with VIP > 1.0
        - 'moderately_important': Features with 0.8 < VIP <= 1.0
        - 'low_importance': Features with VIP <= 0.8
        - 'top_10': Top 10 features by VIP score

    Examples
    --------
    >>> vip_scores = np.array([1.5, 0.9, 0.3, 1.8, 0.7])
    >>> feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
    >>> interpretation = interpret_vip(vip_scores, feature_names)
    >>> print(interpretation['highly_important'])
    [('F4', 1.8), ('F1', 1.5)]
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(vip_scores))]

    # Create (name, score) pairs
    features_with_scores = list(zip(feature_names, vip_scores))

    # Sort by VIP score (descending)
    sorted_features = sorted(features_with_scores, key=lambda x: x[1], reverse=True)

    # Categorize
    highly_important = [(name, score) for name, score in sorted_features if score > 1.0]
    moderately_important = [(name, score) for name, score in sorted_features if 0.8 < score <= 1.0]
    low_importance = [(name, score) for name, score in sorted_features if score <= 0.8]

    return {
        'highly_important': highly_important,
        'moderately_important': moderately_important,
        'low_importance': low_importance,
        'top_10': sorted_features[:10],
        'all_sorted': sorted_features,
    }


__all__ = [
    "calculate_vip",
    "calculate_vip_da",
    "interpret_vip",
]
