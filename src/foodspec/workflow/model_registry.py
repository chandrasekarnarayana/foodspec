"""Model name resolution and registry for workflow orchestration.

Maps model aliases to canonical names and maintains approved model lists.
"""

from __future__ import annotations

from typing import Optional

# Canonical model names (what fit_predict expects)
CANONICAL_MODELS = {
    "logreg",
    "svm_linear",
    "svm_rbf",
    "rf",
    "gboost",
    "xgb",
    "lgbm",
    "knn",
    "mlp",
}

# Model name aliases mapping to canonical names
MODEL_ALIASES = {
    # Logistic Regression
    "logreg": "logreg",
    "logistic": "logreg",
    "logisticregression": "logreg",
    "LogisticRegression": "logreg",
    "lr": "logreg",
    # SVM Linear
    "svm_linear": "svm_linear",
    "svm-linear": "svm_linear",
    "svmlinear": "svm_linear",
    "svc_linear": "svm_linear",
    "linearsvc": "svm_linear",
    "LinearSVC": "svm_linear",
    # SVM RBF
    "svm_rbf": "svm_rbf",
    "svm-rbf": "svm_rbf",
    "svmrbf": "svm_rbf",
    "svc": "svm_rbf",
    "SVC": "svm_rbf",
    # Random Forest
    "rf": "rf",
    "randomforest": "rf",
    "RandomForest": "rf",
    "RandomForestClassifier": "rf",
    # Gradient Boosting
    "gboost": "gboost",
    "gradientboosting": "gboost",
    "GradientBoosting": "gboost",
    "GradientBoostingClassifier": "gboost",
    "gb": "gboost",
    # XGBoost
    "xgb": "xgb",
    "xgboost": "xgb",
    "XGBoost": "xgb",
    "XGBClassifier": "xgb",
    # LightGBM
    "lgbm": "lgbm",
    "lightgbm": "lgbm",
    "LightGBM": "lgbm",
    "LGBMClassifier": "lgbm",
    # K-Nearest Neighbors
    "knn": "knn",
    "KNN": "knn",
    "KNeighborsClassifier": "knn",
    "k-neighbors": "knn",
    # Multi-Layer Perceptron
    "mlp": "mlp",
    "MLP": "mlp",
    "MLPClassifier": "mlp",
    "multilayerperceptron": "mlp",
}

# Regulatory approved models (full names for display)
APPROVED_MODELS = {
    "LogisticRegression",
    "RandomForest",
    "SVC",
    "GradientBoosting",
    "ExtraTreesClassifier",
}

# Mapping from canonical to approved display names
CANONICAL_TO_APPROVED = {
    "logreg": "LogisticRegression",
    "rf": "RandomForest",
    "svm_rbf": "SVC",
    "svm_linear": "SVC",
    "gboost": "GradientBoosting",
    "xgb": "GradientBoosting",  # XGBoost treated as GradientBoosting variant
}


def resolve_model_name(name: Optional[str]) -> Optional[str]:
    """Resolve model name alias to canonical form.

    Parameters
    ----------
    name : Optional[str]
        Model name or alias (can be None)

    Returns
    -------
    Optional[str]
        Canonical model name, or None if input is None

    Raises
    ------
    ValueError
        If name is not recognized

    Examples
    --------
    >>> resolve_model_name("LogisticRegression")
    'logreg'
    >>> resolve_model_name("logreg")
    'logreg'
    >>> resolve_model_name("rf")
    'rf'
    >>> resolve_model_name("RandomForest")
    'rf'
    """
    if name is None:
        return None

    # Normalize: strip whitespace, lowercase for lookup
    normalized = name.strip().lower()

    # Try direct canonical match first
    if normalized in CANONICAL_MODELS:
        return normalized

    # Try alias lookup (case-insensitive via normalized key)
    canonical = MODEL_ALIASES.get(normalized)
    if canonical:
        return canonical

    # Try exact match in aliases (preserves case)
    canonical = MODEL_ALIASES.get(name.strip())
    if canonical:
        return canonical

    # Not found
    raise ValueError(
        f"Unknown model name: '{name}'. "
        f"Supported: {sorted(CANONICAL_MODELS)} or aliases like LogisticRegression, RandomForest, etc."
    )


def is_model_approved(canonical_name: str) -> bool:
    """Check if canonical model name is approved for regulatory use.

    Parameters
    ----------
    canonical_name : str
        Canonical model name (e.g., "logreg", "rf")

    Returns
    -------
    bool
        True if model is approved for regulatory use
    """
    # Map canonical to approved display name
    approved_name = CANONICAL_TO_APPROVED.get(canonical_name)
    if approved_name:
        return approved_name in APPROVED_MODELS

    # Fall back to checking if canonical name is in approved list
    return canonical_name in APPROVED_MODELS


def get_approved_display_name(canonical_name: str) -> str:
    """Get regulatory-approved display name for canonical model.

    Parameters
    ----------
    canonical_name : str
        Canonical model name (e.g., "logreg")

    Returns
    -------
    str
        Approved display name (e.g., "LogisticRegression")
    """
    return CANONICAL_TO_APPROVED.get(canonical_name, canonical_name)


def resolve_scheme_name(scheme: Optional[str]) -> Optional[str]:
    """Resolve cross-validation scheme alias to canonical form.

    Parameters
    ----------
    scheme : Optional[str]
        Scheme name or alias

    Returns
    -------
    Optional[str]
        Canonical scheme name

    Examples
    --------
    >>> resolve_scheme_name("stratified")
    'kfold'
    >>> resolve_scheme_name("k-fold")
    'kfold'
    """
    if scheme is None:
        return None

    normalized = scheme.strip().lower()

    # Canonical schemes supported by fit_predict
    canonical_schemes = {"nested", "kfold", "loso", "lobo"}

    if normalized in canonical_schemes:
        return normalized

    # Aliases
    scheme_aliases = {
        "stratified": "kfold",
        "k-fold": "kfold",
        "k_fold": "kfold",
        "random": "kfold",  # Random CV maps to kfold (with allow_random_cv flag)
        "leave-one-subject-out": "loso",
        "leave-one-batch-out": "lobo",
    }

    canonical = scheme_aliases.get(normalized)
    if canonical:
        return canonical

    raise ValueError(
        f"Unknown validation scheme: '{scheme}'. "
        f"Supported: {sorted(canonical_schemes)} or aliases like 'stratified', 'random', etc."
    )
