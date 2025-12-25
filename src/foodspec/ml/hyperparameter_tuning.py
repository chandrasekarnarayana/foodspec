"""
Automated hyperparameter tuning for ML models.

Provides grid search, Optuna-based Bayesian optimization, and domain-specific search spaces
for spectroscopy models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


def get_search_space_classifier(model_name: str) -> Dict[str, List[Any]]:
    """Get domain-specific hyperparameter search space for classifiers.

    Parameters
    ----------
    model_name : str
        One of: "rf", "svm_rbf", "gboost", "mlp", "knn", "logreg".

    Returns
    -------
    dict
        Parameter grid suitable for GridSearchCV.
    """
    if model_name == "rf":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    elif model_name == "svm_rbf":
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        }
    elif model_name == "gboost":
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        }
    elif model_name == "mlp":
        return {
            "hidden_layer_sizes": [(100,), (100, 50), (200, 100), (200, 100, 50)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01, 0.1],
        }
    elif model_name == "knn":
        return {
            "n_neighbors": [3, 5, 7, 10, 15],
            "weights": ["uniform", "distance"],
        }
    elif model_name == "logreg":
        return {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_search_space_regressor(model_name: str) -> Dict[str, List[Any]]:
    """Get hyperparameter search space for regressors.

    Parameters
    ----------
    model_name : str
        One of: "rf_reg", "svr", "mlp_reg", "ridge", "lasso".

    Returns
    -------
    dict
        Parameter grid.
    """
    if model_name == "rf_reg":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
    elif model_name == "svr":
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.001, 0.01],
            "kernel": ["rbf", "linear"],
        }
    elif model_name == "mlp_reg":
        return {
            "hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        }
    elif model_name == "ridge":
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        }
    elif model_name == "lasso":
        return {
            "alpha": [0.0001, 0.001, 0.01, 0.1],
        }
    else:
        raise ValueError(f"Unknown regressor: {model_name}")


def grid_search_classifier(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    cv: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """Run grid search for classifier hyperparameters.

    Parameters
    ----------
    pipeline : Pipeline
        sklearn Pipeline with fitted preprocessor(s) and unfitted model.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    model_name : str
        Model name (e.g., "rf", "svm_rbf").
    cv : int, default=5
        Cross-validation folds.
    scoring : str, default="f1_weighted"
        Scoring metric.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    best_model : estimator
        Fitted model with best hyperparameters.
    results : dict
        - 'best_params': best hyperparameters
        - 'best_score': best cross-validation score
        - 'cv_results': full GridSearchCV results
    """
    param_grid = get_search_space_classifier(model_name)

    # Prepend step name to param keys
    param_grid = {f"model__{k}": v for k, v in param_grid.items()}

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    return gs.best_estimator_, {
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
        "cv_results": gs.cv_results_,
    }


def grid_search_regressor(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """Run grid search for regressor hyperparameters.

    Parameters
    ----------
    pipeline : Pipeline
        sklearn Pipeline.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    model_name : str
        Model name.
    cv : int, default=5
        Cross-validation folds.
    scoring : str, default="neg_mean_squared_error"
        Scoring metric.
    n_jobs : int, default=-1
        Parallel jobs.

    Returns
    -------
    best_model : estimator
        Fitted regressor.
    results : dict
        Tuning results.
    """
    param_grid = get_search_space_regressor(model_name)
    param_grid = {f"model__{k}": v for k, v in param_grid.items()}

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    return gs.best_estimator_, {
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
        "cv_results": gs.cv_results_,
    }


def quick_tune_classifier(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    cv: int = 3,
) -> Any:
    """Quick hyperparameter tuning for rapid iteration.

    Uses RandomizedSearchCV with smaller candidate set.

    Parameters
    ----------
    pipeline : Pipeline
        sklearn Pipeline.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    model_name : str
        Model name.
    cv : int, default=3
        CV folds.

    Returns
    -------
    best_model : estimator
        Fitted model.
    """
    param_grid = get_search_space_classifier(model_name)
    param_grid = {f"model__{k}": v for k, v in param_grid.items()}

    rs = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=10,
        cv=cv,
        random_state=0,
        n_jobs=-1,
        verbose=0,
    )
    rs.fit(X_train, y_train)
    return rs.best_estimator_


try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def bayesian_tune_classifier(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    n_trials: int = 20,
    cv: int = 3,
) -> Tuple[Any, Dict[str, Any]]:
    """Bayesian hyperparameter optimization using Optuna (if available).

    Parameters
    ----------
    pipeline : Pipeline
        sklearn Pipeline.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    model_name : str
        Model name.
    n_trials : int, default=20
        Number of Optuna trials.
    cv : int, default=3
        CV folds.

    Returns
    -------
    best_model : estimator
        Best fitted model.
    results : dict
        - 'best_params': best parameters
        - 'best_value': best objective value
        - 'best_trial': best trial info
    """
    if not HAS_OPTUNA:
        raise ImportError("optuna is required for Bayesian tuning; install via `pip install optuna`")

    param_space = get_search_space_classifier(model_name)

    def objective(trial):
        # Map trial suggestions to param_space values
        params = {}
        for key, values in param_space.items():
            if isinstance(values[0], int):
                params[key] = trial.suggest_int(key, min(values), max(values))
            elif isinstance(values[0], float):
                params[key] = trial.suggest_float(key, min(values), max(values))
            else:
                params[key] = trial.suggest_categorical(key, values)

        # Clone and fit pipeline
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score

        pipe_clone = clone(pipeline)
        # Update model params
        pipe_clone.named_steps["model"].set_params(**params)

        # Cross-validate
        scores = cross_val_score(pipe_clone, X_train, y_train, cv=cv, scoring="f1_weighted")
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    best_model = pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})
    best_model.fit(X_train, y_train)

    return best_model, {
        "best_params": best_params,
        "best_value": float(study.best_value),
        "best_trial": study.best_trial.number,
    }


__all__ = [
    "get_search_space_classifier",
    "get_search_space_regressor",
    "grid_search_classifier",
    "grid_search_regressor",
    "quick_tune_classifier",
    "bayesian_tune_classifier",
]
