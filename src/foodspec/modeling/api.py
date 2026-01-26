"""Modeling API: unified training + validation entry point."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from foodspec.chemometrics.models import make_classifier, make_pls_da
from foodspec.core.errors import FoodSpecValidationError
from foodspec.modeling.validation.metrics import (
    bootstrap_classification_ci,
    classification_metrics_bundle,
)
from foodspec.modeling.validation.strategies import leave_one_group_out
from foodspec.modeling.metrics_regression import (
    count_metrics,
    regression_metrics,
    residual_diagnostics,
    overdispersion_summary,
)
from foodspec.modeling.models_regression import REGRESSION_REGISTRY, build_regression_model
from foodspec.modeling.outcome import OutcomeType


@dataclass
class FitPredictResult:
    """Result bundle for fit_predict."""

    model: BaseEstimator
    folds: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    metrics_ci: Dict[str, Any]
    per_class: Dict[str, Any]
    confusion_matrix: Dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    groups: Optional[np.ndarray] = None
    classes: Sequence[Any] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    outcome_type: OutcomeType = OutcomeType.CLASSIFICATION
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_")


def _default_param_grid(model_name: str, outcome_type: OutcomeType = OutcomeType.CLASSIFICATION) -> Dict[str, Iterable[Any]]:
    name = _normalize_model_name(model_name)
    if outcome_type == OutcomeType.CLASSIFICATION:
        if name in {"logreg", "logistic_regression"}:
            return {"model__C": [0.1, 1.0, 10.0]}
        if name in {"svm_linear"}:
            return {"model__C": [0.1, 1.0, 10.0]}
        if name in {"svm_rbf"}:
            return {"model__C": [1.0, 10.0], "model__gamma": ["scale", "auto"]}
        if name in {"rf", "random_forest"}:
            return {"model__n_estimators": [200], "model__max_depth": [None, 10]}
        if name in {"pls_da", "plsda"}:
            return {"pls_proj__n_components": [2, 5, 10], "clf__C": [0.1, 1.0, 10.0]}
        if name in {"xgb", "xgboost"}:
            return {"model__n_estimators": [200], "model__max_depth": [3, 6]}
        if name in {"lgbm", "lightgbm"}:
            return {"model__n_estimators": [200], "model__num_leaves": [31, 63]}
        return {}

    # Regression / count defaults (lightweight search only)
    if name in {"ridge"}:
        return {"alpha": [0.1, 1.0, 10.0]}
    if name in {"pcr"}:
        return {"pca__n_components": [2, 5, 10]}
    if name in {"poisson", "neg_binom"}:
        return {"alpha": [0.0, 0.1, 1.0]}
    return {}


def _build_estimator(model_name: str, *, outcome_type: OutcomeType = OutcomeType.CLASSIFICATION, random_state: int = 0) -> BaseEstimator:
    name = _normalize_model_name(model_name)

    if outcome_type == OutcomeType.CLASSIFICATION:
        if name in {"pls_da", "plsda"}:
            return make_pls_da()

        model_kwargs: Dict[str, Any] = {}
        if name in {"logreg", "logistic_regression"}:
            model_name = "logreg"
            model_kwargs["random_state"] = random_state
        elif name in {"svm_linear"}:
            model_name = "svm_linear"
        elif name in {"svm_rbf"}:
            model_name = "svm_rbf"
        elif name in {"rf", "random_forest"}:
            model_name = "rf"
            model_kwargs["random_state"] = random_state
        elif name in {"xgb", "xgboost"}:
            model_name = "xgb"
            model_kwargs["random_state"] = random_state
        elif name in {"lgbm", "lightgbm"}:
            model_name = "lgbm"
            model_kwargs["random_state"] = random_state
        else:
            model_name = name

        estimator = make_classifier(model_name, **model_kwargs)

        if name in {"logreg", "logistic_regression", "svm_linear", "svm_rbf"}:
            return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        if name in {"rf", "random_forest", "xgb", "xgboost", "lgbm", "lightgbm"}:
            return Pipeline([("model", estimator)])
        return estimator

    # Regression / count shelf
    key = name
    if key not in REGRESSION_REGISTRY:
        raise FoodSpecValidationError(f"Model '{model_name}' is not registered for regression/count tasks.")
    entry = REGRESSION_REGISTRY[key]
    if outcome_type == OutcomeType.COUNT and entry.get("type") != "count":
        raise FoodSpecValidationError(f"Model '{model_name}' is not suitable for count outcomes.")
    return build_regression_model(key)


def _remap_bundle(bundle: Dict[str, Any], class_labels: Sequence[Any]) -> Dict[str, Any]:
    label_map = {str(i): str(class_labels[i]) for i in range(len(class_labels))}
    mapped = dict(bundle)
    confusion = dict(bundle.get("confusion_matrix", {}))
    if "labels" in confusion:
        confusion["labels"] = [label_map.get(str(l), str(l)) for l in confusion["labels"]]
        mapped["confusion_matrix"] = confusion
    per_class = bundle.get("per_class")
    if isinstance(per_class, dict):
        mapped["per_class"] = {label_map.get(str(k), str(k)): v for k, v in per_class.items()}
    return mapped


def _generate_splits(
    scheme: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    outer_splits: int,
    seed: int,
    *,
    allow_random: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    scheme = scheme.lower()
    if scheme in {"nested", "nested_cv"}:
        if groups is None:
            splitter = StratifiedKFold(
                n_splits=min(outer_splits, len(np.unique(y))),
                shuffle=True,
                random_state=seed,
            )
            return list(splitter.split(X, y))
        try:
            splitter = StratifiedGroupKFold(
                n_splits=min(outer_splits, len(np.unique(groups))),
                shuffle=True,
                random_state=seed,
            )
            return list(splitter.split(X, y, groups))
        except Exception:
            splitter = StratifiedKFold(
                n_splits=min(outer_splits, len(np.unique(y))),
                shuffle=True,
                random_state=seed,
            )
            return list(splitter.split(X, y))

    if scheme in {"lobo", "loso", "leave_one_group_out", "leave_one_batch_out", "leave_one_stage_out"}:
        if groups is None:
            raise FoodSpecValidationError("Group column required for LOBO/LOSO schemes.")
        if len(np.unique(groups)) < 2:
            raise FoodSpecValidationError("LOBO/LOSO requires at least two groups.")
        return list(leave_one_group_out(groups))

    if scheme in {"random", "kfold"}:
        if not allow_random:
            raise FoodSpecValidationError(
                "Random CV is blocked for food data. Use --unsafe-random-cv to override."
            )
        splitter = StratifiedKFold(
            n_splits=min(outer_splits, len(np.unique(y))),
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(X, y))

    raise FoodSpecValidationError(f"Unknown validation scheme '{scheme}'.")


def _generate_regression_splits(
    scheme: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    outer_splits: int,
    seed: int,
    *,
    allow_random: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    scheme = scheme.lower()
    n_samples = len(y)

    def _kfold_splits() -> List[Tuple[np.ndarray, np.ndarray]]:
        if groups is not None:
            unique_groups = np.unique(groups)
            if len(unique_groups) < 2:
                raise FoodSpecValidationError("Group-aware CV requires at least two groups.")
            splitter = GroupKFold(n_splits=min(outer_splits, len(unique_groups)))
            return list(splitter.split(X, y, groups))
        splitter = KFold(n_splits=min(outer_splits, n_samples), shuffle=True, random_state=seed)
        return list(splitter.split(X, y))

    if scheme in {"nested", "nested_cv", "kfold"}:
        return _kfold_splits()

    if scheme in {"random"}:
        if not allow_random:
            raise FoodSpecValidationError("Random CV is blocked; pass allow_random_cv=True to override.")
        return _kfold_splits()

    if scheme in {"lobo", "loso", "leave_one_group_out", "leave_one_batch_out", "leave_one_stage_out"}:
        if groups is None:
            raise FoodSpecValidationError("Group column required for LOBO/LOSO schemes.")
        if len(np.unique(groups)) < 2:
            raise FoodSpecValidationError("LOBO/LOSO requires at least two groups.")
        return list(leave_one_group_out(groups))

    raise FoodSpecValidationError(f"Unknown validation scheme '{scheme}' for regression/count tasks.")


def _fit_with_grid(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    param_grid: Dict[str, Iterable[Any]],
    inner_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> tuple[BaseEstimator, Dict[str, Any]]:
    if not param_grid:
        model = clone(estimator)
        model.fit(X, y)
        return model, {}
    if groups is None:
        cv = StratifiedKFold(
            n_splits=min(inner_splits, len(np.unique(y))),
            shuffle=True,
            random_state=seed,
        )
    else:
        try:
            cv = StratifiedGroupKFold(
                n_splits=min(inner_splits, len(np.unique(groups))),
                shuffle=True,
                random_state=seed,
            )
        except Exception:
            cv = StratifiedKFold(
                n_splits=min(inner_splits, len(np.unique(y))),
                shuffle=True,
                random_state=seed,
            )
    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=None,
    )
    if groups is None:
        search.fit(X, y)
    else:
        search.fit(X, y, groups=groups)
    return search.best_estimator_, dict(search.best_params_)


def _fit_with_grid_regression(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    param_grid: Dict[str, Iterable[Any]],
    inner_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
    scoring: str = "neg_root_mean_squared_error",
) -> tuple[BaseEstimator, Dict[str, Any]]:
    if not param_grid:
        model = clone(estimator)
        model.fit(X, y)
        return model, {}

    if groups is None:
        cv = KFold(n_splits=min(inner_splits, len(y)), shuffle=True, random_state=seed)
    else:
        cv = GroupKFold(n_splits=min(inner_splits, len(np.unique(groups))))

    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=None,
    )
    if groups is None:
        search.fit(X, y)
    else:
        search.fit(X, y, groups=groups)
    return search.best_estimator_, dict(search.best_params_)


def _bootstrap_regression_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    outcome_type: OutcomeType = OutcomeType.REGRESSION,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics_list: Dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt = y_true[idx]
        yp = y_pred[idx]
        if outcome_type == OutcomeType.COUNT:
            bundle = count_metrics(yt, yp)
        else:
            bundle = regression_metrics(yt, yp)
        for key, val in bundle.items():
            metrics_list.setdefault(key, []).append(float(val))

    ci: Dict[str, Dict[str, float]] = {}
    for key, values in metrics_list.items():
        dist = np.asarray(values, dtype=float)
        ci[key] = {
            "mean": float(np.mean(dist)),
            "ci_low": float(np.quantile(dist, alpha / 2.0)),
            "ci_high": float(np.quantile(dist, 1.0 - alpha / 2.0)),
        }
    return ci


def _regression_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    outcome_type: OutcomeType,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if outcome_type == OutcomeType.COUNT:
        metrics_out = count_metrics(y_true, y_pred)
        ratio, mean_count = overdispersion_summary(y_true)
        diag = residual_diagnostics(y_true, y_pred)
        diag.update({"overdispersion_ratio": ratio, "mean_count": mean_count})
    else:
        metrics_out = regression_metrics(y_true, y_pred)
        diag = residual_diagnostics(y_true, y_pred)
    return metrics_out, diag


def fit_predict(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str,
    scheme: str = "nested",
    groups: Optional[np.ndarray] = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 0,
    allow_random_cv: bool = False,
    param_grid: Optional[Dict[str, Iterable[Any]]] = None,
    outcome_type: OutcomeType | str = OutcomeType.CLASSIFICATION,
    embedding: Optional[Dict[str, Any]] = None,
) -> FitPredictResult:
    """Fit a model and run validation according to the selected scheme."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise FoodSpecValidationError("X must be a 2D feature matrix.")
    if len(y) != X.shape[0]:
        raise FoodSpecValidationError("y length must match X rows.")
    if groups is not None and len(groups) != X.shape[0]:
        raise FoodSpecValidationError("groups length must match X rows.")

    outcome_enum = OutcomeType(outcome_type) if isinstance(outcome_type, str) else outcome_type
    if outcome_enum == OutcomeType.COUNT and (y < 0).any():
        raise FoodSpecValidationError("Count outcomes must be non-negative.")

    # Optional embedding (e.g., PCA) applied fold-wise to avoid leakage
    embed_method = None
    embed_params: Dict[str, Any] = {}
    embed_info: Dict[str, Any] = {}
    embed_builder = None
    if embedding:
        embed_method = embedding.get("method")
        embed_params = dict(embedding.get("params", {}))
        embed_seed = embedding.get("seed", seed)
        if embed_seed is not None and "random_state" not in embed_params:
            embed_params["random_state"] = embed_seed
        if embed_method:
            def _build_embed_component():
                from foodspec.multivariate import build_component

                comp = build_component(embed_method, **embed_params)
                if getattr(comp, "requires_second_view", False):
                    raise FoodSpecValidationError(
                        "Embedding method requires paired views, which is not supported in the modeling pipeline."
                    )
                return comp

            embed_builder = _build_embed_component
            embed_info = {"method": embed_method, "params": embed_params, "n_features_in": int(X.shape[1])}

    # --- Classification path -------------------------------------------------
    if outcome_enum == OutcomeType.CLASSIFICATION:
        if len(np.unique(y)) < 2:
            raise FoodSpecValidationError("At least two classes are required for classification.")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        estimator = _build_estimator(model_name, outcome_type=outcome_enum, random_state=seed)
        param_grid = param_grid if param_grid is not None else _default_param_grid(model_name, outcome_enum)

        splits = _generate_splits(
            scheme,
            X,
            y_encoded,
            groups,
            outer_splits,
            seed,
            allow_random=allow_random_cv,
        )

        folds: List[Dict[str, Any]] = []
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        y_proba_all: List[np.ndarray] = []
        group_all: List[np.ndarray] = []
        best_params: Dict[str, Any] = {}

        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            if embed_builder is not None:
                embed_comp = embed_builder()
                embed_comp.fit(X_train, y_train if getattr(embed_comp, "requires_y", False) else None)
                X_train = embed_comp.transform(X_train)
                X_test = embed_comp.transform(X_test)
                embed_info.setdefault("n_components", int(X_train.shape[1]))
            model, fold_params = _fit_with_grid(
                estimator,
                X_train,
                y_train,
                param_grid=param_grid,
                inner_splits=inner_splits,
                seed=seed + fold_idx,
                groups=groups[train_idx] if groups is not None else None,
            )
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            bundle = classification_metrics_bundle(y_test, y_pred, y_proba)
            bundle = _remap_bundle(bundle, label_encoder.classes_)
            fold_entry: Dict[str, Any] = {
                "fold": fold_idx,
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "metrics": bundle["metrics"],
                "confusion_matrix": bundle["confusion_matrix"],
            }
            if fold_params:
                fold_entry["best_params"] = fold_params
                best_params = fold_params
            if groups is not None:
                fold_entry["test_groups"] = list(np.unique(groups[test_idx]).astype(str))
                group_all.append(groups[test_idx])
            folds.append(fold_entry)
            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            if y_proba is not None:
                y_proba_all.append(np.asarray(y_proba))

        y_true_all_arr = np.concatenate(y_true_all, axis=0)
        y_pred_all_arr = np.concatenate(y_pred_all, axis=0)
        y_proba_all_arr = np.concatenate(y_proba_all, axis=0) if y_proba_all else None
        group_all_arr = np.concatenate(group_all, axis=0) if group_all else None

        bundle = classification_metrics_bundle(y_true_all_arr, y_pred_all_arr, y_proba_all_arr)
        bundle = _remap_bundle(bundle, label_encoder.classes_)
        metrics_ci = bootstrap_classification_ci(
            y_true_all_arr,
            y_pred_all_arr,
            y_proba_all_arr,
            n_bootstrap=200,
            seed=seed,
        )

        X_fit = X
        if embed_builder is not None:
            embed_final = embed_builder()
            embed_final.fit(X, y_encoded if getattr(embed_final, "requires_y", False) else None)
            X_fit = embed_final.transform(X)
            embed_info.setdefault("n_components", int(X_fit.shape[1]))

        final_model, final_params = _fit_with_grid(
            estimator,
            X_fit,
            y_encoded,
            param_grid=param_grid,
            inner_splits=inner_splits,
            seed=seed,
            groups=groups,
        )
        if final_params:
            best_params = final_params

        return FitPredictResult(
            model=final_model,
            folds=folds,
            metrics=bundle["metrics"],
            metrics_ci=metrics_ci,
            per_class=bundle["per_class"],
            confusion_matrix=bundle["confusion_matrix"],
            y_true=y_true_all_arr,
            y_pred=y_pred_all_arr,
            y_proba=y_proba_all_arr,
            groups=group_all_arr,
            classes=list(label_encoder.classes_),
            best_params=best_params,
            outcome_type=outcome_enum,
            diagnostics={"embedding": embed_info} if embed_info else {},
        )

    # --- Regression / count path -------------------------------------------
    estimator = _build_estimator(model_name, outcome_type=outcome_enum, random_state=seed)
    param_grid = param_grid if param_grid is not None else _default_param_grid(model_name, outcome_enum)
    scoring = "neg_mean_poisson_deviance" if outcome_enum == OutcomeType.COUNT else "neg_root_mean_squared_error"

    splits = _generate_regression_splits(
        scheme,
        X,
        y,
        groups,
        outer_splits,
        seed,
        allow_random=allow_random_cv,
    )

    folds: List[Dict[str, Any]] = []
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    group_all: List[np.ndarray] = []
    best_params: Dict[str, Any] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if embed_builder is not None:
            embed_comp = embed_builder()
            embed_comp.fit(X_train, y_train if getattr(embed_comp, "requires_y", False) else None)
            X_train = embed_comp.transform(X_train)
            X_test = embed_comp.transform(X_test)
            embed_info.setdefault("n_components", int(X_train.shape[1]))
        model, fold_params = _fit_with_grid_regression(
            estimator,
            X_train,
            y_train,
            param_grid=param_grid,
            inner_splits=inner_splits,
            seed=seed + fold_idx,
            groups=groups[train_idx] if groups is not None else None,
            scoring=scoring,
        )
        y_pred = model.predict(X_test)
        metrics_out, diag_out = _regression_bundle(y_test, y_pred, outcome_type=outcome_enum)
        fold_entry: Dict[str, Any] = {
            "fold": fold_idx,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "metrics": metrics_out,
            "diagnostics": diag_out,
        }
        if fold_params:
            fold_entry["best_params"] = fold_params
            best_params = fold_params
        if groups is not None:
            fold_entry["test_groups"] = list(np.unique(groups[test_idx]).astype(str))
            group_all.append(groups[test_idx])
        folds.append(fold_entry)
        y_true_all.append(np.asarray(y_test))
        y_pred_all.append(np.asarray(y_pred))

    y_true_all_arr = np.concatenate(y_true_all, axis=0)
    y_pred_all_arr = np.concatenate(y_pred_all, axis=0)
    group_all_arr = np.concatenate(group_all, axis=0) if group_all else None

    metrics_out, diag_out = _regression_bundle(y_true_all_arr, y_pred_all_arr, outcome_type=outcome_enum)
    metrics_ci = _bootstrap_regression_ci(
        y_true_all_arr,
        y_pred_all_arr,
        n_bootstrap=200,
        seed=seed,
        outcome_type=outcome_enum,
    )

    X_fit = X
    if embed_builder is not None:
        embed_final = embed_builder()
        embed_final.fit(X, y if getattr(embed_final, "requires_y", False) else None)
        X_fit = embed_final.transform(X)
        embed_info.setdefault("n_components", int(X_fit.shape[1]))

    final_model, final_params = _fit_with_grid_regression(
        estimator,
        X_fit,
        y,
        param_grid=param_grid,
        inner_splits=inner_splits,
        seed=seed,
        groups=groups,
        scoring=scoring,
    )
    if final_params:
        best_params = final_params

    if embed_info:
        diag_out = dict(diag_out)
        diag_out["embedding"] = embed_info

    return FitPredictResult(
        model=final_model,
        folds=folds,
        metrics=metrics_out,
        metrics_ci=metrics_ci,
        per_class={},
        confusion_matrix={},
        y_true=y_true_all_arr,
        y_pred=y_pred_all_arr,
        y_proba=None,
        groups=group_all_arr,
        classes=[],
        best_params=best_params,
        outcome_type=outcome_enum,
        diagnostics=diag_out,
    )


def metrics_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    groups: Sequence[Any],
    *,
    class_labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Compute metrics per group."""
    groups = np.asarray(groups)
    results: Dict[str, Any] = {}
    for group in np.unique(groups):
        mask = groups == group
        bundle = classification_metrics_bundle(
            y_true[mask],
            y_pred[mask],
            y_proba[mask] if y_proba is not None else None,
        )
        if class_labels:
            bundle = _remap_bundle(bundle, class_labels)
        results[str(group)] = {
            "n_samples": int(mask.sum()),
            "metrics": bundle["metrics"],
            "confusion_matrix": bundle["confusion_matrix"],
            "per_class": bundle["per_class"],
        }
    return results


__all__ = ["FitPredictResult", "fit_predict", "metrics_by_group"]
