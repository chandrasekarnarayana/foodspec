"""Model lifecycle utilities: train → freeze → predict.

TrainablePipeline wraps preprocessing + feature construction (peaks/ratios) +
an ML model (RF or LogisticRegression) and can produce a FrozenModel that
embeds all transforms and metadata for reproducible prediction.

Deprecated shim:
    Use foodspec.ml.ModelLifecycle instead. This module will be removed in v2.0.0.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from foodspec.features.rq import PeakDefinition, RatioDefinition
from foodspec.preprocessing_pipeline import PreprocessingConfig, detect_input_mode, run_full_preprocessing
from foodspec.registry import FeatureModelRegistry

warnings.warn(
    "foodspec.model_lifecycle is deprecated and will be removed in v2.0.0. "
    "Use foodspec.ml.ModelLifecycle instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


def _compute_ratios(df: pd.DataFrame, ratios: List[RatioDefinition]) -> pd.DataFrame:
    df_out = df.copy()
    for r in ratios:
        if r.numerator not in df_out.columns or r.denominator not in df_out.columns:
            df_out[r.name] = np.nan
            continue
        num = df_out[r.numerator].astype(float)
        den = df_out[r.denominator].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            df_out[r.name] = num / den
    return df_out


@dataclass
class TrainablePipeline:
    label_col: str
    peaks: List[PeakDefinition]
    ratios: List[RatioDefinition]
    preprocess_config: PreprocessingConfig
    model_type: str = "rf"  # rf | lr
    model_params: Dict[str, Any] = None

    def train(self, df: pd.DataFrame) -> "FrozenModel":
        # Preprocess if raw spectra
        mode = detect_input_mode(df)
        if mode == "raw_spectra":
            df_proc = run_full_preprocessing(df, self.preprocess_config)
        else:
            df_proc = df.copy()
        # Construct ratios
        df_feat = _compute_ratios(df_proc, self.ratios)
        feature_cols = [p.column for p in self.peaks if p.column in df_feat.columns] + [r.name for r in self.ratios]
        X = df_feat[feature_cols].astype(float)
        y = df_feat[self.label_col].astype(str)
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)

        if self.model_type == "rf":
            params = self.model_params or {"n_estimators": 300, "random_state": 0}
            model = RandomForestClassifier(**params)
        else:
            params = self.model_params or {"max_iter": 2000, "penalty": "l2"}
            model = LogisticRegression(**params)
        model.fit(Xz, y)
        return FrozenModel(
            preprocess_config=self.preprocess_config,
            peaks=self.peaks,
            ratios=self.ratios,
            feature_names=feature_cols,
            label_col=self.label_col,
            model=model,
            scaler=scaler,
            metadata={
                "model_type": self.model_type,
                "classes": list(model.classes_),
                "trained_features": feature_cols,
            },
        )


@dataclass
class FrozenModel:
    preprocess_config: PreprocessingConfig
    peaks: List[PeakDefinition]
    ratios: List[RatioDefinition]
    feature_names: List[str]
    label_col: str
    model: Any
    scaler: StandardScaler
    metadata: Dict[str, Any]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = detect_input_mode(df)
        if mode == "raw_spectra":
            df_proc = run_full_preprocessing(df, self.preprocess_config)
        else:
            df_proc = df.copy()
        df_feat = _compute_ratios(df_proc, self.ratios)
        # Ensure all feature columns exist
        missing = [col for col in self.feature_names if col not in df_feat.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Input is missing expected features: {missing_str}. "
                "Ensure preprocessing/ratio definitions match the frozen model."
            )
        for col in self.feature_names:
            if col not in df_feat.columns:
                df_feat[col] = np.nan
        X = df_feat[self.feature_names].astype(float)
        mask = ~X.isna().any(axis=1)
        Xz = self.scaler.transform(X.fillna(0))
        preds = self.model.predict(Xz)
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xz)
        out = df.copy()
        out["prediction"] = preds
        if proba is not None:
            for i, cls in enumerate(self.model.classes_):
                out[f"proba_{cls}"] = proba[:, i]
        out["mask_valid"] = mask.values
        return out

    def save(self, path: Path):
        path = Path(path)
        payload = {
            "preprocess_config": asdict(self.preprocess_config),
            "peaks": [asdict(p) for p in self.peaks],
            "ratios": [asdict(r) for r in self.ratios],
            "feature_names": self.feature_names,
            "label_col": self.label_col,
            "metadata": self.metadata,
        }
        (path.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (path.with_suffix(".pkl")).open("wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

        # Optional registry logging
        reg_path = os.environ.get("FOODSPEC_REGISTRY")
        run_id = os.environ.get("FOODSPEC_RUN_ID", path.stem)
        if reg_path:
            try:
                reg = FeatureModelRegistry(Path(reg_path))
                reg.register_model(
                    run_id=run_id,
                    model_path=str(path),
                    model_metadata={
                        "protocol_name": self.metadata.get("protocol_name"),
                        "protocol_version": self.metadata.get("protocol_version"),
                        "preprocessing": payload["preprocess_config"],
                        "features": [{"name": fn} for fn in self.feature_names],
                        "model_type": self.metadata.get("model_type"),
                        "metrics": self.metadata.get("metrics", {}),
                        "validation_strategy": self.metadata.get("validation_strategy"),
                    },
                )
            except Exception:
                pass

    @staticmethod
    def load(path: Path) -> "FrozenModel":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        with (path.with_suffix(".pkl")).open("rb") as f:
            payload = pickle.load(f)
        pp_cfg = PreprocessingConfig(**meta["preprocess_config"])
        peaks = [PeakDefinition(**p) for p in meta["peaks"]]
        ratios = [RatioDefinition(**r) for r in meta["ratios"]]
        return FrozenModel(
            preprocess_config=pp_cfg,
            peaks=peaks,
            ratios=ratios,
            feature_names=meta["feature_names"],
            label_col=meta["label_col"],
            model=payload["model"],
            scaler=payload["scaler"],
            metadata=meta["metadata"],
        )


__all__ = ["TrainablePipeline", "FrozenModel"]
