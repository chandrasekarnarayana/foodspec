"""Hybrid feature construction utilities."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from foodspec.features.bands import extract_band_features
from foodspec.features.embeddings import pca_embeddings, pls_embeddings
from foodspec.features.peaks import extract_peak_features
from foodspec.features.ratios import compute_ratios
from foodspec.features.schema import FeatureConfig, FeatureInfo, RatioSpec, normalize_assignment


def combine_feature_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate feature tables column-wise."""

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, axis=1)


def scale_features(df: pd.DataFrame, method: Optional[str]) -> tuple[pd.DataFrame, Optional[Any]]:
    """Scale features with standard or robust scaling."""

    if method is None or str(method).lower() in {"none", "off"}:
        return df, None
    method = str(method).lower()
    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    values = scaler.fit_transform(df.to_numpy(dtype=float))
    return pd.DataFrame(values, columns=df.columns), scaler


def _ratio_feature_info(
    ratios: list[RatioSpec],
    assignments: dict[str, str],
) -> list[FeatureInfo]:
    info: list[FeatureInfo] = []
    for ratio in ratios:
        num_assign = assignments.get(ratio.numerator, "unassigned")
        den_assign = assignments.get(ratio.denominator, "unassigned")
        desc = f"Ratio {ratio.numerator}/{ratio.denominator} ({num_assign} over {den_assign})."
        info.append(
            FeatureInfo(
                name=ratio.name,
                ftype="ratio",
                assignment=f"{num_assign}/{den_assign}",
                description=desc,
                params={"numerator": ratio.numerator, "denominator": ratio.denominator},
            )
        )
    return info


def extract_features(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    feature_type: str,
    config: FeatureConfig,
    labels: Optional[np.ndarray] = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[FeatureInfo], dict[str, Any]]:
    """Extract a feature table based on a feature backend."""

    feature_type = feature_type.lower()
    info: list[FeatureInfo] = []
    details: dict[str, Any] = {"feature_type": feature_type}

    if feature_type == "raw":
        cols = [f"{wn:.4f}" for wn in wavenumbers]
        df = pd.DataFrame(np.asarray(X, dtype=float), columns=cols)
        for col in cols:
            info.append(
                FeatureInfo(
                    name=col,
                    ftype="raw",
                    assignment="unassigned",
                    description=f"Raw intensity at {col} cm^-1.",
                    params={"wavenumber": col},
                )
            )
        return df, info, details

    if feature_type == "peaks":
        df, info = extract_peak_features(
            X,
            wavenumbers,
            config.peaks,
            metrics=config.peak_metrics,
            default_window=5.0,
            default_baseline="none",
        )
    elif feature_type == "bands":
        df, info = extract_band_features(X, wavenumbers, config.bands, metrics=config.band_metrics)
    elif feature_type == "pca":
        df, meta = pca_embeddings(X, n_components=config.n_components)
        details["embedding"] = meta
        for col in df.columns:
            info.append(
                FeatureInfo(
                    name=col,
                    ftype="pca",
                    assignment="unassigned",
                    description=f"PCA component {col.split('_')[-1]}",
                    params={"n_components": config.n_components},
                )
            )
    elif feature_type == "pls":
        if labels is None:
            raise ValueError("PLS embeddings require labels.")
        df, meta = pls_embeddings(X, labels, n_components=config.n_components, mode="classification")
        details["embedding"] = meta
        for col in df.columns:
            info.append(
                FeatureInfo(
                    name=col,
                    ftype="pls",
                    assignment="unassigned",
                    description=f"PLS component {col.split('_')[-1]}",
                    params={"n_components": config.n_components},
                )
            )
    elif feature_type == "hybrid":
        tables: list[pd.DataFrame] = []
        if config.peaks:
            peaks_df, peak_info = extract_peak_features(
                X,
                wavenumbers,
                config.peaks,
                metrics=config.peak_metrics,
                default_window=5.0,
                default_baseline="none",
            )
            tables.append(peaks_df)
            info.extend(peak_info)
        if config.bands:
            bands_df, band_info = extract_band_features(X, wavenumbers, config.bands, metrics=config.band_metrics)
            tables.append(bands_df)
            info.extend(band_info)
        embedding = config.embedding.lower()
        if embedding == "pls":
            if labels is None:
                raise ValueError("Hybrid PLS embeddings require labels.")
            embed_df, meta = pls_embeddings(X, labels, n_components=config.n_components, mode="classification")
        else:
            embed_df, meta = pca_embeddings(X, n_components=config.n_components)
        tables.append(embed_df)
        details["embedding"] = meta
        for col in embed_df.columns:
            info.append(
                FeatureInfo(
                    name=col,
                    ftype="embedding",
                    assignment="unassigned",
                    description=f"{embedding.upper()} component {col.split('_')[-1]}",
                    params={"n_components": config.n_components},
                )
            )
        df = combine_feature_tables(tables)
        df, _scaler = scale_features(df, config.scaling)
        details["scaling"] = config.scaling
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    assignments = {entry.name: normalize_assignment(entry.assignment) for entry in info}
    if config.ratios:
        available = [
            ratio for ratio in config.ratios if ratio.numerator in df.columns and ratio.denominator in df.columns
        ]
        if available:
            ratio_map = {ratio.name: (ratio.numerator, ratio.denominator) for ratio in available}
            df = compute_ratios(df, ratio_map)
            info.extend(_ratio_feature_info(available, assignments))
        else:
            details["ratio_skipped"] = [ratio.name for ratio in config.ratios]

    details["n_features"] = int(df.shape[1])
    details["seed"] = seed
    return df, info, details


__all__ = ["combine_feature_tables", "scale_features", "extract_features"]
