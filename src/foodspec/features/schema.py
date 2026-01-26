"""Feature engineering schemas and protocol parsing helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeakSpec:
    """Definition for a peak-centered feature."""

    name: str
    center: float
    window: float = 5.0
    baseline: str = "none"
    assignment: Optional[str] = None


@dataclass(frozen=True)
class BandSpec:
    """Definition for a band integration feature."""

    name: str
    start: float
    end: float
    baseline: str = "none"
    assignment: Optional[str] = None


@dataclass(frozen=True)
class RatioSpec:
    """Definition for a ratio feature."""

    name: str
    numerator: str
    denominator: str


@dataclass(frozen=True)
class FeatureInfo:
    """Human- and machine-readable description of a feature."""

    name: str
    ftype: str
    assignment: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ftype": self.ftype,
            "assignment": self.assignment,
            "description": self.description,
            "params": dict(self.params),
        }


@dataclass
class FeatureConfig:
    """Parsed feature configuration from a protocol file."""

    peaks: list[PeakSpec] = field(default_factory=list)
    bands: list[BandSpec] = field(default_factory=list)
    ratios: list[RatioSpec] = field(default_factory=list)
    embedding: str = "pca"
    n_components: int = 2
    peak_metrics: tuple[str, ...] = ("height", "area")
    band_metrics: tuple[str, ...] = ("integral",)
    scaling: str = "standard"


def normalize_assignment(value: Optional[str]) -> str:
    """Normalize assignment strings to avoid hallucinated labels."""

    if value is None:
        return "unassigned"
    stripped = str(value).strip()
    return stripped or "unassigned"


def load_protocol_payload(path: Path) -> dict[str, Any]:
    """Load protocol YAML/JSON as a raw dictionary."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - environment guard
            raise ImportError("PyYAML is required to parse protocol YAML files") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Protocol payload must be a JSON/YAML object.")
    return payload


def _dedupe_by_name(items: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        name = getattr(item, "name", None)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(item)
    return out


def _parse_peak_specs(items: Iterable[dict[str, Any]], *, default_window: float, default_baseline: str) -> list[PeakSpec]:
    peaks: list[PeakSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or f"peak_{item.get('wavenumber', item.get('center', ''))}"
        center = item.get("wavenumber", item.get("center", item.get("wn")))
        if center is None:
            continue
        window = float(item.get("window", item.get("tolerance", default_window)))
        baseline = str(item.get("baseline", default_baseline))
        assignment = item.get("assignment") or item.get("chemical")
        peaks.append(
            PeakSpec(
                name=str(name),
                center=float(center),
                window=window,
                baseline=baseline,
                assignment=assignment,
            )
        )
    return peaks


def _parse_band_specs(items: Iterable[dict[str, Any]], *, default_baseline: str) -> list[BandSpec]:
    bands: list[BandSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("label") or "band"
        start = item.get("start", item.get("min", None))
        end = item.get("end", item.get("max", None))
        if start is None and isinstance(item.get("range"), (list, tuple)):
            start = item["range"][0]
            end = item["range"][1] if len(item["range"]) > 1 else None
        if start is None or end is None:
            continue
        baseline = str(item.get("baseline", default_baseline))
        assignment = item.get("assignment") or item.get("chemical")
        bands.append(
            BandSpec(
                name=str(name),
                start=float(start),
                end=float(end),
                baseline=baseline,
                assignment=assignment,
            )
        )
    return bands


def _parse_ratio_specs(items: Iterable[dict[str, Any]]) -> list[RatioSpec]:
    ratios: list[RatioSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        numerator = item.get("numerator")
        denominator = item.get("denominator")
        if not numerator or not denominator:
            continue
        name = item.get("name") or f"{numerator}/{denominator}"
        ratios.append(RatioSpec(name=str(name), numerator=str(numerator), denominator=str(denominator)))
    return ratios


def parse_feature_config(protocol_path: Path) -> FeatureConfig:
    """Parse a protocol file to construct a FeatureConfig."""

    payload = load_protocol_payload(protocol_path)
    features_payload = payload.get("features", {}) if isinstance(payload.get("features"), dict) else {}

    peaks: list[PeakSpec] = []
    bands: list[BandSpec] = []
    ratios: list[RatioSpec] = []

    default_window = float(features_payload.get("peak_window", 5.0))
    default_peak_baseline = str(features_payload.get("peak_baseline", "none"))
    default_band_baseline = str(features_payload.get("band_baseline", "none"))

    peaks.extend(_parse_peak_specs(features_payload.get("peaks", []), default_window=default_window, default_baseline=default_peak_baseline))
    bands.extend(_parse_band_specs(features_payload.get("bands", []), default_baseline=default_band_baseline))
    ratios.extend(_parse_ratio_specs(features_payload.get("ratios", [])))

    for step in payload.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_type = step.get("type")
        params = step.get("params", {}) if isinstance(step.get("params"), dict) else {}
        if step_type in {"preprocess", "features"}:
            peaks.extend(
                _parse_peak_specs(
                    params.get("peaks", []),
                    default_window=float(params.get("peak_window", default_window)),
                    default_baseline=str(params.get("peak_baseline", default_peak_baseline)),
                )
            )
            bands.extend(
                _parse_band_specs(
                    params.get("bands", []),
                    default_baseline=str(params.get("band_baseline", default_band_baseline)),
                )
            )
        if step_type in {"rq_analysis", "features"}:
            ratios.extend(_parse_ratio_specs(params.get("ratios", [])))

    assignment_overrides = features_payload.get("assignments", {}) if isinstance(features_payload.get("assignments"), dict) else {}
    if assignment_overrides:
        peaks = [
            PeakSpec(
                name=p.name,
                center=p.center,
                window=p.window,
                baseline=p.baseline,
                assignment=assignment_overrides.get(p.name, p.assignment),
            )
            for p in peaks
        ]
        bands = [
            BandSpec(
                name=b.name,
                start=b.start,
                end=b.end,
                baseline=b.baseline,
                assignment=assignment_overrides.get(b.name, b.assignment),
            )
            for b in bands
        ]

    peaks = _dedupe_by_name(peaks)
    bands = _dedupe_by_name(bands)
    ratios = _dedupe_by_name(ratios)

    embedding = str(features_payload.get("embedding", "pca"))
    n_components = int(features_payload.get("n_components", 2))
    peak_metrics = tuple(features_payload.get("peak_metrics", ("height", "area")))
    band_metrics = tuple(features_payload.get("band_metrics", ("integral",)))
    scaling = str(features_payload.get("scaling", "standard"))

    return FeatureConfig(
        peaks=peaks,
        bands=bands,
        ratios=ratios,
        embedding=embedding,
        n_components=n_components,
        peak_metrics=peak_metrics,
        band_metrics=band_metrics,
        scaling=scaling,
    )


def split_spectral_dataframe(
    df: pd.DataFrame,
    *,
    exclude: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Split a wide DataFrame into spectral matrix, axis, and metadata."""

    exclude_set = {col for col in (exclude or []) if col is not None}
    spectral_cols: list[tuple[float, str]] = []
    for col in df.columns:
        if col in exclude_set:
            continue
        try:
            wn = float(col)
        except (TypeError, ValueError):
            continue
        spectral_cols.append((wn, col))

    if not spectral_cols:
        raise ValueError("No spectral columns found with numeric column names.")

    spectral_cols.sort(key=lambda x: x[0])
    ordered_cols = [col for _, col in spectral_cols]
    wavenumbers = np.array([wn for wn, _ in spectral_cols], dtype=float)
    X = df[ordered_cols].to_numpy(dtype=float)
    meta = df.drop(columns=ordered_cols)
    return X, wavenumbers, meta


__all__ = [
    "PeakSpec",
    "BandSpec",
    "RatioSpec",
    "FeatureInfo",
    "FeatureConfig",
    "normalize_assignment",
    "load_protocol_payload",
    "parse_feature_config",
    "split_spectral_dataframe",
]
