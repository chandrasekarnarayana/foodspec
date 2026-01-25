"""Chemistry-aware feature engine with FeatureSpec definitions."""
from __future__ import annotations


import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from foodspec.features.bands import compute_band_features
from foodspec.features.interpretation import explain_feature_spec
from foodspec.features.peaks import PeakFeatureExtractor


@dataclass
class FeatureSpec:
    name: str
    ftype: str  # peak, band, ratio, index, embedding
    regions: Optional[List[Tuple[float, float]]] = None
    formula: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    citation: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:8]


class FeatureEngine:
    """Evaluate FeatureSpec definitions on a FoodSpectrumSet."""

    def __init__(self, specs: Sequence[FeatureSpec]):
        self.specs = list(specs)

    def _validate(self, ds, spec: FeatureSpec) -> None:
        modality = spec.constraints.get("modality")
        if modality and ds.modality != modality:
            raise ValueError(f"Spec {spec.name} requires modality {modality}, got {ds.modality}")
        wn_range = spec.constraints.get("wavenumber_range")
        if wn_range is not None:
            wn_min, wn_max = wn_range
            if ds.wavenumbers.min() > wn_min or ds.wavenumbers.max() < wn_max:
                raise ValueError(f"Spec {spec.name} requires axis covering {wn_min}-{wn_max}")

    def _compute_band(self, ds, spec: FeatureSpec) -> pd.DataFrame:
        regions = spec.regions or []
        bands = [(spec.name if len(regions) == 1 else f"{spec.name}_{i}", lo, hi) for i, (lo, hi) in enumerate(regions)]
        metrics = spec.params.get("metrics", ("integral",))
        return compute_band_features(ds.x, ds.wavenumbers, bands, metrics=metrics)

    def _compute_peak(self, ds, spec: FeatureSpec) -> pd.DataFrame:
        regions = spec.regions or []
        centers = [(lo + hi) / 2 for lo, hi in regions]
        tol = spec.params.get("tolerance", 5.0)
        metrics = spec.params.get("metrics", ("height", "area"))
        extractor = PeakFeatureExtractor(expected_peaks=centers, tolerance=tol, features=metrics)
        values = extractor.transform(ds.x, wavenumbers=ds.wavenumbers)
        cols = extractor.get_feature_names_out()
        return pd.DataFrame(values, columns=cols)

    def _compute_ratio_or_index(self, df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        safe_locals = {col: df[col] for col in df.columns}
        expr = spec.formula
        if not expr:
            raise ValueError(f"Spec {spec.name} missing formula")
        try:
            result = pd.eval(expr, local_dict=safe_locals, engine="python")
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to evaluate formula for {spec.name}: {exc}")
        return pd.DataFrame({spec.name: result})

    def evaluate(self, ds) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        features = pd.DataFrame(index=np.arange(len(ds.x)))
        diag: Dict[str, Any] = {"applied": []}

        for spec in self.specs:
            self._validate(ds, spec)
            if spec.ftype == "band":
                df = self._compute_band(ds, spec)
            elif spec.ftype == "peak":
                df = self._compute_peak(ds, spec)
            elif spec.ftype in {"ratio", "index"}:
                df = self._compute_ratio_or_index(features, spec)
            else:
                raise ValueError(f"Unsupported feature type {spec.ftype}")

            features = pd.concat([features, df], axis=1)
            diag["applied"].append(
                {
                    "name": spec.name,
                    "type": spec.ftype,
                    "hash": spec.hash(),
                    "explanation": explain_feature_spec(spec, modality=getattr(ds, "modality", None)),
                }
            )

        return features, diag


__all__ = ["FeatureSpec", "FeatureEngine"]
