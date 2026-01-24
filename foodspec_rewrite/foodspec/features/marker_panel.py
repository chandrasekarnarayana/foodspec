"""
MarkerPanel artifact helpers: serialization and saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from foodspec.core.artifacts import ArtifactRegistry


@dataclass
class MarkerPanel:
    """Publication-grade marker panel artifact.

    Stores selected feature names with selection frequencies and
    optional peak/band mappings when available, along with provenance.

    Attributes
    ----------
    selected_feature_names : list[str]
        Names of selected features (e.g., "pca_1", "ratio@1652/1742").
    selection_frequencies : list[float]
        Selection frequencies for all features in the original space.
    peak_mappings : dict | None
        Optional mappings for peaks (e.g., peak locations per name).
    band_mappings : dict | None
        Optional mappings for bands (e.g., band ranges per name).
    created_by : str
        Identifier of component that created the panel (e.g., "StabilitySelector").
    created_at : str
        ISO 8601 timestamp in UTC.
    protocol_hash : str
        Hash/identifier of the protocol configuration to ensure reproducibility.
    extra : dict
        Additional optional fields.
    """

    selected_feature_names: List[str]
    selection_frequencies: List[float]
    selected_indices: List[int] = field(default_factory=list)
    selected_wavenumbers: Optional[List[float]] = None
    # Provenance fields commonly expected at top-level
    n_splits: Optional[int] = None
    n_resamples: Optional[int] = None
    subsample_fraction: Optional[float] = None
    selection_threshold: Optional[float] = None
    seed: Optional[int] = None
    peak_mappings: Optional[Dict[str, Any]] = None
    band_mappings: Optional[Dict[str, Any]] = None
    created_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    protocol_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Serialize to a JSON-serializable dict."""
        payload: Dict[str, Any] = {
            "selected_feature_names": list(self.selected_feature_names),
            "selected_indices": list(self.selected_indices),
            "selection_frequencies": list(self.selection_frequencies),
            "n_splits": self.n_splits,
            "n_resamples": self.n_resamples,
            "subsample_fraction": self.subsample_fraction,
            "selection_threshold": self.selection_threshold,
            "seed": self.seed,
            "peak_mappings": self.peak_mappings,
            "band_mappings": self.band_mappings,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "protocol_hash": self.protocol_hash,
        }
        if self.selected_wavenumbers is not None:
            payload["selected_wavenumbers"] = self.selected_wavenumbers
        if self.extra:
            payload["extra"] = self.extra
        return payload

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "MarkerPanel":
        """Construct from a JSON dict."""
        return cls(
            selected_feature_names=list(payload.get("selected_feature_names", [])),
            selected_indices=list(payload.get("selected_indices", [])),
            selection_frequencies=list(payload.get("selection_frequencies", [])),
            selected_wavenumbers=payload.get("selected_wavenumbers"),
            n_splits=payload.get("n_splits"),
            n_resamples=payload.get("n_resamples"),
            subsample_fraction=payload.get("subsample_fraction"),
            selection_threshold=payload.get("selection_threshold"),
            seed=payload.get("seed"),
            peak_mappings=payload.get("peak_mappings"),
            band_mappings=payload.get("band_mappings"),
            created_by=str(payload.get("created_by", "")),
            created_at=str(payload.get("created_at", datetime.now(timezone.utc).isoformat())),
            protocol_hash=str(payload.get("protocol_hash", "")),
            extra=dict(payload.get("extra", {})),
        )

    def save(self, registry: ArtifactRegistry) -> None:
        """Save panel to marker_panel.json using ArtifactRegistry."""
        registry.ensure_layout()
        path = registry.root / "marker_panel.json"
        registry.write_json(path, self.to_json())


__all__ = ["MarkerPanel"]
