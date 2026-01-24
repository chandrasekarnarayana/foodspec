import json
from pathlib import Path

import numpy as np

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.features import MarkerPanel


def test_marker_panel_roundtrip_and_save(tmp_path):
    # Create marker panel
    selected_names = ["pca:pca_1", "pls:pls_1", "ratio@1652/1742"]
    freqs = [0.8, 0.7, 0.65]
    peak_map = {"ratio@1652/1742": {"numerator": 1652, "denominator": 1742}}
    band_map = None
    panel = MarkerPanel(
        selected_feature_names=selected_names,
        selection_frequencies=freqs,
        peak_mappings=peak_map,
        band_mappings=band_map,
        created_by="StabilitySelector",
        protocol_hash="abc123",
        extra={"note": "phase8"},
    )

    # Roundtrip via JSON
    payload = panel.to_json()
    panel2 = MarkerPanel.from_json(payload)
    assert panel2.selected_feature_names == selected_names
    assert panel2.selection_frequencies == freqs
    assert panel2.peak_mappings == peak_map
    assert panel2.band_mappings == band_map
    assert panel2.created_by == "StabilitySelector"
    assert panel2.protocol_hash == "abc123"
    assert panel2.extra["note"] == "phase8"
    assert panel2.created_at  # has a timestamp

    # Save via ArtifactRegistry
    reg = ArtifactRegistry(tmp_path / "run")
    panel2.save(reg)
    p = reg.root / "marker_panel.json"
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded["selected_feature_names"] == selected_names
    assert loaded["selection_frequencies"] == freqs
