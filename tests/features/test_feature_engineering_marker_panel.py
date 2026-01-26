import json
from pathlib import Path

import pandas as pd

from foodspec.features.marker_panel import build_marker_panel, export_marker_panel


def test_marker_panel_export(tmp_path: Path):
    stability = pd.Series({"f1": 0.9, "f2": 0.5, "f3": 0.2})
    feature_info = [
        {"name": "f1", "assignment": "C=O stretch", "description": "Peak at 1742 cm^-1"},
        {"name": "f2", "assignment": "unassigned", "description": "Unknown band"},
    ]
    panel = build_marker_panel(stability, k=2, feature_info=feature_info)
    json_path, csv_path = export_marker_panel(panel, tmp_path)

    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text())
    assert "panel" in payload
    assert len(payload["panel"]) == 2

    required_fields = {"rank", "feature", "stability", "performance", "interpretability", "score", "assignment", "explanation"}
    first = payload["panel"][0]
    assert required_fields.issubset(first.keys())
    assert first["assignment"]
