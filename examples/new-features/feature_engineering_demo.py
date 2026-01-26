from __future__ import annotations

from pathlib import Path

import pandas as pd

from foodspec.features.hybrid import extract_features
from foodspec.features.marker_panel import build_marker_panel, export_marker_panel
from foodspec.features.schema import parse_feature_config, split_spectral_dataframe
from foodspec.features.selection import feature_importance_scores, stability_selection
from foodspec.utils.run_artifacts import safe_json_dump


def main() -> None:
    csv_path = Path("examples/data/oil_synthetic.csv")
    protocol_path = Path("examples/protocols/EdibleOil_Classification_v1.yaml")
    out_dir = Path("runs/examples/feature_engineering")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    config = parse_feature_config(protocol_path)
    label_col = "oil_type"

    X, wavenumbers, meta = split_spectral_dataframe(df, exclude=[label_col])
    features_df, info, details = extract_features(
        X,
        wavenumbers,
        feature_type="peaks",
        config=config,
        labels=meta[label_col].to_numpy(),
    )

    combined = pd.concat([meta.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    features_path = out_dir / "features.csv"
    combined.to_csv(features_path, index=False)
    safe_json_dump(out_dir / "feature_info.json", {"features": [entry.to_dict() for entry in info], "details": details})

    stability = stability_selection(features_df, meta[label_col], seed=42, n_resamples=15)
    perf = feature_importance_scores(features_df, meta[label_col])
    panel = build_marker_panel(stability, performance=perf, k=6, feature_info=[entry.to_dict() for entry in info])
    export_marker_panel(panel, out_dir)

    print(f"Saved features to {features_path}")


if __name__ == "__main__":
    main()
