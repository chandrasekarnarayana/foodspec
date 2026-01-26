from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from foodspec.cli.main import app


def test_cli_features_extract_peaks(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "1000.0": [1.0, 1.2, 0.9, 1.1],
            "1005.0": [0.5, 0.6, 0.4, 0.55],
            "1010.0": [0.2, 0.25, 0.18, 0.22],
        }
    )
    csv_path = tmp_path / "spectra.csv"
    df.to_csv(csv_path, index=False)

    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "name": "features-test",
                "expected_columns": {"label_col": "label"},
                "steps": [
                    {
                        "type": "preprocess",
                        "params": {
                            "peaks": [
                                {"name": "I_1005", "wavenumber": 1005.0, "window": 2.0},
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "run"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "features",
            "extract",
            "--csv",
            str(csv_path),
            "--protocol",
            str(protocol_path),
            "--type",
            "peaks",
            "--outdir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "run_summary.json").exists()
    assert (out_dir / "logs" / "run.log").exists()
    assert (out_dir / "features" / "features.csv").exists()
    assert (out_dir / "features" / "feature_info.json").exists()
