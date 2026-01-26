from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from foodspec.cli.main import app


def test_cli_train_csv_modeling(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "feat1": rng.normal(size=n),
            "feat2": rng.normal(size=n),
            "label": [0, 1] * (n // 2),
            "stage": ["A", "B", "C", "D"] * (n // 4),
        }
    )
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps({"name": "test-protocol", "expected_columns": {"label_col": "label"}}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "run"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "--csv",
            str(csv_path),
            "--protocol",
            str(protocol_path),
            "--scheme",
            "loso",
            "--group",
            "stage",
            "--model",
            "logreg",
            "--outdir",
            str(out_dir),
            "--outer-splits",
            "3",
            "--inner-splits",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "run_summary.json").exists()
    assert (out_dir / "logs" / "run.log").exists()
    assert (out_dir / "models" / "best_model.joblib").exists()
    assert (out_dir / "models" / "model_card.json").exists()
    assert (out_dir / "validation" / "folds.json").exists()
    assert (out_dir / "metrics" / "metrics.json").exists()
    assert (out_dir / "metrics" / "metrics_by_group.json").exists()

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary.get("status") == "success"
