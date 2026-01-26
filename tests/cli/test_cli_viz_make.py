from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from foodspec.cli.main import app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_cli_viz_make_all(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", {"protocol_snapshot": {"steps": [{"type": "load"}]}})
    _write_json(run_dir / "run_summary.json", {"status": "success"})

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"y_true": [0, 1, 0, 1], "y_pred": [0, 1, 1, 1], "proba": [0.1, 0.9, 0.7, 0.8]}
    ).to_csv(tables_dir / "predictions.csv", index=False)
    pd.DataFrame([[1, 2, 3], [3, 2, 1]]).to_csv(tables_dir / "features.csv", index=False)
    pd.DataFrame([[0.1, 0.2], [0.2, 0.3]]).to_csv(tables_dir / "spectra_raw.csv", index=False)
    pd.DataFrame([[0.08, 0.18], [0.18, 0.28]]).to_csv(tables_dir / "spectra_processed.csv", index=False)

    out_dir = tmp_path / "viz_out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["viz", "make", "--run", str(run_dir), "--outdir", str(out_dir), "--all"],
    )
    assert result.exit_code == 0, result.output

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "run_summary.json").exists()
    assert (out_dir / "logs" / "run.log").exists()
    assert (out_dir / "figures").exists()
    assert any(out_dir.joinpath("figures").glob("*.png"))
