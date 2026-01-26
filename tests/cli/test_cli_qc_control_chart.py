from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from foodspec.cli.main import app


def test_cli_qc_control_chart(tmp_path: Path) -> None:
    csv_path = tmp_path / "values.csv"
    df = pd.DataFrame({"value": [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03]})
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "qc_out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["qc", "control-chart", str(csv_path), "--value-col", "value", "--chart", "imr", "--run-dir", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "qc" / "control_charts.json").exists()
