import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_protocol_batch_mode(tmp_path: Path):
    # Two tiny inputs
    for i in range(2):
        df = pd.DataFrame(
            {
                "oil_type": ["A", "B"],
                "matrix": ["oil", "oil"],
                "heating_stage": [0, 1],
                "I_1742": [10, 6],
                "I_2720": [5, 5],
            }
        )
        df.to_csv(tmp_path / f"toy_{i}.csv", index=False)

    proto = tmp_path / "proto.json"
    proto.write_text(
        json.dumps(
            {
                "name": "toy_protocol",
                "steps": [
                    {
                        "type": "rq_analysis",
                        "params": {
                            "oil_col": "oil_type",
                            "matrix_col": "matrix",
                            "heating_col": "heating_stage",
                            "ratios": [{"name": "1742/2720", "numerator": "I_1742", "denominator": "I_2720"}],
                        },
                    },
                    {"type": "output", "params": {"output_dir": str(tmp_path / "runs")}},
                ],
            }
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "foodspec.cli_protocol",
        "--input-dir",
        str(tmp_path),
        "--glob",
        "toy_*.csv",
        "--protocol",
        str(proto),
        "--output-dir",
        str(tmp_path / "runs"),
        "--quiet",
        "--cv-folds",
        "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
