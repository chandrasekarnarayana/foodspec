import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _make_spiky_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "oil_type": ["A", "B"],
            "1000": [1.0, 0.9],
            "1010": [50.0, 0.95],  # clear spike in first row
            "1020": [1.0, 0.97],
        }
    )
    df.to_csv(path, index=False)


def _make_preprocess_protocol(path: Path, out_dir: Path) -> None:
    proto = {
        "name": "spike_toggle_protocol",
        "version": "0.1.0",
        "steps": [
            {
                "type": "preprocess",
                "params": {
                    "baseline_enabled": False,
                    "smooth_enabled": False,
                    "normalization": "none",
                    # spike_removal will be overridden by CLI flags
                },
            },
            {"type": "output", "params": {"output_dir": str(out_dir)}},
        ],
    }
    path.write_text(json.dumps(proto), encoding="utf-8")


def test_cli_protocol_spike_flags(tmp_path: Path):
    csv = tmp_path / "spiky.csv"
    _make_spiky_csv(csv)
    runs = tmp_path / "runs"
    proto = tmp_path / "proto.json"
    _make_preprocess_protocol(proto, runs)

    # Run with spike removal ON
    cmd_on = [
        sys.executable,
        "-m",
        "foodspec.cli_protocol",
        "--input",
        str(csv),
        "--protocol",
        str(proto),
        "--output-dir",
        str(runs),
        "--spike-removal",
    ]
    res_on = subprocess.run(cmd_on, capture_output=True, text=True)
    assert res_on.returncode == 0, res_on.stderr
    out_dir_on = runs / f"spike_toggle_protocol_{csv.stem}"
    meta_on = json.loads((out_dir_on / "metadata.json").read_text())
    assert meta_on.get("preprocessing", {}).get("spike_removal") is True

    # Run with spike removal OFF
    cmd_off = [
        sys.executable,
        "-m",
        "foodspec.cli_protocol",
        "--input",
        str(csv),
        "--protocol",
        str(proto),
        "--output-dir",
        str(runs),
        "--no-spike-removal",
    ]
    res_off = subprocess.run(cmd_off, capture_output=True, text=True)
    assert res_off.returncode == 0, res_off.stderr
    out_dir_off = runs / f"spike_toggle_protocol_{csv.stem}"
    meta_off = json.loads((out_dir_off / "metadata.json").read_text())
    assert meta_off.get("preprocessing", {}).get("spike_removal") is False
