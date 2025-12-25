import json
from pathlib import Path

import pandas as pd

from foodspec.output_bundle import save_index


def test_output_bundle_index(tmp_path: Path):
    run_dir = tmp_path
    metadata = {"models": ["m1"], "foo": "bar"}
    tables = {"t1": pd.DataFrame({"a": [1]})}
    figures = {"f1": None}
    warnings = ["w1"]
    save_index(run_dir, metadata, tables, figures, warnings)
    idx = json.loads((run_dir / "index.json").read_text())
    assert "run_id" in idx
    assert idx["models"] == ["m1"]
    assert "t1" in idx["tables"]
