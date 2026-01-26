from __future__ import annotations

from pathlib import Path

import pandas as pd

from foodspec.experiment import Experiment, ExperimentConfig, ValidationScheme
from foodspec.protocol.config import ProtocolConfig


def _write_csv(path: Path, offset: float) -> Path:
    df = pd.DataFrame(
        {
            "f1": [0.1 + offset, 0.2 + offset, 0.3 + offset, 0.4 + offset],
            "f2": [1.0 + offset, 1.1 + offset, 1.2 + offset, 1.3 + offset],
            "label": ["a", "b", "a", "b"],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_run_batch_creates_summary(tmp_path: Path) -> None:
    proto = ProtocolConfig(name="test")
    config = ExperimentConfig(protocol_config=proto, scheme=ValidationScheme.NESTED, model="logreg")
    exp = Experiment(config)

    csv1 = _write_csv(tmp_path / "batch1.csv", 0.0)
    csv2 = _write_csv(tmp_path / "batch2.csv", 1.0)

    outdir = tmp_path / "batch_runs"
    result = exp.run_batch([csv1, csv2], outdir, parallel=False, seed=123)

    assert result.summary_path is not None
    assert result.summary_path.exists()
    assert result.success_count + result.failure_count == 2
