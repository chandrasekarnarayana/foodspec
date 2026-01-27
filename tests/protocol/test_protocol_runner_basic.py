from __future__ import annotations

from pathlib import Path

import pandas as pd

from foodspec.protocol.config import ProtocolConfig
from foodspec.protocol.runner import ProtocolRunner


def test_protocol_runner_minimal_csv(tmp_path):
    protocol_path = Path("tests/fixtures/minimal_protocol.yaml")
    config = ProtocolConfig.from_file(protocol_path)
    runner = ProtocolRunner(config)

    df = pd.DataFrame(
        {
            "wavenumber_1": [1.0, 2.0, 3.0],
            "wavenumber_2": [1.1, 2.1, 3.1],
            "wavenumber_3": [1.2, 2.2, 3.2],
            "wavenumber_4": [1.3, 2.3, 3.3],
            "wavenumber_5": [1.4, 2.4, 3.4],
            "label": ["a", "b", "a"],
        }
    )

    result = runner.run([df])
    assert result.metadata["protocol"] == config.name
    assert isinstance(result.logs, list)

    out_dir = tmp_path / "protocol_outputs"
    runner.save_outputs(result, out_dir)
    assert (out_dir / "metadata.json").exists()
