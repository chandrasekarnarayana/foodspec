"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
"""

from pathlib import Path

import pytest

from foodspec.core.orchestrator import ExecutionEngine
from foodspec.core.protocol import ProtocolV2, DataSpec, TaskSpec


def test_minimal_run_writes_manifest_and_logs(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    data_file.write_text("id,value\n1,2\n")

    protocol = ProtocolV2(
        data=DataSpec(
            input=str(data_file),
            modality="raman",
            label="value",
            metadata_map={"sample_id": "id", "modality": "modality", "label": "value"},
        ),
        task=TaskSpec(name="classification", objective="max"),
    )

    outdir = tmp_path / "run"
    engine = ExecutionEngine()
    result = engine.run(protocol, outdir, seed=123)

    manifest_path = outdir / "manifest.json"
    logs_path = outdir / "logs.txt"

    assert manifest_path.exists()
    assert logs_path.exists()

    text = manifest_path.read_text()
    assert "protocol_hash" in text
    assert "data_fingerprint" in text
    assert result.manifest.data_fingerprint
    assert "Seeds set to 123" in logs_path.read_text()


def test_requested_stage_raises_not_implemented(tmp_path: Path) -> None:
    protocol = ProtocolV2(
        data=DataSpec(
            input=str(tmp_path / "missing.csv"),
            modality="raman",
            label="value",
            metadata_map={"sample_id": "id", "modality": "modality", "label": "value"},
        ),
        task=TaskSpec(name="classification", objective="max"),
    )
    protocol = protocol.model_copy(update={"preprocess": {"steps": [{"component": "normalize"}]}})

    engine = ExecutionEngine()

    with pytest.raises(NotImplementedError):
        engine.run(protocol, tmp_path / "run")
