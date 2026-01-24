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

from foodspec.core.manifest import RunManifest


def test_protocol_hash_and_data_fingerprint(tmp_path: Path) -> None:
    protocol = {"version": "2.0.0", "task": {"name": "cls"}}
    data_file = tmp_path / "data.csv"
    data_file.write_text("id,value\n1,2\n")

    manifest = RunManifest.build(
        protocol_snapshot=protocol,
        data_path=data_file,
        seed=42,
        artifacts={"metrics": "metrics.csv"},
        warnings=["note"],
    )

    # Hashes are deterministic
    assert manifest.protocol_hash == RunManifest.compute_protocol_hash(protocol)
    assert manifest.data_fingerprint == RunManifest.compute_data_fingerprint(data_file)
    assert manifest.seed == 42
    assert manifest.artifacts["metrics"] == "metrics.csv"
    assert manifest.warnings == ["note"]
    assert manifest.duration_seconds >= 0.0


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    protocol = {"version": "2.0.0", "task": {"name": "cls"}}
    manifest = RunManifest.build(
        protocol_snapshot=protocol,
        data_path=None,
        seed=None,
        artifacts={"metrics": "metrics.csv", "manifest": "manifest.json"},
    )

    path = tmp_path / "manifest.json"
    manifest.save(path)

    loaded = RunManifest.load(path)
    assert loaded.protocol_hash == manifest.protocol_hash
    assert loaded.python_version == manifest.python_version
    assert loaded.artifacts["manifest"] == "manifest.json"
    assert loaded.duration_seconds == manifest.duration_seconds
