from pathlib import Path

from foodspec.registry import FeatureModelRegistry


def test_registry_run_and_model():
    reg_path = Path("registry_test.json")
    if reg_path.exists():
        reg_path.unlink()
    reg = FeatureModelRegistry(reg_path)
    reg.register_run(
        "run123",
        {
            "protocol": "proto",
            "protocol_version": "0.1",
            "inputs": ["a.csv"],
            "preprocessing": {},
            "validation_strategy": "standard",
        },
    )
    reg.register_model(
        "run123",
        "model.pkl",
        {
            "protocol_name": "proto",
            "protocol_version": "0.1",
            "features": [{"name": "I_1000"}],
            "model_type": "rf",
        },
    )
    entries = reg.query_by_protocol("proto")
    assert entries
    assert any(e.model_path for e in entries)
    reg_path.unlink(missing_ok=True)
