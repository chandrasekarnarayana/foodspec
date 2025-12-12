import subprocess
import sys
from pathlib import Path

from foodspec.registry import FeatureModelRegistry, RegistryEntry


def test_cli_registry_query(tmp_path: Path):
    reg_path = tmp_path / "registry.json"
    reg = FeatureModelRegistry(reg_path)
    reg.register_run("runX", {"protocol": "p1", "protocol_version": "0.1", "inputs": [], "preprocessing": {}})
    reg.register_model("runX", "model.pkl", {"protocol_name": "p1", "protocol_version": "0.1", "features": [{"name": "I_1000"}]})

    cmd = [sys.executable, "-m", "foodspec.cli_registry", "--registry", str(reg_path), "--query-protocol", "p1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "p1" in result.stdout
