import subprocess
import sys


def test_cli_plugin_list_runs():
    cmd = [sys.executable, "-m", "foodspec.cli_plugin", "list"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Should not error; may list zero plugins
    assert result.returncode == 0
