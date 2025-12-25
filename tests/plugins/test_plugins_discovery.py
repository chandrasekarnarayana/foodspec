import sys
from pathlib import Path

import pandas as pd
import pytest


class _EP:
    def __init__(self, name: str, module: str):
        self.name = name
        self.module = module

    def load(self):
        return __import__(self.module)


class _EPs:
    def __init__(self, entries):
        self._entries = entries

    def select(self, group: str):
        if group == "foodspec.plugins":
            return self._entries
        return []


@pytest.fixture(scope="module", autouse=True)
def add_example_plugins_to_syspath():
    # Add example plugin package roots to sys.path for import
    roots = [
        Path("examples/plugins/plugin_example_vendor").resolve(),
        Path("examples/plugins/plugin_example_indices").resolve(),
        Path("examples/plugins/plugin_example_workflow").resolve(),
    ]
    for r in roots:
        sys.path.append(str(r))
    yield
    for r in roots:
        try:
            sys.path.remove(str(r))
        except ValueError:
            pass


def test_plugin_discovery_loads_indices_workflows_and_vendor(monkeypatch):
    from foodspec.plugins import load_plugins

    entries = [
        _EP("plugin_example_vendor", "plugin_example_vendor"),
        _EP("plugin_example_indices", "plugin_example_indices"),
        _EP("plugin_example_workflow", "plugin_example_workflow"),
    ]

    def _fake_entry_points():
        return _EPs(entries)

    monkeypatch.setattr("importlib.metadata.entry_points", _fake_entry_points, raising=False)

    pm = load_plugins(force=True)
    assert "toy" in pm.vendor_loaders, "Vendor loader 'toy' not discovered"
    assert "demo_index" in pm.feature_indices, "Feature index 'demo_index' not discovered"
    assert "demo_workflow" in pm.workflows, "Workflow 'demo_workflow' not discovered"


def test_vendor_routing_toy_suffix(monkeypatch, tmp_path):
    # Stub entry points as above
    entries = [
        _EP("plugin_example_vendor", "plugin_example_vendor"),
    ]

    def _fake_entry_points():
        return _EPs(entries)

    monkeypatch.setattr("importlib.metadata.entry_points", _fake_entry_points, raising=False)

    # Create a minimal .toy file that the plugin expects (CSV columns: meta,int1,int2)
    toy_path = tmp_path / "sample.toy"
    df = pd.DataFrame(
        {
            "meta": ["A", "B"],
            "int1": [1.0, 3.0],
            "int2": [2.0, 4.0],
            "int3": [5.0, 6.0],
        }
    )
    df.to_csv(toy_path, index=False)

    from foodspec.io.ingest import load_vendor

    res = load_vendor(str(toy_path))
    ds = res.dataset
    # Verify shape and axis metrics exist
    assert ds.x.shape == (2, 3)
    assert len(ds.wavenumbers) == 3
    assert res.metrics["total_spectra"] == 2
    assert res.diagnostics["format"] in {"toy", "unknown"}
