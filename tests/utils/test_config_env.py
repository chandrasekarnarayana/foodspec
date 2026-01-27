from __future__ import annotations

import json

from foodspec.check_env import check_env
from foodspec.config import load_config


def test_load_config_json(tmp_path):
    cfg = {"foo": "bar", "nested": {"x": 1}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["foo"] == "bar"
    assert loaded["nested"]["x"] == 1


def test_check_env_returns_text():
    text = check_env()
    assert "Python:" in text
