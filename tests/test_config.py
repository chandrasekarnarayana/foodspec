import json
from pathlib import Path

import pytest

from foodspec.config import load_config, merge_cli_overrides


def test_load_config_yaml_and_json(tmp_path: Path):
    yml = tmp_path / "config.yml"
    yml.write_text("input_hdf5: a.h5\ncv_splits: 5\n", encoding="utf-8")
    js = tmp_path / "config.json"
    js.write_text(json.dumps({"input_hdf5": "b.h5", "cv_splits": 3}), encoding="utf-8")

    cfg_yml = load_config(yml)
    cfg_js = load_config(js)
    assert cfg_yml == {"input_hdf5": "a.h5", "cv_splits": 5}
    assert cfg_js == {"input_hdf5": "b.h5", "cv_splits": 3}


def test_load_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_bad_extension(tmp_path: Path):
    bad = tmp_path / "config.txt"
    bad.write_text("foo", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_merge_cli_overrides_nested():
    base = {"a": 1, "b": {"inner": 10, "keep": "x"}}
    overrides = {"b": {"inner": 20}}
    merged = merge_cli_overrides(base, overrides)
    assert merged["b"]["inner"] == 20
    assert merged["b"]["keep"] == "x"
