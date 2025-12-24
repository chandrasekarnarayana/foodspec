from pathlib import Path
from types import SimpleNamespace
import textwrap

import pytest
from typer.testing import CliRunner

from foodspec import __version__
from foodspec import cli


runner = CliRunner()


def _write_exp(tmp_path: Path, dataset_path: Path) -> Path:
    yaml_content = textwrap.dedent(
        f"""
        dataset:
          path: {dataset_path}
          modality: raman
          schema:
            label_column: label
        preprocessing:
          preset: standard
        qc:
          method: robust_z
          thresholds:
            outlier_rate: 0.2
        features:
          preset: standard
        modeling:
          suite:
            - algorithm: rf
              params:
                n_estimators: 5
        reporting:
          targets: [metrics]
        outputs:
          base_dir: {tmp_path/'runs'}
        """
    ).strip()
    exp_path = tmp_path / "exp.yml"
    exp_path.write_text(yaml_content)
    return exp_path


class DummyFoodSpec:
    instances = []

    def __init__(self, source, modality="raman", **kwargs):
        self.source = source
        self.modality = modality
        self.bundle = SimpleNamespace(run_record=None)
        self.called = []
        DummyFoodSpec.instances.append(self)

    def qc(self, method="robust_z", threshold=None, **kwargs):
        self.called.append(("qc", method, threshold))
        return self

    def preprocess(self, preset="auto", **kwargs):
        self.called.append(("preprocess", preset, kwargs))
        return self

    def features(self, preset="standard", specs=None, **kwargs):
        self.called.append(("features", preset, specs))
        return self

    def train(self, algorithm="rf", label_column="label", cv_folds=5, **kwargs):
        self.called.append(("train", algorithm, label_column, cv_folds, kwargs))
        return self

    def export(self, path, formats=None):
        self.called.append(("export", Path(path)))
        return Path(path)


def test_run_exp_executes_pipeline(monkeypatch, tmp_path):
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text("x,y\n1,2\n")
    exp_path = _write_exp(tmp_path, dataset_path)

    DummyFoodSpec.instances.clear()
    monkeypatch.setattr(cli, "FoodSpec", DummyFoodSpec)

    result = runner.invoke(cli.app, ["run-exp", str(exp_path), "--output-dir", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert DummyFoodSpec.instances
    inst = DummyFoodSpec.instances[0]
    # Ensure pipeline steps executed
    assert any(call[0] == "preprocess" for call in inst.called)
    assert any(call[0] == "train" for call in inst.called)
    assert any(call[0] == "export" for call in inst.called)


def test_run_exp_dry_run(monkeypatch, tmp_path):
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text("x,y\n1,2\n")
    exp_path = _write_exp(tmp_path, dataset_path)

    # If FoodSpec is called, fail
    def _boom(*_, **__):
        raise AssertionError("FoodSpec should not be instantiated during dry-run")

    monkeypatch.setattr(cli, "FoodSpec", _boom)

    result = runner.invoke(cli.app, ["run-exp", str(exp_path), "--dry-run"])

    assert result.exit_code == 0
    assert "config_hash" in result.stdout
