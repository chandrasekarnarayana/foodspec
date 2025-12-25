import textwrap
from pathlib import Path

import pandas as pd

from foodspec.core.run_record import RunRecord, _hash_path
from foodspec.repro.diff import diff_runs
from foodspec.repro.experiment import ExperimentConfig


def _write_yaml(path: Path, dataset_path: Path, outputs_path: Path) -> None:
    content = textwrap.dedent(
        f"""
        dataset:
          path: {dataset_path}
          modality: raman
          schema:
            label_column: label
            wavenumber_column: wn
        preprocessing:
          preset: standard
        qc:
          method: robust_z
          thresholds:
            outlier_rate: 0.1
        features:
          preset: specs
          specs:
            - name: band_1
              ftype: band
              regions:
                - [100, 200]
        modeling:
          suite:
            - algorithm: rf
              params:
                n_estimators: 10
        reporting:
          targets: [metrics, diagnostics]
        seeds:
          numpy_seed: 123
        outputs:
          base_dir: {outputs_path}
        """
    ).strip()
    path.write_text(content)


def test_experiment_config_builds_run_record(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(data_path, index=False)
    outputs_path = tmp_path / "outputs"
    yaml_path = tmp_path / "exp.yml"
    _write_yaml(yaml_path, data_path, outputs_path)

    config = ExperimentConfig.from_yaml(yaml_path)
    record = config.build_run_record()

    assert record.dataset_hash == _hash_path(data_path)
    assert record.random_seeds == config.seeds
    assert str(outputs_path) in record.output_paths


def test_run_record_tracks_output_paths():
    record = RunRecord(
        workflow_name="exp",
        config={"k": 1},
        dataset_hash="hash",
        random_seeds={"numpy_seed": 999},
    )
    record.add_output_path("/tmp/out1")
    record.add_output_path("/tmp/out2")
    serialized = record.to_dict()

    assert serialized["output_paths"] == ["/tmp/out1", "/tmp/out2"]


def test_diff_runs_reports_changes():
    env = {"python_version": "3.12", "platform": "linux"}
    run_a = RunRecord("exp", {"key": "a"}, "hash_a", environment=env, random_seeds={"numpy_seed": 1})
    run_b = RunRecord("exp", {"key": "b"}, "hash_b", environment=env, random_seeds={"numpy_seed": 2})
    run_a.add_step("preprocess", "aaaa")
    run_b.add_step("preprocess", "bbbb")

    diff = diff_runs(run_a, run_b)
    fields = {c["field"] for c in diff["changes"]}

    assert "dataset_hash" in fields
    assert "config_hash" in fields
    assert any("Pipeline" in msg for msg in diff["expected_consequences"])
