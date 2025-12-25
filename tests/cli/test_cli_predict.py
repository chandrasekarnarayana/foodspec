import subprocess
import sys
from pathlib import Path

import pandas as pd

from foodspec.features.rq import PeakDefinition, RatioDefinition
from foodspec.model_lifecycle import TrainablePipeline
from foodspec.preprocessing_pipeline import PreprocessingConfig


def test_cli_predict(tmp_path: Path):
    # Train and save a tiny model
    df = pd.DataFrame({"oil_type": ["A", "B"], "I_1000": [1.0, 2.0], "I_1100": [1.0, 2.0]})
    peaks = [
        PeakDefinition(name="I_1000", column="I_1000", wavenumber=1000),
        PeakDefinition(name="I_1100", column="I_1100", wavenumber=1100),
    ]
    ratios = [RatioDefinition(name="1000/1100", numerator="I_1000", denominator="I_1100")]
    pipe = TrainablePipeline("oil_type", peaks, ratios, PreprocessingConfig(), model_type="rf")
    frozen = pipe.train(df)
    model_prefix = tmp_path / "toy_model"
    frozen.save(model_prefix)

    # Prediction input
    df_new = pd.DataFrame({"I_1000": [1.5], "I_1100": [1.5]})
    data_path = tmp_path / "data.csv"
    df_new.to_csv(data_path, index=False)
    out_path = tmp_path / "preds.csv"

    cmd = [
        sys.executable,
        "-m",
        "foodspec.cli_predict",
        "--model",
        str(model_prefix),
        "--input",
        str(data_path),
        "--output",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
