import pandas as pd

from foodspec.model_lifecycle import FrozenModel, TrainablePipeline
from foodspec.preprocessing_pipeline import PreprocessingConfig
from foodspec.features.rq import PeakDefinition, RatioDefinition


def test_train_and_predict_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "oil_type": ["A", "A", "B", "B"],
            "matrix": ["oil"] * 4,
            "heating_stage": [0, 1, 0, 1],
            "I_1000": [10, 11, 5, 6],
            "I_1100": [8, 9, 4, 5],
        }
    )
    peaks = [
        PeakDefinition(name="I_1000", column="I_1000", wavenumber=1000),
        PeakDefinition(name="I_1100", column="I_1100", wavenumber=1100),
    ]
    ratios = [RatioDefinition(name="1000/1100", numerator="I_1000", denominator="I_1100")]
    pp = PreprocessingConfig()
    pipe = TrainablePipeline(label_col="oil_type", peaks=peaks, ratios=ratios, preprocess_config=pp, model_type="rf")
    frozen = pipe.train(df)
    save_path = tmp_path / "model"
    frozen.save(save_path)
    loaded = FrozenModel.load(save_path)
    preds = loaded.predict(df)
    assert "prediction" in preds.columns
    assert preds["prediction"].isin(["A", "B"]).all()
