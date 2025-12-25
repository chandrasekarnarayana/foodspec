import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.features.specs import FeatureEngine, FeatureSpec


def _ds():
    x_axis = np.linspace(1000, 1100, 50)
    spectra = []
    for i in range(4):
        y = 0.1 * i + np.sin((x_axis - 1000) / 20.0)
        # add peak near 1050
        y += np.exp(-0.5 * ((x_axis - 1050) / 2) ** 2)
        spectra.append(y)
    spectra = np.vstack(spectra)
    meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(4)], "batch_id": ["b1", "b1", "b2", "b2"]})
    return FoodSpectrumSet(x=spectra, wavenumbers=x_axis, metadata=meta, modality="raman", batch_col="batch_id")


def test_band_and_peak_specs():
    ds = _ds()
    specs = [
        FeatureSpec(name="band_area", ftype="band", regions=[(1030, 1060)]),
        FeatureSpec(name="peak1050", ftype="peak", regions=[(1048, 1052)], params={"tolerance": 3.0, "metrics": ("height",)}),
    ]
    engine = FeatureEngine(specs)
    feats, diag = engine.evaluate(ds)
    assert "band_area" in feats.columns
    assert any(col.startswith("peak_1050") for col in feats.columns)
    assert diag["applied"]


def test_ratio_spec_uses_previous_features():
    ds = _ds()
    specs = [
        FeatureSpec(name="band_a", ftype="band", regions=[(1030, 1040)]),
        FeatureSpec(name="band_b", ftype="band", regions=[(1060, 1070)]),
        FeatureSpec(name="ratio_ab", ftype="ratio", formula="band_a / band_b"),
    ]
    feats, _ = FeatureEngine(specs).evaluate(ds)
    assert "ratio_ab" in feats.columns
    assert np.isfinite(feats["ratio_ab"]).all()


def test_constraints_modality_enforced():
    ds = _ds()
    specs = [FeatureSpec(name="only_nir", ftype="band", regions=[(1030, 1040)], constraints={"modality": "nir"})]
    engine = FeatureEngine(specs)
    try:
        engine.evaluate(ds)
    except ValueError:
        return
    assert False, "Expected modality constraint to raise"
