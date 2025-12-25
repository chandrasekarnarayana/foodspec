import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.features.bands import compute_band_features
from foodspec.features.peaks import PeakFeatureExtractor
from foodspec.features.metrics import feature_cv, discriminative_power, robustness_vs_variations


def _ds():
    x_axis = np.linspace(1000, 1100, 100)
    spectra = []
    labels = []
    for i in range(6):
        base = np.sin((x_axis - 1000) / 20.0) + 0.05 * np.random.randn(len(x_axis))
        # Add class-dependent peak at 1050 vs 1070
        if i < 3:
            base += 1.0 * np.exp(-0.5 * ((x_axis - 1050) / 2) ** 2)
            labels.append(0)
        else:
            base += 1.0 * np.exp(-0.5 * ((x_axis - 1070) / 2) ** 2)
            labels.append(1)
        spectra.append(base)
    spectra = np.vstack(spectra)
    meta = pd.DataFrame({"label": labels, "group": ["g1"] * 3 + ["g2"] * 3})
    return FoodSpectrumSet(x=spectra, wavenumbers=x_axis, metadata=meta, modality="raman", label_col="label", group_col="group")


def test_band_features_include_mean_max_slope():
    ds = _ds()
    bands = [("bandA", 1030, 1040)]
    df = compute_band_features(ds.x, ds.wavenumbers, bands, metrics=("integral", "mean", "max", "slope"))
    assert set(df.columns) == {"bandA_integral", "bandA_mean", "bandA_max", "bandA_slope"}


def test_peak_extractor_width_centroid_symmetry():
    ds = _ds()
    extractor = PeakFeatureExtractor([1050], tolerance=5.0, features=("height", "area", "width", "centroid", "symmetry"))
    feats = extractor.transform(ds.x, wavenumbers=ds.wavenumbers)
    cols = extractor.get_feature_names_out()
    assert "peak_1050_width" in cols
    assert "peak_1050_centroid" in cols
    assert "peak_1050_symmetry" in cols
    assert feats.shape[1] == len(cols)


def test_feature_metrics():
    ds = _ds()
    # simple feature: mean intensity per spectrum
    feat_df = pd.DataFrame({"mean": ds.x.mean(axis=1)})
    cv = feature_cv(feat_df)
    assert "mean" in cv.index
    disc = discriminative_power(feat_df, labels=ds.labels)
    assert "anova_f_mean" in disc
    robust = robustness_vs_variations([feat_df, feat_df + 0.01])
    assert robust <= 1.0
