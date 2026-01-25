import numpy as np
import pytest

from foodspec.features import FeatureUnion, PCAFeatureExtractor, PLSFeatureExtractor


def make_data(seed=123, n_train=60, n_test=25, n_features=30):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_reg = rng.normal(size=(n_train,))
    y_cls = rng.choice([0, 1], size=n_train)
    return X_train, X_test, y_reg, y_cls


class TestFeatureUnion:
    def test_concatenation_and_names_prefix(self):
        X_train, X_test, y_reg, y_cls = make_data(seed=0)
        pca = PCAFeatureExtractor(n_components=4, seed=42)
        pls = PLSFeatureExtractor(n_components=3, mode="regression")
        union = FeatureUnion([pca, pls], prefix=True)

        union.fit(X_train, y_reg)
        fs = union.transform(X_test)

        # Shapes: sum of component counts
        assert fs.Xf.shape == (X_test.shape[0], 7)
        # Names: prefixed and unique
        assert fs.feature_names[:4] == [f"pca:pca_{i+1}" for i in range(4)]
        assert fs.feature_names[4:] == [f"pls:pls_{i+1}" for i in range(3)]
        assert len(set(fs.feature_names)) == 7
        # Meta: per-feature extractors aligned
        assert fs.feature_meta["per_feature_extractors"] == ["pca"] * 4 + ["pls"] * 3
        assert fs.feature_meta["extractor_names"] == ["pca", "pls"]

    def test_stable_order_and_deterministic_outputs(self):
        X_train, X_test, y_reg, y_cls = make_data(seed=1)
        pca = PCAFeatureExtractor(n_components=2, seed=0)
        pls = PLSFeatureExtractor(n_components=2, mode="regression")
        union = FeatureUnion([pca, pls], prefix=True)

        union.fit(X_train, y_reg)
        fs1 = union.transform(X_test)
        fs2 = union.transform(X_test)

        # Determinism: same transform twice yields same results
        assert np.allclose(fs1.Xf, fs2.Xf)
        # Stable ordering across runs
        assert fs1.feature_names == fs2.feature_names
        # Names aligned to columns
        assert fs1.n_features == len(fs1.feature_names)

    def test_requires_fit(self):
        X_train, X_test, y_reg, y_cls = make_data(seed=2)
        pca = PCAFeatureExtractor(n_components=2, seed=0)
        pls = PLSFeatureExtractor(n_components=1, mode="regression")
        union = FeatureUnion([pca, pls])
        with pytest.raises(RuntimeError):
            _ = union.transform(X_test)
