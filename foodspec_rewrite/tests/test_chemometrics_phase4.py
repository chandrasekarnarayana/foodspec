import numpy as np
import pytest

from foodspec.features.chemometrics import PCAFeatureExtractor, PLSFeatureExtractor


def make_data(seed=42, n_train=60, n_test=25, n_features=20):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_reg = rng.normal(size=(n_train,))
    y_cls = rng.choice(["classA", "classB", "classC"], size=n_train)
    return X_train, X_test, y_reg, y_cls


class TestPCAFeatureExtractor:
    def test_requires_fit(self):
        X_train, X_test, y_reg, y_cls = make_data()
        pca = PCAFeatureExtractor(n_components=3, seed=0)
        with pytest.raises(RuntimeError):
            _ = pca.transform(X_test)

    def test_fit_transform_deterministic_and_shapes(self):
        X_train, X_test, y_reg, y_cls = make_data(seed=0)
        pca = PCAFeatureExtractor(n_components=5, seed=123, whiten=False)
        pca.fit(X_train)
        Z1, names1 = pca.transform(X_test)
        Z2, names2 = pca.transform(X_test)
        assert Z1.shape == (X_test.shape[0], 5)
        assert names1 == [f"pca_{i+1}" for i in range(5)]
        assert np.allclose(Z1, Z2)
        # explained variance ratio exists and has correct length
        assert hasattr(pca, "explained_variance_ratio_")
        assert pca.explained_variance_ratio_.shape[0] == 5


class TestPLSFeatureExtractor:
    def test_requires_fit(self):
        X_train, X_test, y_reg, y_cls = make_data()
        pls = PLSFeatureExtractor(n_components=2, mode="regression")
        with pytest.raises(RuntimeError):
            _ = pls.transform(X_test)

    def test_pls_regression_deterministic_shapes(self):
        X_train, X_test, y_reg, y_cls = make_data(seed=1)
        pls = PLSFeatureExtractor(n_components=3, mode="regression")
        pls.fit(X_train, y_reg)
        Zt1, names1 = pls.transform(X_test)
        Zt2, names2 = pls.transform(X_test)
        assert Zt1.shape == (X_test.shape[0], 3)
        assert names1 == [f"pls_{i+1}" for i in range(3)]
        assert np.allclose(Zt1, Zt2)

    def test_pls_classification_label_encoding(self):
        X_train, X_test, y_reg, y_cls = make_data(seed=2)
        pls = PLSFeatureExtractor(n_components=2, mode="classification")
        pls.fit(X_train, y_cls)
        Zt, names = pls.transform(X_test)
        assert Zt.shape == (X_test.shape[0], 2)
        assert names == ["pls_1", "pls_2"]

    def test_pls_regression_non_numeric_y_errors(self):
        X_train, X_test, y_reg, y_cls = make_data(seed=3)
        pls = PLSFeatureExtractor(n_components=2, mode="regression")
        with pytest.raises(ValueError):
            pls.fit(X_train, y_cls)
