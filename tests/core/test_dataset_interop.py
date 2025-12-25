import numpy as np

from foodspec.core.dataset import FoodSpectrumSet, from_sklearn, to_sklearn


def test_from_sklearn_creates_dataset_with_labels_and_wavenumbers():
    X = np.arange(15.0).reshape(5, 3)
    y = np.array(["A", "B", "A", "B", "A"])
    wn = [100.0, 200.0, 300.0]

    ds = from_sklearn(X, y=y, wavenumbers=wn, modality="raman", labels_name="class")

    assert isinstance(ds, FoodSpectrumSet)
    assert ds.x.shape == (5, 3)
    assert ds.wavenumbers.shape == (3,)
    np.testing.assert_allclose(ds.wavenumbers, np.array(wn))

    # metadata contains labels and aligns with samples
    assert ds.metadata.shape[0] == 5
    assert "class" in ds.metadata.columns
    assert ds.label_col == "class"
    np.testing.assert_array_equal(ds.metadata["class"].to_numpy(), y)


def test_to_sklearn_roundtrip_matches_X_and_y():
    X = np.random.RandomState(42).randn(4, 3)
    y = np.array([1, 0, 1, 0])

    ds = from_sklearn(X, y=y, modality="raman", labels_name="target")
    X2, y2 = to_sklearn(ds)

    np.testing.assert_allclose(X2, X)
    np.testing.assert_array_equal(y2, y)
