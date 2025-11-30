import numpy as np

from foodspec.preprocess.normalization import AreaNormalizer, VectorNormalizer


def test_vector_normalizer_simple():
    X = np.array([[3.0, 4.0], [1.0, 1.0]])
    norm = VectorNormalizer(norm="l2")
    Xn = norm.transform(X)
    assert np.allclose(np.linalg.norm(Xn, axis=1), 1.0)


def test_area_normalizer_simple():
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    ar = AreaNormalizer()
    Xa = ar.transform(X)
    areas = np.trapezoid(Xa, axis=1, dx=1.0)
    assert np.allclose(areas, 1.0)
