import numpy as np

from foodspec.data_objects import Spectrum


def test_spectrum_basic():
    x = np.linspace(400, 1800, 10)
    y = np.linspace(1, 2, 10)
    spec = Spectrum(x=x, y=y, kind="raman", metadata={"sample_id": "s1"})

    assert spec.n_points == 10
    assert spec.metadata["sample_id"] == "s1"
    norm = spec.normalize("vector")
    assert np.isclose(np.linalg.norm(norm.y), 1.0)

