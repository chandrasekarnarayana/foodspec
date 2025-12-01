import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from foodspec.core.hyperspectral import HyperSpectralCube
from foodspec.viz.hyperspectral import plot_cluster_map, plot_ratio_map


def _make_cube():
    h, w = 2, 2
    wn = np.linspace(1000, 1010, 5)
    cube = np.random.rand(h, w, wn.size)
    meta = pd.DataFrame({"sample_id": [f"p{i}" for i in range(h * w)]})
    return HyperSpectralCube(cube=cube, wavenumbers=wn, metadata=meta, image_shape=(h, w))


def test_to_pixel_spectra_and_labels():
    cube = _make_cube()
    ds = cube.to_pixel_spectra()
    assert ds.x.shape[0] == 4
    labels = np.array([0, 1, 2, 3])
    img = cube.from_pixel_labels(labels)
    assert img.shape == (2, 2)
    assert img[0, 0] == 0


def test_ratio_and_cluster_plots():
    cube = _make_cube()
    fig1, ax1 = plt.subplots()
    plot_ratio_map(cube, num1=1002.0, num2=1005.0, ax=ax1, window=2.5)
    plt.close(fig1)
    label_img = np.array([[0, 1], [1, 0]])
    fig2, ax2 = plt.subplots()
    plot_cluster_map(label_img, ax=ax2)
    plt.close(fig2)


def test_pixel_roundtrip_labels():
    cube = _make_cube()
    fs = cube.to_pixel_spectra()
    labels = np.array([0, 1, 2, 3])
    img = cube.from_pixel_labels(labels)
    assert img.shape == (2, 2)
    assert np.array_equal(img.ravel(), labels)
    assert fs.metadata["row"].tolist() == [0, 0, 1, 1]
    assert fs.metadata["col"].tolist() == [0, 1, 0, 1]
