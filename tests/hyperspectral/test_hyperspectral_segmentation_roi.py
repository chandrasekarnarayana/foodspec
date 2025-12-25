import numpy as np
import pandas as pd

from foodspec.core.spectral_dataset import HyperspectralDataset


def test_hyperspectral_segmentation_kmeans_labels_shape():
    rng = np.random.RandomState(21)
    wn = np.linspace(600.0, 1600.0, 32)
    cube = rng.randn(10, 12, wn.size)
    meta = pd.DataFrame({"scan": ["S1"]})

    h = HyperspectralDataset.from_cube(cube, wn, meta, {"protocol_name": "seg_demo"})
    labels = h.segment(method="kmeans", n_clusters=3)

    assert h.label_map.shape == h.shape_xy
    assert labels.shape == h.shape_xy
    # labels in expected range
    assert labels.min() >= 0 and labels.max() < 3


def test_hyperspectral_roi_spectrum_and_projection_band():
    rng = np.random.RandomState(22)
    wn = np.linspace(500.0, 1500.0, 40)
    cube = rng.randn(8, 5, wn.size)
    meta = pd.DataFrame({"scan": ["S2"]})

    h = HyperspectralDataset.from_cube(cube, wn, meta, {"protocol_name": "roi_demo"})
    # Mask: select half of pixels
    mask = np.zeros(h.shape_xy, dtype=bool)
    mask[:4, :] = True

    roi_ds = h.roi_spectrum(mask)
    assert roi_ds.spectra.shape == (1, wn.size)
    assert roi_ds.metadata["roi_pixels"][0] == 4 * h.shape_xy[1]

    # Band projection
    band = (700.0, 900.0)
    proj = h.projection(mode="band", band=band)
    assert proj.shape == h.shape_xy
