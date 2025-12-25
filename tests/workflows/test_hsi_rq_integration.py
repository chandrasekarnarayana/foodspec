import numpy as np
import pandas as pd

from foodspec.core.spectral_dataset import HyperspectralDataset
from foodspec.protocol import ProtocolConfig, ProtocolRunner


def test_hsi_segment_and_roi_feeds_rq(tmp_path):
    # Synthetic cube: two regions with different intensities
    wns = np.array([1000.0, 1100.0, 1200.0])
    cube = np.zeros((4, 4, 3))
    cube[:2, :, :] = 1.0  # region A
    cube[2:, :, :] = 5.0  # region B
    meta = pd.DataFrame({"pixel_id": np.arange(cube.shape[0] * cube.shape[1])})
    hsi = HyperspectralDataset.from_cube(cube, wns, meta)

    proto = {
        "name": "hsi_proto",
        "steps": [
            {"type": "hsi_segment", "params": {"method": "kmeans", "n_clusters": 2}},
            {
                "type": "hsi_roi_to_1d",
                "params": {
                    "peaks": [
                        {"name": "I_1000", "column": "I_1000", "wavenumber": 1000.0},
                        {"name": "I_1100", "column": "I_1100", "wavenumber": 1100.0},
                    ],
                    "ratios": [{"name": "1000/1100", "numerator": "I_1000", "denominator": "I_1100"}],
                    "run_rq": True,
                    "oil_col": "roi_label",
                },
            },
            {"type": "output", "params": {"output_dir": str(tmp_path / "runs")}},
        ],
    }
    cfg = ProtocolConfig.from_dict(proto)
    runner = ProtocolRunner(cfg)
    res = runner.run([hsi])

    assert "hsi_label_counts" in res.tables
    assert "hsi_roi_peaks" in res.tables
    assert res.tables["hsi_roi_peaks"]["roi_label"].nunique() == 2
