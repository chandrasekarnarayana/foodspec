import numpy as np
import pandas as pd

from foodspec.protocol import ProtocolConfig, ProtocolRunner
from foodspec.core.spectral_dataset import HyperspectralDataset


def test_protocol_hsi_segment_and_roi_to_1d():
    wns = np.array([1000.0, 1100.0])
    cube = np.array(
        [
            [[1.0, 2.0], [1.1, 2.1]],
            [[5.0, 6.0], [5.1, 6.1]],
        ]
    )  # shape (2,2,2)
    meta = pd.DataFrame({"pixel_id": np.arange(4)})
    hsi = HyperspectralDataset.from_cube(cube, wns, meta)

    proto = {
        "name": "hsi_roi_demo",
        "steps": [
            {"type": "hsi_segment", "params": {"method": "kmeans", "n_clusters": 2}},
            {
                "type": "hsi_roi_to_1d",
                "params": {
                    "peaks": [
                        {"name": "I_1000", "column": "I_1000", "wavenumber": 1000.0},
                        {"name": "I_1100", "column": "I_1100", "wavenumber": 1100.0},
                    ]
                },
            },
        ],
    }
    cfg = ProtocolConfig.from_dict(proto)
    runner = ProtocolRunner(cfg)
    res = runner.run([hsi])

    assert "hsi_label_counts" in res.tables
    assert "hsi_roi_peaks" in res.tables
    assert any("hsi" in name for name in res.figures)
    assert res.tables["hsi_roi_peaks"].shape[0] >= 2
