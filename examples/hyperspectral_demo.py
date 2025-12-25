"""
Hyperspectral demo: load cube, preprocess, segment, extract ROI spectra, run RQ.
"""
import numpy as np
import pandas as pd

from foodspec.core.spectral_dataset import HyperspectralDataset, PreprocessingConfig
from foodspec.features.rq import PeakDefinition, RatioDefinition, RatioQualityEngine, RQConfig


def main():
    # Synthetic cube (y=5, x=4, wn=3 for brevity)
    y, x, wn_len = 5, 4, 3
    wn = np.array([1000, 1100, 1200], dtype=float)
    cube = np.random.rand(y, x, wn_len)
    meta = pd.DataFrame({"y": np.repeat(np.arange(y), x), "x": np.tile(np.arange(x), y)})
    hsi = HyperspectralDataset.from_cube(cube, wn, metadata=meta)

    # Preprocess
    hsi_proc = hsi.preprocess(PreprocessingConfig(normalization="vector", smoothing_method="moving_average", smoothing_window=3))

    # Segment
    labels = hsi_proc.segment(method="kmeans", n_clusters=2)

    # Extract ROI spectra per label
    roi_spectra = []
    for k in np.unique(labels):
        mask = (labels == k)
        roi_ds = hsi_proc.roi_spectrum(mask)
        roi_spectra.append(roi_ds)

    # Combine ROI spectra into a peak table
    peaks = [PeakDefinition(name=f"I_{int(wn_i)}", column=f"I_{int(wn_i)}", wavenumber=float(wn_i)) for wn_i in wn]
    ratios = [RatioDefinition(name=f"I_{int(wn[0])}/I_{int(wn[1])}", numerator=f"I_{int(wn[0])}", denominator=f"I_{int(wn[1])}")]

    dfs = []
    for idx, roi_ds in enumerate(roi_spectra):
        df_peaks = roi_ds.to_peaks(peaks)
        df_peaks["oil_type"] = f"segment_{idx}"
        dfs.append(df_peaks)
    peak_df = pd.concat(dfs, ignore_index=True)

    cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage")
    res = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg).run_all(peak_df)
    print("=== ROI RQ report (truncated) ===")
    print("\n".join(res.text_report.splitlines()[:20]))


if __name__ == "__main__":
    main()
