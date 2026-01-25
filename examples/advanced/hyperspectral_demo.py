"""
Hyperspectral demo: load cube, preprocess, segment, extract ROI spectra, run RQ.

Teaching Goal:
    Demonstrate hyperspectral image processing workflow with:
    - 3D cube loading and preprocessing
    - Spatial segmentation (K-means clustering)
    - ROI spectrum extraction and aggregation
    - Ratio Quality (RQ) analysis on ROI-level data
    - Spatial label map visualization
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foodspec.core.spectral_dataset import HyperspectralDataset, PreprocessingConfig
from foodspec.features.rq import PeakDefinition, RatioDefinition, RatioQualityEngine, RQConfig


def main():
    out_dir = Path("outputs/hyperspectral_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("HYPERSPECTRAL MAPPING: Spatial Analysis Example")
    print("=" * 70)

    # Synthetic cube (y=5, x=4, wn=3 for brevity)
    print("\n1. Creating synthetic hyperspectral cube...")
    y, x, wn_len = 5, 4, 3
    wn = np.array([1000, 1100, 1200], dtype=float)
    cube = np.random.rand(y, x, wn_len)
    meta = pd.DataFrame({"y": np.repeat(np.arange(y), x), "x": np.tile(np.arange(x), y)})
    hsi = HyperspectralDataset.from_cube(cube, wn, metadata=meta)
    print(f"   Cube shape: {y} (y) × {x} (x) × {wn_len} (wavenumbers)")

    # Segment (work directly on HyperspectralDataset)
    print("\n2. Segmenting image with K-means (k=2)...")
    labels = hsi.segment(method="kmeans", n_clusters=2)
    print(f"   ✓ Segmentation complete: found {len(np.unique(labels))} clusters")

    # Extract ROI spectra per label
    print("\n3. Extracting ROI spectra...")
    roi_spectra = []
    for k in np.unique(labels):
        mask = (labels == k)
        roi_ds = hsi.roi_spectrum(mask)
        roi_spectra.append(roi_ds)
        print(f"   ROI {k}: {mask.sum()} pixels → mean spectrum extracted")

    # Combine ROI spectra into a peak table
    print("\n4. Extracting peaks and running RQ analysis...")
    peaks = [
        PeakDefinition(name=f"I_{int(wn_i)}", column=f"I_{int(wn_i)}", wavenumber=float(wn_i))
        for wn_i in wn
    ]
    ratios = [
        RatioDefinition(
            name=f"I_{int(wn[0])}/I_{int(wn[1])}",
            numerator=f"I_{int(wn[0])}",
            denominator=f"I_{int(wn[1])}"
        )
    ]

    dfs = []
    for idx, roi_ds in enumerate(roi_spectra):
        df_peaks = roi_ds.to_peaks(peaks)
        df_peaks["oil_type"] = f"segment_{idx}"
        dfs.append(df_peaks)
    peak_df = pd.concat(dfs, ignore_index=True)

    cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage")
    res = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg).run_all(peak_df)

    print("\n5. RQ Report (first 20 lines):") 
    print("\n".join(res.text_report.splitlines()[:20]))

    # Visualize label map
    print(f"\n6. Generating segmentation map...")
    fig, ax = plt.subplots(figsize=(6, 5))
    label_map = labels.reshape(y, x)
    im = ax.imshow(label_map, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Hyperspectral Segmentation Map (K-means clusters)")
    plt.colorbar(im, ax=ax, label="Cluster ID")
    fig.savefig(out_dir / "hsi_label_map.png", dpi=150, bbox_inches="tight")
    print(f"   ✓ Saved: {out_dir / 'hsi_label_map.png'}")

    # Visualize ROI spectra
    print(f"\n7. Generating ROI spectra plot...")
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, roi_ds in enumerate(roi_spectra):
        spectrum = roi_ds.spectra.squeeze()
        ax.plot(roi_ds.wavenumbers, spectrum, label=f"ROI {idx}", linewidth=2, alpha=0.7)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Mean Spectra by Segmented ROI")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "roi_spectra.png", dpi=150, bbox_inches="tight")
    print(f"   ✓ Saved: {out_dir / 'roi_spectra.png'}")

    plt.close("all")

    print("\n" + "=" * 70)
    print(f"✓ All outputs saved to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
