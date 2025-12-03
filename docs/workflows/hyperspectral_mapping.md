# Workflow: Hyperspectral Mapping

> New to workflow design? See [Designing & reporting workflows](workflow_design_and_reporting.md). For model/metric choices, see [ML & DL models](../ml/models_and_best_practices.md) and [Metrics & evaluation](../metrics/metrics_and_evaluation.md).
> For troubleshooting (SNR, misalignment), see [Common problems & solutions](../troubleshooting/common_problems_and_solutions.md).

Hyperspectral mapping treats each pixel as a spectrum; goals include spatial localization of components, contaminants, or quality gradients.

Suggested visuals: intensity/ratio maps, cluster maps, pixel-level spectra, PCA score maps.

```mermaid
flowchart LR
  A[Flat spectra (pixels)] --> B[Preprocessing (baseline, norm, crop)]
  B --> C[Cube rebuild (HyperSpectralCube)]
  C --> D[Feature extraction (ratios/PCs)]
  D --> E[Clustering/classification]
  E --> F[Maps & metrics (IoU/accuracy), reports]
```

## 1. Problem and dataset
- **Use cases:** spatial adulteration, contamination localization, tissue/structure mapping.
- **Inputs:** per-pixel spectra flattened to rows; wavenumber axis; image_shape metadata.
- **Typical size:** thousands of pixels; consider subsampling for development.

## 2. Pipeline (default)
- Preprocess per spectrum (baseline, normalization).
- Rebuild cube with `HyperSpectralCube.from_spectrum_set`.
- Extract ratios or PCs per pixel.
- Segment via k-means/thresholds or classify with trained model.

## 3. Python sketch
```python
from foodspec.core.hyperspectral import HyperSpectralCube
from foodspec.viz.hyperspectral import plot_hyperspectral_intensity_map
from foodspec.viz import plot_correlation_heatmap

cube = HyperSpectralCube.from_spectrum_set(fs_pixels, image_shape=(h, w))
fig = plot_hyperspectral_intensity_map(cube, target_wavenumber=1655, window=5)
```

## 4. Metrics & interpretation
- If labeled masks exist: pixel-wise accuracy/IoU; confusion matrix on pixel labels.
- For unsupervised segments: stability across runs; correlation of ratio maps vs reference metrics.
- Inspect edge artifacts and spatial smoothness.

## Summary
- Preprocess → rebuild cube → ratios/PCs → clustering/classification → maps + metrics.
- Pair maps with pixel spectra and QC plots; report preprocessing and class definitions.
