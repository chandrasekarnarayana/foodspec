# Hyperspectral Mapping (Teaching Walkthrough)

**Focus**: 3D cube • Segmentation • ROI spectra

## What you will learn
- Represent a hyperspectral cube (H × W × λ)
- Segment pixels into healthy vs. defective regions
- Extract mean spectra per region and compare quality

## Prerequisites
- Python 3.10+
- numpy, matplotlib, scikit-learn

## Minimal runnable code
```python
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(42)
H, W, L = 50, 50, 1500
w = np.linspace(800, 3000, L)
base = np.exp(-((w-1600)/200)**2)
cube = np.tile(base, (H, W, 1)) + 0.02*np.random.randn(H, W, L)
cube[20:35, 15:30, :] *= 0.6  # bruised patch

mean_img = cube.mean(axis=2).reshape(-1, 1)
labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(mean_img)
labels = labels.reshape(H, W)

healthy = cube[labels == 0].mean(axis=0)
bruised = cube[labels == 1].mean(axis=0)
ratio = bruised.max() / healthy.max()
print({"pixels_healthy": int((labels==0).sum()), "pixels_bruised": int((labels==1).sum()), "quality_ratio": float(ratio)})
```

## Explain the outputs
- `pixels_healthy` vs `pixels_bruised` ⇒ segmentation balance
- `quality_ratio` < 1 ⇒ bruised region has lower intensity (degradation)
- Mean spectra per region reveal feature shifts due to defects

## Full resources
- Full script: https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/hyperspectral_demo.py
- Teaching notebook (download/run): https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/04_hyperspectral_mapping_teaching.ipynb
- Example figure: https://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/hyperspectral_demo/hsi_label_map.png

## Run it yourself
```bash
python examples/hyperspectral_demo.py
jupyter notebook examples/tutorials/04_hyperspectral_mapping_teaching.ipynb
```

## Related docs
- Workflow: hyperspectral mapping → ../workflows/spatial/hyperspectral_mapping.md
