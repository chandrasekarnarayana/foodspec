# Raman/FTIR preprocessing guide

Questions this page answers
- Why preprocess Raman/FTIR spectra?
- Which baseline, smoothing, normalization, and helpers should I use?
- How do I configure them in Python and CLI?

## Why preprocessing matters
- Baseline: removes fluorescence (Raman) or sloping background (FTIR).
- Smoothing: reduces noise while preserving peaks.
- Scatter/normalization: corrects intensity scaling, pathlength, and scatter differences.
- Derivatives: enhance subtle features and reduce baseline.
- FTIR/Raman helpers: handle ATR effects, atmospheric bands, and cosmic rays.

## Baseline correction
- **ALSBaseline**: general-purpose baseline removal; tune `lambda_`, `p`.
- **RubberbandBaseline**: convex-hull baseline; useful for concave backgrounds.
- **PolynomialBaseline**: low-degree fit for globally smooth baselines.

## Smoothing
- **Savitzky–Golay**: preserves peak shape; choose odd window length, polyorder < window.
- **MovingAverageSmoother**: simple denoising; may broaden peaks.

## Scatter & normalization
- **Vector/Area/Max normalizers**: scale spectra to unit norm/area; remove overall intensity differences.
- **SNVNormalizer**: subtract mean, divide by std per spectrum; removes additive/multiplicative scatter.
- **MSCNormalizer**: regress onto reference (mean spectrum) and correct slope/intercept; good for scatter variation.
- **InternalPeakNormalizer**: normalize to a stable internal band (mean intensity in a window).

## Derivatives
- **DerivativeTransformer**: Savitzky–Golay derivatives (1st/2nd) to emphasize subtle bands and suppress baseline.

## FTIR/Raman-specific helpers
- **AtmosphericCorrector**: subtracts water/CO₂ components (FTIR).
- **SimpleATRCorrector**: heuristic ATR depth correction (FTIR).
- **CosmicRayRemover**: detects/replaces spikes (Raman).

## Example pipeline (Python)
```python
from sklearn.pipeline import Pipeline
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.preprocess.cropping import RangeCropper

pipe = Pipeline([
    ("als", ALSBaseline(lambda_=1e5, p=0.01, max_iter=10)),
    ("savgol", SavitzkyGolaySmoother(window_length=9, polyorder=3)),
    ("norm", VectorNormalizer(norm="l2")),
    ("crop", RangeCropper(min_wn=600, max_wn=1800)),
])
X_proc = pipe.fit_transform(fs.x)
```

## CLI usage
Most workflows (oil-auth, heating) include defaults. For raw folders:
```bash
foodspec preprocess raw_folder out.h5 --modality raman --min-wn 600 --max-wn 1800
```

See also
- `preprocessing_guide.md`
- `oil_auth_tutorial.md`
- `heating_tutorial.md`
- `keyword_index.md`
