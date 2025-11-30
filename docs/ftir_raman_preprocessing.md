# FTIR and Raman Preprocessing (simplified)

These utilities provide basic corrections for FTIR atmospheric components, ATR effects, and Raman cosmic-ray spikes. They are simplified and intended as starting points, not vendor-grade implementations.

## Atmospheric and ATR (FTIR)

```python
from foodspec.preprocess.ftir import AtmosphericCorrector, SimpleATRCorrector

corr = AtmosphericCorrector()
corr.fit(X, wavenumbers=wn)
X_corr = corr.transform(X)

atr = SimpleATRCorrector()
atr.fit(X_corr, wavenumbers=wn)
X_atr = atr.transform(X_corr)
```

- `AtmosphericCorrector` builds synthetic water/CO2 bases and subtracts scaled contributions via least squares.
- `SimpleATRCorrector` applies an approximate wavelength-dependent scaling based on refractive index ratio.

## Cosmic-ray removal (Raman)

```python
from foodspec.preprocess.raman import CosmicRayRemover

cr = CosmicRayRemover()
X_clean = cr.fit_transform(X_raman)
```

`CosmicRayRemover` detects spikes relative to a local median/MAD and interpolates neighbors.

**Note:** These are simplified algorithms meant for lightweight correction; for production or regulatory-grade workflows, further refinement and validation are recommended.

## Wavenumber-aware usage

Some transformers need the wavenumber axis. Use `set_wavenumbers` before placing them in a pipeline:

```python
from foodspec.preprocess.ftir import AtmosphericCorrector, SimpleATRCorrector
from sklearn.pipeline import Pipeline

corr = AtmosphericCorrector().set_wavenumbers(wn)
atr = SimpleATRCorrector().set_wavenumbers(wn)

pipe = Pipeline([("atm", corr), ("atr", atr)])
X_corr = pipe.fit_transform(X)
# now corr/atr can be part of a sklearn Pipeline without passing wavenumbers each call
```
