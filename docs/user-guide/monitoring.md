# Streaming Monitoring

FoodSpec provides a lightweight streaming API for real-time QC and drift tracking.
It consumes batches of spectra and emits alerts based on health, outlier rate, and drift.

Example

```python
import numpy as np
from foodspec.monitoring import StreamingMonitor

monitor = StreamingMonitor(drift_method="pca_cusum", outlier_method="robust_z")
batch = np.random.randn(8, 200)
event = monitor.update(batch)
print(event.alerts)
```

What it checks

- Health scores from spectral QC
- Outlier rate using robust or model-based detectors
- Drift detection via PCA-CUSUM or batch shift methods
