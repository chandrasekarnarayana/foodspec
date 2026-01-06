# Heating Stability (Teaching Walkthrough)

**Focus**: Time series • Degradation trend • Shelf-life estimate

## What you will learn
- Extract ratios that track chemical change over time
- Fit a trend to predict when quality crosses a threshold
- Visualize degradation and recommended replacement time

## Prerequisites
- Python 3.10+
- numpy, pandas, matplotlib, scipy

## Minimal runnable code
```python
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Simulated time series (hours) and degradation ratio
np.random.seed(42)
time_h = np.arange(0, 20, 2)
ratio = 0.05 * time_h + 0.05 * np.random.randn(len(time_h))

df = pd.DataFrame({"time_h": time_h, "ratio": ratio})
slope, intercept, r, p, _ = linregress(df["time_h"], df["ratio"])

threshold = 0.5
etime = (threshold - intercept) / slope
print({"slope": slope, "r2": r**2, "p": p, "time_to_threshold_h": etime})
```

## Explain the outputs
- `slope` > 0 ⇒ degradation is increasing
- `r2` near 1 ⇒ linear trend is a good fit
- `time_to_threshold_h` ⇒ estimated shelf-life (when ratio hits threshold)

## Full resources
- Full script: https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/heating_quality_quickstart.py
- Teaching notebook (download/run): https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/02_heating_stability_teaching.ipynb
- Example figure: https://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/heating_ratio_vs_time.png

## Run it yourself
```bash
python examples/heating_quality_quickstart.py
jupyter notebook examples/tutorials/02_heating_stability_teaching.ipynb
```

## Related docs
- Workflow: heating quality monitoring → ../workflows/quality-monitoring/heating_quality_monitoring.md
