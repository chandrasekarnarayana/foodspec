# Method Comparison

Bland–Altman analysis helps compare two measurement methods (e.g., Raman vs FTIR, instrument A vs B).

## Python API

```python
import numpy as np
from foodspec.stats.method_comparison import bland_altman, bland_altman_plot

A = np.array([1,2,3,4,5], dtype=float)
B = np.array([1.1,2.1,2.9,4.2,4.8], dtype=float)
res = bland_altman(A, B)
print(res)
fig = bland_altman_plot(A, B, title="Raman vs FTIR")
fig.savefig("bland_altman.png")
```

- Outputs: mean difference and limits of agreement.
- Plot: scatter of average vs difference with mean and LoA lines.
