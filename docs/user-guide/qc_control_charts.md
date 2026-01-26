# QC Control Charts

FoodSpec provides control chart utilities for monitoring batch stability and QC metrics. These charts help detect small shifts, sustained drifts, or instability in production runs.

## Supported charts

- X-bar and R charts
- X-bar and S charts
- Individuals and Moving Range (I-MR)
- CUSUM and EWMA
- Attribute charts: P, NP, C, U
- Levey-Jennings

## CLI usage

Generate an Individuals/MR chart from a CSV column:

```bash
foodspec qc control-chart data/qc_metrics.csv \
  --value-col intensity_mean \
  --chart imr \
  --run-dir runs/qc_imr
```

Generate an X-bar/R chart with subgroup size:

```bash
foodspec qc control-chart data/qc_metrics.csv \
  --value-col intensity_mean \
  --chart xbar_r \
  --subgroup-size 5 \
  --run-dir runs/qc_xbar_r
```

Generate an attribute chart (P chart):

```bash
foodspec qc control-chart data/qc_metrics.csv \
  --value-col intensity_mean \
  --chart p \
  --defect-col defect_count \
  --sample-size-col sample_size \
  --run-dir runs/qc_p_chart
```

Outputs:
- `runs/qc_*/qc/control_charts.json`
- `runs/qc_*/manifest.json`
- `runs/qc_*/run_summary.json`

## Programmatic API

```python
from foodspec.qc.control_charts import xbar_r_chart

result = xbar_r_chart(values, subgroup_size=5)
print(result.xbar.center, result.xbar.ucl, result.xbar.lcl)
```

## Interpretation notes

- Use `CUSUM` and `EWMA` for detecting small, sustained shifts.
- Use `I-MR` when data is collected one sample at a time.
- For attribute charts, ensure consistent sampling plans and clear defect definitions.
