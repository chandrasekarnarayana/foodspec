# Example: Aging Workflow (Storage Stability)

**Goal:** Model degradation trajectories over storage time (days/months) and estimate remaining shelf-life.

## 🔬 Minimal Reproducible Example (Synthetic)

```python
from examples.aging_quickstart import main
# Runs synthetic dataset generation, trajectory fitting, and shelf-life estimation
main()
```

**Expected Output:**
```yaml
Dataset: ~300 spectra, time range 0-180 days
Entities: 5

Trajectory metrics (first 5 rows):
  entity  slope  acceleration
  ...     ...    ...

Saved: outputs/aging_degradation_trajectories.png
Shelf-life estimates (days):
  entity  t_star  ci_low  ci_high  slope  intercept
  ...     ...     ...     ...      ...    ...

Saved: outputs/aging_shelf_life_estimates.png
```

## 📈 Figures

- Degradation Trajectories: [docs/assets/workflows/aging/degradation_trajectories.png](../assets/workflows/aging/degradation_trajectories.png)
- Shelf-Life Estimates: [docs/assets/workflows/aging/shelf_life_estimates.png](../assets/workflows/aging/shelf_life_estimates.png)
- Residual Diagnostics: [docs/assets/workflows/aging/residual_plot.png](../assets/workflows/aging/residual_plot.png)

## 🔗 See Also

- Methods: [Baseline Correction](../methods/preprocessing/baseline_correction.md), [Normalization & Smoothing](../methods/preprocessing/normalization_smoothing.md), [Feature Extraction](../methods/preprocessing/feature_extraction.md)
- API: [Aging Trajectories](../api/workflows.md), [Shelf-Life Estimation](../api/workflows.md#estimate_remaining_shelf_life), [Time Metrics](../api/stats.md)
