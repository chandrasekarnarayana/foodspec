---
title: Temporal & Aging Workflows
---

This guide covers FoodSpec's time-aware analysis for heating, oxidation, and shelf-life studies.

Highlights:
- Time-aware dataset: `TimeSpectrumSet` with `time_col` and `entity_col`.
- Degradation trajectories: linear or spline fits per entity.
- Stage classification: early / mid / late along the timeline.
- Shelf-life estimation: remaining time to threshold with 95% CI.

Quickstart (CLI)
- Aging trajectories and stages:
  - `foodspec aging input.h5 --value-col degrade --method linear --time-col time --entity-col sample_id --output-dir ./out`
- Shelf-life estimates:
  - `foodspec shelf-life input.h5 --value-col degrade --threshold 2.0 --time-col time --entity-col sample_id --output-dir ./out`

Python API
- Build a `TimeSpectrumSet` from a `FoodSpectrumSet`:
  - `ts = TimeSpectrumSet(x=fs.x, wavenumbers=fs.wavenumbers, metadata=fs.metadata, modality=fs.modality, time_col='time', entity_col='sample_id')`
- Trajectories:
  - `from foodspec.workflows.aging import compute_degradation_trajectories`
  - `res = compute_degradation_trajectories(ts, value_col='degrade', method='linear')`
- Shelf-life:
  - `from foodspec.workflows.shelf_life import estimate_remaining_shelf_life`
  - `df = estimate_remaining_shelf_life(ts, value_col='degrade', threshold=2.0)`
