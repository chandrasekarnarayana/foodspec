# QC System

Quality control is mandatory in FoodSpec. QC spans both spectral quality and dataset integrity.

## Spectral QC
- Noise and spike detection
- Baseline/fluorescence diagnostics
- Saturation/clipping checks

## Dataset QC
- Class balance
- Leakage detection
- Replicate consistency

## Multivariate QC (new)
- PCA/Hotelling T² outlier limits (chi-square, quantile, or MAD-based)
- Batch centroid drift with warn/fail thresholds
- Policy actions: flag | drop | down_weight

### YAML example
```yaml
qc:
	multivariate:
		enabled: true
		outliers:
			method: "hotelling_t2"
			alpha: 0.01
			policy: "flag"
			threshold_strategy: "chi2"
		drift:
			enabled: true
			metric: "centroid_l2"
			warn_threshold: 2.0
			fail_threshold: 4.0
```

### Artifacts
- `qc/multivariate_outliers.csv`
- `qc/multivariate_drift.csv`
- `qc/qc_summary.json` (ingested by report)

## Outputs
- `qc_results.json`
- Optional plots and reports

