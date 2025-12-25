# Model Lifecycle, Calibration, and Drift

This guide explains how FoodSpec supports production-grade model lifecycle management: calibration diagnostics, drift detection, aging/sunset rules, and champion–challenger promotion.

## Calibration Diagnostics

- API: `foodspec.ml.calibration.compute_calibration_diagnostics(y_true, y_proba, n_bins=10)`
- Metrics returned: slope/intercept, bias, ECE/MCE, Brier score, reliability curve.
- Recalibration: `recalibrate_classifier(clf, X_cal, y_cal, method="platt"|"isotonic")` wraps scikit-learn calibration.
- When to recalibrate: elevated ECE (>0.05), slope far from 1, or production drift indicates miscalibration.

## Drift Monitoring

- API: `foodspec.qc.drift.detect_feature_drift(ref, prod, threshold=0.1)` returns KL + PSI per feature.
- Production drift wrapper: `detect_production_drift(ref_df, prod_df, ref_perf, prod_perf)` → `ProductionDriftReport`.
- Recalibration trigger: `should_recalibrate(report, kl_thresh=0.2, perf_thresh=0.02)` flags when to update calibration or retrain.

## Model Aging & Sunset Rules

- Track performance over time with `ModelLifecycleTracker.record_performance(timestamp, metric_value, n_samples, **meta)`.
- Compute aging: `ModelLifecycleTracker.compute_aging_score(current_time=...)` → `ModelAgingScore` (age, decay, trend p-value, recommendation).
- Sunset policy: `SunsetRule(max_age_days=..., min_performance=..., max_decay_rate=..., grace_period_days=7)` applied inside `compute_aging_score`.
- Typical thresholds: max_age_days=180, min_performance=0.80, max_decay_rate=0.005/day after burn-in.

## Champion–Challenger Promotion

- Compare models: `compare_champion_challenger(champion_scores, challenger_scores, test_type="mcnemar"|"paired_ttest"|"wilcoxon")` → significance, CI, recommendation.
- Promote: `promote_challenger(comparison, champion_path, challenger_path, force=False)` backs up champion and swaps artifacts when approved.
- Interpretation: `✅ PROMOTE` when challenger is significantly better; `❌ REJECT` when challenger underperforms.

## Acting on QC Flags

- Runtime guard: `evaluate_prediction_qc(probs, drift_score=..., ece=...)` returns `qc_do_not_trust` and textual reasons (confidence, entropy, margin, drift, calibration).
- CLI: `foodspec-predict` now emits `qc_do_not_trust` and `qc_notes` columns when probabilities are available; flagged rows also trigger a stderr warning.
- Example handling:

```python
do_not_trust, warnings = guard_prediction(probs, min_confidence=0.7)
if do_not_trust:
	route_to_human_review(sample_id, warnings)
```

- Suggested policy: block automation when `qc_do_not_trust=True`, log `qc_notes`, and trigger recalibration if many rows get flagged.

## Recommended Workflow

1) Calibrate a candidate model on holdout data using Platt or isotonic.
2) Deploy with drift hooks: log predictions, inputs, and outcomes to feed drift + aging trackers.
3) Schedule drift checks (daily/weekly) with PSI/KL thresholds; trigger recalibration if drifted.
4) Track performance snapshots; if aging score flags decay or age > rule, schedule retrain/retire.
5) Run champion–challenger A/B: compare on fresh batches, promote when statistically better.

## Reproducibility

- Use `foodspec.report.methods` to auto-generate manuscript-ready methods text.
- Capture reproducibility status with `foodspec.report.checklist` and journal presets under `foodspec.report.journals`.
