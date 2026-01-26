# Trust & Uncertainty

FoodSpec treats calibration, conformal prediction, and interpretability as core outputs.

## Calibration
- Platt scaling (logistic calibration) and isotonic regression for post-hoc calibration.
- Metrics reported: Expected Calibration Error (ECE), Brier score, and NLL.
- Calibration is always fit on a held-out calibration split, never on test data.

## Conformal prediction
- Mondrian (group-conditional) conformal prediction sets with conditional coverage.
- Coverage and efficiency are reported overall and by group (when group sizes allow).
- Empirical coverage/efficiency curves are bootstrapped for uncertainty bounds.

## Abstention / reject option
- Abstain if confidence is low, conformal sets are too large, or the sample is in a low-density region.
- Reported outputs: reject rate, accuracy on accepted predictions, risk–coverage curve.

## Interpretability
- Coefficients and feature importances are reported for accepted predictions to avoid misleading attributions.
- Marker peak explanations are tied to accepted subsets for transparency.

## Regulatory readiness
- A readiness score (0–100) summarizes validation rigor, calibration quality, uncertainty guarantees,
  drift monitoring, documentation completeness, and reproducibility artifacts.
- Readiness is a heuristic checklist score, not a regulatory claim.

## Do / Don’t
**Do**
- Use a dedicated calibration split and keep the test set untouched.
- Report conditional coverage by relevant groups (stage, batch, instrument).
- Provide abstention rates and accuracy on answered samples.
- Save reproducibility artifacts (manifest, run summary, logs) with every run.

**Don’t**
- Claim “certified” or “approved” performance from calibration alone.
- Use calibration on the same data used for evaluation.
- Report point estimates without uncertainty intervals for coverage.
