"""
Example: Trust & Uncertainty Quantification for Oil Authentication.

Demonstrates:
1. Conformal prediction with Mondrian conditioning (per-batch coverage)
2. Probability calibration (temperature scaling)
3. Abstention with decision rules
4. Group-aware coverage diagnostics
5. Artifact saving and reporting
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from foodspec.trust.conformal import MondrianConformalClassifier
from foodspec.trust.calibration import TemperatureScaler, expected_calibration_error
from foodspec.trust.abstain import evaluate_abstention
from foodspec.trust.evaluator import TrustEvaluator
from foodspec.core.artifacts import ArtifactRegistry


def generate_synthetic_oil_data(n_samples=1000, n_features=20, random_state=42):
    """Generate synthetic oil authentication dataset."""
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # Linear classification with some noise
    weights = np.random.randn(n_features)
    logits = X @ weights
    y = (logits > 0).astype(int)
    
    # Add batch/instrument labels
    batches = np.random.choice(["batch_A", "batch_B", "batch_C"], n_samples)
    batch_ids = pd.factorize(batches)[0]
    
    return X, y, batch_ids, batches


def main():
    """Run complete trust evaluation workflow."""
    
    print("=" * 70)
    print("TRUST & UNCERTAINTY QUANTIFICATION EXAMPLE")
    print("Oil Authentication Task")
    print("=" * 70)
    
    # 1. Generate synthetic data
    print("\n[1] Generating synthetic oil authentication data...")
    X, y, batch_ids, batches = generate_synthetic_oil_data(n_samples=800)
    
    # Split: 50% train, 25% calibration, 25% test
    X_train, X_temp, y_train, y_temp, batch_train, batch_temp = train_test_split(
        X, y, batch_ids, test_size=0.5, random_state=42
    )
    X_cal, X_test, y_cal, y_test, batch_cal, batch_test = train_test_split(
        X_temp, y_temp, batch_temp, test_size=0.5, random_state=42
    )
    print(f"  Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")
    
    # 2. Train base model
    print("\n[2] Training base classifier...")
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    print(f"  Training accuracy: {train_acc:.1%}")
    
    # 3. Fit probability calibration
    print("\n[3] Fitting probability calibration (temperature scaling)...")
    proba_cal = model.predict_proba(X_cal)
    calibrator = TemperatureScaler()
    calibrator.fit(y_cal, proba_cal)
    print(f"  Temperature scale: {calibrator.temperature:.4f}")
    
    # Apply calibration to all predictions
    proba_test_raw = model.predict_proba(X_test)
    proba_test_cal = calibrator.predict(proba_test_raw)
    
    ece_raw = expected_calibration_error(y_test, proba_test_raw)
    ece_cal = expected_calibration_error(y_test, proba_test_cal)
    print(f"  ECE before: {ece_raw:.4f}")
    print(f"  ECE after:  {ece_cal:.4f}")
    
    # 4. Fit conformal prediction
    print("\n[4] Fitting conformal prediction (target coverage: 90%)...")
    cp = MondrianConformalClassifier(model, target_coverage=0.90)
    cp._fitted = True
    cp._n_classes = 2
    cp.calibrate(X_cal, y_cal, bins=batch_cal)
    
    # Predict uncertainty sets
    cp_result = cp.predict_sets(X_test, bins=batch_test, y_true=y_test)
    print(f"  Achieved coverage: {cp_result.coverage:.1%}")
    print(f"  Mean set size: {np.mean(cp_result.set_sizes):.2f}")
    print(f"  Median set size: {np.median(cp_result.set_sizes):.1f}")
    
    # 5. Group-aware coverage
    print("\n[5] Group-aware coverage analysis...")
    print("  Per-batch coverage:")
    for batch_id, coverage in cp_result.per_bin_coverage.items():
        print(f"    Batch {batch_id}: {coverage:.1%}")
    
    # 6. Abstention evaluation
    print("\n[6] Evaluating abstention rules...")
    abstain_result = evaluate_abstention(
        proba_test_cal,
        y_test,
        threshold=0.75,
        prediction_sets=cp_result.prediction_sets,
        max_set_size=1,
    )
    print(f"  Abstention rate: {abstain_result.abstention_rate:.1%}")
    print(f"  Coverage (non-abstained): {abstain_result.coverage:.1%}")
    if abstain_result.accuracy_non_abstained is not None:
        print(f"  Accuracy (non-abstained): {abstain_result.accuracy_non_abstained:.1%}")
    
    # 7. High-level evaluator (integrated workflow)
    print("\n[7] Running integrated TrustEvaluator...")
    output_dir = Path("/tmp/trust_example_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    registry = ArtifactRegistry(output_dir)
    registry.ensure_layout()
    
    evaluator = TrustEvaluator(
        model,
        artifact_registry=registry,
        target_coverage=0.90,
        abstention_threshold=0.75,
        calibration_method="temperature",
        random_state=42,
    )
    
    # Fit evaluator
    evaluator.fit_conformal(X_cal, y_cal, bins_cal=batch_cal)
    evaluator.fit_calibration(y_cal, proba_cal)
    
    # Comprehensive evaluation
    eval_result = evaluator.evaluate(
        X_test, y_test,
        bins_test=batch_test,
        batch_ids=batch_test,
        model_name="oil_auth_v1",
    )
    
    # 8. Save artifacts
    print("\n[8] Saving artifacts...")
    artifacts = evaluator.save_artifacts(
        eval_result,
        prediction_sets=cp_result.prediction_sets,
        set_sizes=cp_result.set_sizes,
        abstention_mask=abstain_result.abstain_mask,
        output_dir=output_dir / "trust",
    )
    print(f"  Saved artifacts: {list(artifacts.keys())}")
    
    # 9. Generate report
    print("\n[9] Evaluation Report:")
    print(evaluator.report(eval_result))
    
    # 10. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Coverage guarantee (Mondrian): {eval_result.conformal_coverage:.1%}")
    print(f"✓ Calibration error (ECE): {eval_result.ece:.4f}")
    print(f"✓ Abstention strategy: {eval_result.abstention_rate:.1%} reject rate")
    print(f"✓ Efficiency gain: {eval_result.efficiency_gain:.2f}x (coverage/baseline)")
    print(f"✓ Artifacts saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
