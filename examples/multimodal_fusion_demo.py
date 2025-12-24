#!/usr/bin/env python
"""
Multi-modal fusion example: Combining Raman + FTIR for olive oil authentication.

This script demonstrates:
1. Creating synthetic multi-modal data (Raman + FTIR)
2. Late fusion (feature concatenation)
3. Decision fusion (voting)
4. Agreement metrics
"""

import numpy as np
from foodspec.core import FoodSpectrumSet, MultiModalDataset
from foodspec.ml.fusion import late_fusion_concat, decision_fusion_vote, decision_fusion_weighted
from foodspec.stats.fusion_metrics import (
    modality_agreement_kappa,
    modality_consistency_rate,
    cross_modality_correlation,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def create_synthetic_multimodal_data(n_samples=100, seed=42):
    """Create synthetic Raman + FTIR data for olive oil authentication."""
    np.random.seed(seed)
    
    # Create two classes: authentic (0) vs adulterated (1)
    labels = np.array([0] * 50 + [1] * 50)
    
    # Raman spectra (400-1800 cm⁻¹, 200 points)
    raman_wave = np.linspace(400, 1800, 200)
    raman_spectra = np.zeros((n_samples, 200))
    for i in range(n_samples):
        if labels[i] == 0:  # authentic
            # Strong peak at 1650 (C=C stretch)
            raman_spectra[i] = 10 * np.exp(-((raman_wave - 1650) / 100) ** 2)
        else:  # adulterated
            # Shifted peak at 1600 + weak shoulder at 1700
            raman_spectra[i] = 8 * np.exp(-((raman_wave - 1600) / 100) ** 2)
            raman_spectra[i] += 3 * np.exp(-((raman_wave - 1700) / 80) ** 2)
        # Add noise
        raman_spectra[i] += np.random.normal(0, 0.5, 200)
    
    # FTIR spectra (400-4000 cm⁻¹, 300 points)
    ftir_wave = np.linspace(400, 4000, 300)
    ftir_spectra = np.zeros((n_samples, 300))
    for i in range(n_samples):
        if labels[i] == 0:  # authentic
            # Carbonyl peak at 1745 cm⁻¹ (ester)
            ftir_spectra[i] = 15 * np.exp(-((ftir_wave - 1745) / 80) ** 2)
        else:  # adulterated
            # Broader carbonyl at 1730 + acid shoulder at 1710
            ftir_spectra[i] = 12 * np.exp(-((ftir_wave - 1730) / 90) ** 2)
            ftir_spectra[i] += 5 * np.exp(-((ftir_wave - 1710) / 70) ** 2)
        # Add noise
        ftir_spectra[i] += np.random.normal(0, 0.3, 300)
    
    # Create FoodSpectrumSet objects
    import pandas as pd
    
    raman_ds = FoodSpectrumSet(
        x=raman_spectra,
        wavenumbers=raman_wave,
        modality="raman",
        metadata=pd.DataFrame({
            "sample_id": [f"olive_{i:03d}" for i in range(n_samples)],
            "authentic": labels,
            "batch": ["A" if i < 50 else "B" for i in range(n_samples)],
        }),
        label_col="authentic",
    )
    
    ftir_ds = FoodSpectrumSet(
        x=ftir_spectra,
        wavenumbers=ftir_wave,
        modality="ftir",
        metadata=pd.DataFrame({
            "sample_id": [f"olive_{i:03d}" for i in range(n_samples)],
            "authentic": labels,
            "batch": ["A" if i < 50 else "B" for i in range(n_samples)],
        }),
        label_col="authentic",
    )
    
    return raman_ds, ftir_ds, labels


def demo_late_fusion(mmd, labels):
    """Demonstrate late fusion (feature concatenation)."""
    print("\n" + "=" * 60)
    print("LATE FUSION (Feature-Level Concatenation)")
    print("=" * 60)
    
    # Extract features
    features = mmd.to_feature_dict()
    print(f"Raman features: {features['raman'].shape}")
    print(f"FTIR features:  {features['ftir'].shape}")
    
    # Concatenate
    result = late_fusion_concat(features)
    print(f"Fused features: {result.X_fused.shape}")
    print(f"Boundaries:     {result.modality_boundaries}")
    
    # Train joint model
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(clf, result.X_fused, labels, cv=5)
    
    print(f"\n5-fold CV Accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
    return result, clf


def demo_decision_fusion(mmd, labels):
    """Demonstrate decision fusion (voting)."""
    print("\n" + "=" * 60)
    print("DECISION FUSION (Prediction-Level Voting)")
    print("=" * 60)
    
    predictions = {}
    probas = {}
    
    for modality, ds in mmd.datasets.items():
        X = ds.x
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Train
        clf.fit(X, labels)
        
        # Predict
        predictions[modality] = clf.predict(X)
        probas[modality] = clf.predict_proba(X)
        
        accuracy = (predictions[modality] == labels).mean()
        print(f"{modality.upper()} model accuracy: {accuracy:.2%}")
    
    # Majority voting
    vote_result = decision_fusion_vote(predictions, strategy="majority")
    print(f"\nMajority voting accuracy: {(vote_result.predictions == labels).mean():.2%}")
    
    # Unanimous voting
    unanimous_result = decision_fusion_vote(predictions, strategy="unanimous")
    agreed_mask = unanimous_result.predictions != -1
    unanimous_fraction = agreed_mask.mean()
    unanimous_accuracy = (unanimous_result.predictions[agreed_mask] == labels[agreed_mask]).mean() if unanimous_fraction > 0 else 0.0
    print(f"Unanimous agreement rate: {unanimous_fraction:.1%}")
    print(f"Unanimous accuracy (on agreed samples): {unanimous_accuracy:.2%}")
    
    # Weighted fusion (Raman more reliable for this task)
    weights = {"raman": 0.6, "ftir": 0.4}
    weighted_result = decision_fusion_weighted(probas, weights=weights)
    print(f"\nWeighted fusion (Raman=0.6, FTIR=0.4) accuracy: {(weighted_result.predictions == labels).mean():.2%}")
    
    return predictions, probas


def demo_agreement_metrics(predictions, features):
    """Demonstrate agreement metrics."""
    print("\n" + "=" * 60)
    print("AGREEMENT METRICS")
    print("=" * 60)
    
    # Cohen's kappa
    kappa_df = modality_agreement_kappa(predictions)
    print("\nCohen's Kappa (inter-rater agreement):")
    print(kappa_df)
    print("\n(κ > 0.8 = excellent, 0.6-0.8 = good, 0.4-0.6 = moderate)")
    
    # Consistency rate
    consistency = modality_consistency_rate(predictions)
    print(f"\nUnanimous consistency rate: {consistency:.1%}")
    
    # Cross-modality correlation
    corr_df = cross_modality_correlation(features, method="pearson")
    print("\nCross-modality feature correlation (Pearson):")
    print(corr_df)


def main():
    print("=" * 60)
    print("Multi-Modal Fusion Demo: Olive Oil Authentication")
    print("=" * 60)
    
    # 1. Create synthetic data
    print("\n[1/4] Creating synthetic Raman + FTIR data...")
    raman_ds, ftir_ds, labels = create_synthetic_multimodal_data(n_samples=100)
    
    # 2. Create multi-modal dataset
    print("[2/4] Creating MultiModalDataset...")
    mmd = MultiModalDataset.from_datasets({
        "raman": raman_ds,
        "ftir": ftir_ds,
    })
    print(f"Multi-modal dataset: {len(mmd.modalities())} modalities, {len(mmd.metadata)} samples")
    
    # 3. Late fusion
    print("\n[3/4] Running late fusion...")
    features = mmd.to_feature_dict()
    late_result, late_clf = demo_late_fusion(mmd, labels)
    
    # 4. Decision fusion
    print("\n[4/4] Running decision fusion...")
    predictions, probas = demo_decision_fusion(mmd, labels)
    
    # 5. Agreement metrics
    demo_agreement_metrics(predictions, features)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Late fusion: Joint model on concatenated features")
    print("✅ Decision fusion: Separate models + voting/weighted averaging")
    print("✅ Agreement metrics: Cohen's kappa, consistency rate, correlation")
    print("\nSee docs/multimodal_workflows.md for full guide!")


if __name__ == "__main__":
    main()
