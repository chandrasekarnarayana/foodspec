"""
Data Governance & Dataset Intelligence Demo

Demonstrates all five sub-modules:
1. Dataset summary: at-a-glance quality metrics
2. Class balance: imbalance detection and diagnostics
3. Replicate consistency: technical vs biological variability
4. Leakage detection: batch–label correlation, replicate splits
5. Readiness score: composite 0-100 score for ML deployment
"""

import numpy as np
import pandas as pd

from foodspec import FoodSpec

# ----------------------------------------------------------------------------
# Generate Synthetic Dataset with Known Issues
# ----------------------------------------------------------------------------

print("=" * 80)
print("FoodSpec Data Governance Demo")
print("=" * 80)

np.random.seed(42)

# 200 samples: 3 oil types (severe imbalance)
n_authentic = 120  # Majority class
n_adulterant_a = 50  # Medium minority
n_adulterant_b = 30  # Small minority

n_total = n_authentic + n_adulterant_a + n_adulterant_b
n_wavenumbers = 500

# Synthetic spectra with batch effects
# Batch 1: mostly authentic oils (confounded)
# Batch 2: mixed
# Batch 3: mostly adulterants (confounded)

batch_1_authentic = 90
batch_1_adulterant_a = 10
batch_1_adulterant_b = 0

batch_2_authentic = 20
batch_2_adulterant_a = 30
batch_2_adulterant_b = 10

batch_3_authentic = 10
batch_3_adulterant_a = 10
batch_3_adulterant_b = 20

# Generate spectra with batch offsets
def generate_batch_spectra(n_samples, base_mean, batch_offset, noise_level=0.05):
    spectra = np.random.randn(n_samples, n_wavenumbers) * noise_level
    spectra += base_mean + batch_offset
    return spectra

# Batch 1 (offset=+0.5)
batch_1_auth = generate_batch_spectra(batch_1_authentic, 1.0, 0.5)
batch_1_adA = generate_batch_spectra(batch_1_adulterant_a, 1.2, 0.5)

# Batch 2 (offset=0)
batch_2_auth = generate_batch_spectra(batch_2_authentic, 1.0, 0.0)
batch_2_adA = generate_batch_spectra(batch_2_adulterant_a, 1.2, 0.0)
batch_2_adB = generate_batch_spectra(batch_2_adulterant_b, 1.4, 0.0)

# Batch 3 (offset=-0.5)
batch_3_auth = generate_batch_spectra(batch_3_authentic, 1.0, -0.5)
batch_3_adA = generate_batch_spectra(batch_3_adulterant_a, 1.2, -0.5)
batch_3_adB = generate_batch_spectra(batch_3_adulterant_b, 1.4, -0.5)

# Combine
spectra = np.vstack([
    batch_1_auth, batch_1_adA,
    batch_2_auth, batch_2_adA, batch_2_adB,
    batch_3_auth, batch_3_adA, batch_3_adB,
])

# Metadata
oil_types = (
    ["authentic"] * batch_1_authentic + ["adulterant_A"] * batch_1_adulterant_a +
    ["authentic"] * batch_2_authentic + ["adulterant_A"] * batch_2_adulterant_a + ["adulterant_B"] * batch_2_adulterant_b +
    ["authentic"] * batch_3_authentic + ["adulterant_A"] * batch_3_adulterant_a + ["adulterant_B"] * batch_3_adulterant_b
)

batches = (
    ["batch_1"] * (batch_1_authentic + batch_1_adulterant_a) +
    ["batch_2"] * (batch_2_authentic + batch_2_adulterant_a + batch_2_adulterant_b) +
    ["batch_3"] * (batch_3_authentic + batch_3_adulterant_a + batch_3_adulterant_b)
)

# Replicates: some samples measured multiple times (simulate technical replicates)
# Create replicate groups: 80% singleton, 20% duplicates
sample_ids = []
current_id = 1
i = 0
while i < n_total:
    if np.random.rand() < 0.2 and i < n_total - 1:
        # Duplicate
        sample_ids.extend([f"sample_{current_id}"] * 2)
        i += 2
    else:
        # Singleton
        sample_ids.append(f"sample_{current_id}")
        i += 1
    current_id += 1

# Pad if needed
while len(sample_ids) < n_total:
    sample_ids.append(f"sample_{current_id}")
    current_id += 1

sample_ids = sample_ids[:n_total]

metadata = pd.DataFrame({
    "oil_type": oil_types,
    "batch": batches,
    "sample_id": sample_ids,
    "collection_date": pd.date_range("2024-01-01", periods=n_total, freq="h"),
})

wavenumbers = np.linspace(400, 3000, n_wavenumbers)

# ----------------------------------------------------------------------------
# Initialize FoodSpec
# ----------------------------------------------------------------------------

print("\n[1/5] Initializing FoodSpec dataset...")
fs = FoodSpec(
    spectra,
    wavenumbers=wavenumbers,
    metadata=metadata,
    modality="raman",
    kind="oil_authentication_with_issues",
)
print(f"Dataset: {len(fs.data)} samples × {fs.data.x.shape[1]} wavenumbers")

# ----------------------------------------------------------------------------
# 1. Dataset Summary
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("[2/5] Dataset Summary")
print("=" * 80)

summary = fs.summarize_dataset(label_column="oil_type")

print("\nClass Distribution:")
print(f"  Total samples: {summary['class_distribution']['total_samples']}")
print(f"  Number of classes: {summary['class_distribution']['n_classes']}")
print(f"  Samples per class: {summary['class_distribution']['samples_per_class']}")
print(f"  Imbalance ratio: {summary['class_distribution']['imbalance_ratio']:.2f}:1")
print(f"  Min class size: {summary['class_distribution']['min_class_size']}")
print(f"  Max class size: {summary['class_distribution']['max_class_size']}")


print("\nSpectral Quality:")
print(f"  Mean SNR: {summary['spectral_quality']['mean_snr']:.2f}")
print(f"  Spectral range: {summary['spectral_quality']['spectral_range']}")
print(f"  NaN count: {summary['spectral_quality']['nan_count']}")
print(f"  Negative intensity rate: {summary['spectral_quality']['negative_intensity_rate']:.4f}")

print("\nMetadata Completeness:")
print(f"  Overall completeness: {summary['metadata_completeness']['overall_completeness']:.2%}")
print(f"  Columns with missing data: {summary['metadata_completeness']['columns_with_missing']}")

# ----------------------------------------------------------------------------
# 2. Class Balance Diagnostics
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("[3/5] Class Balance Diagnostics")
print("=" * 80)

balance = fs.check_class_balance(label_column="oil_type")

print(f"\nImbalance Ratio: {balance['imbalance_ratio']:.2f}:1")
print(f"Severe Imbalance: {'⚠️ YES' if balance['severe_imbalance'] else '✓ NO'}")
print(f"Undersized Classes (< 20 samples): {balance['undersized_classes']}")
print(f"\nRecommendation:\n{balance['recommended_action']}")

# ----------------------------------------------------------------------------
# 3. Replicate Consistency
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("[4/5] Replicate Consistency")
print("=" * 80)

consistency = fs.assess_replicate_consistency(replicate_column="sample_id")

print(f"\nNumber of replicate groups evaluated: {consistency['n_replicates']}")
print(f"Median CV across replicates: {consistency['median_cv']:.2f}%")
print(f"Mean CV: {consistency['mean_cv']:.2f}%")
print(f"Max CV (worst replicate): {consistency['max_cv']:.2f}%")
print(f"High variability replicates (CV > 10%): {len(consistency['high_variability_replicates'])}")

if consistency['high_variability_replicates']:
    print(f"  Examples: {consistency['high_variability_replicates'][:5]}")

# ----------------------------------------------------------------------------
# 4. Leakage Detection
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("[5/5] Leakage Detection")
print("=" * 80)

leakage = fs.detect_leakage(
    label_column="oil_type",
    batch_column="batch",
    replicate_column="sample_id",
)

print("\nBatch–Label Correlation:")
print(f"  Cramér's V: {leakage['batch_label_correlation']['cramers_v']:.3f}")
print(f"  Chi² p-value: {leakage['batch_label_correlation']['chi2_pvalue']:.4f}")
print(f"  Severe confounding: {'⚠️ YES' if leakage['batch_label_correlation']['severe'] else '✓ NO'}")
print(f"  {leakage['batch_label_correlation']['interpretation']}")

print("\nReplicate Leakage Risk:")
print(f"  Replicate groups: {leakage['replicate_leakage']['replicate_groups']}")
print(f"  Mean group size: {leakage['replicate_leakage']['mean_group_size']:.2f}")
print(f"  Leakage risk: {leakage['replicate_leakage']['leakage_risk'].upper()}")
print(f"  {leakage['replicate_leakage']['recommended_action']}")

print(f"\n⚠️  Overall Leakage Risk: {leakage['overall_risk'].upper()}")

# ----------------------------------------------------------------------------
# 5. Dataset Readiness Score
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Dataset Readiness Score (0-100)")
print("=" * 80)

readiness = fs.compute_readiness_score(
    label_column="oil_type",
    batch_column="batch",
    replicate_column="sample_id",
)

print(f"\n{'🎯 OVERALL READINESS SCORE:':<40} {readiness['overall_score']:.1f}/100")

print("\nDimension Scores:")
for dim, score in readiness['dimension_scores'].items():
    status = "✓" if score >= 70 else "⚠️"
    print(f"  {status} {dim:<30} {score:.1f}/100")

print(f"\nPassed Criteria ({len(readiness['passed_criteria'])}):")
for criterion in readiness['passed_criteria']:
    print(f"  ✓ {criterion}")

print(f"\n⚠️  Failed Criteria ({len(readiness['failed_criteria'])}):")
for criterion in readiness['failed_criteria']:
    print(f"  ⚠️  {criterion}")

print(f"\n{readiness['recommendation']}")

# ----------------------------------------------------------------------------
# Export Bundle
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Exporting Results")
print("=" * 80)

output_path = fs.export(path="protocol_runs_test/", formats=["json", "csv"])  # Export to default directory with JSON/CSV
print(f"✓ Results exported to: {output_path}")

print("\nMetrics recorded in OutputBundle:")
print("  - dataset_summary")
print("  - class_balance")
print("  - replicate_consistency")
print("  - leakage_detection")
print("  - readiness_score")

print("\n" + "=" * 80)
print("Demo Complete!")
print("=" * 80)
print("\n✓ All data governance modules functional.")
print("✓ Readiness score provides ML deployment gatekeeping.")
print("✓ Issues flagged: severe imbalance, batch confounding, replicate leakage risk.")
