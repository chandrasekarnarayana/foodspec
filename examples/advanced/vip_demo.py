"""Example: Using Variable Importance in Projection (VIP) for spectral interpretation.

This script demonstrates how to use VIP scores to identify important
wavenumbers/wavelengths in PLS models for food spectroscopy applications.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from foodspec.chemometrics import calculate_vip, calculate_vip_da, interpret_vip


def example_regression_vip():
    """Example: VIP for PLS regression (e.g., predicting oil quality)."""
    print("="*60)
    print("Example 1: VIP for PLS Regression")
    print("="*60)

    # Simulate Raman spectra for olive oil quality prediction
    np.random.seed(42)
    n_samples = 80
    n_wavenumbers = 200

    # Create synthetic spectra with important peaks
    X = np.random.randn(n_samples, n_wavenumbers) * 0.1

    # Add important peaks (e.g., C=C stretch at wavenumber 50, C-H at 100)
    X[:, 50] += np.random.randn(n_samples) * 0.5 + 2.0  # Important peak 1
    X[:, 100] += np.random.randn(n_samples) * 0.5 + 3.0  # Important peak 2
    X[:, 150] += np.random.randn(n_samples) * 0.5 + 1.5  # Important peak 3

    # Quality score depends on these peaks
    y = 0.5 * X[:, 50] + 0.8 * X[:, 100] + 0.3 * X[:, 150] + np.random.randn(n_samples) * 0.2

    # Fit PLS model
    pls = PLSRegression(n_components=5)
    pls.fit(X, y)

    # Calculate VIP scores
    vip_scores = calculate_vip(pls, X, y)

    # Interpret VIP
    wavenumber_names = [f"WN_{i}" for i in range(n_wavenumbers)]
    interpretation = interpret_vip(vip_scores, wavenumber_names)

    print(f"\nNumber of highly important features (VIP > 1.0): {len(interpretation['highly_important'])}")
    print("Top 10 important wavenumbers:")
    for name, score in interpretation['top_10']:
        print(f"  {name}: VIP = {score:.3f}")

    # Check if our known important wavenumbers have high VIP
    important_wavenumbers = [50, 100, 150]
    print("\nVIP scores for our known important wavenumbers:")
    for idx in important_wavenumbers:
        print(f"  Wavenumber {idx}: VIP = {vip_scores[idx]:.3f}")


def example_classification_vip():
    """Example: VIP for PLS-DA (e.g., oil authentication)."""
    print("\n" + "="*60)
    print("Example 2: VIP for PLS-DA Classification")
    print("="*60)

    # Simulate spectra for authentic vs adulterated oil
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    X = np.random.randn(n_samples, n_features) * 0.5

    # Create binary classes (0=authentic, 1=adulterated)
    y = np.zeros(n_samples, dtype=int)
    y[50:] = 1

    # Adulterated oils have different profiles at specific features
    X[y == 1, 10] += 2.0  # Discriminating feature 1
    X[y == 1, 20] += 1.5  # Discriminating feature 2
    X[y == 1, 30] -= 1.0  # Discriminating feature 3

    # Fit PLS-DA model
    from sklearn.preprocessing import LabelBinarizer
    pls_step = PLSRegression(n_components=3)
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    if y_encoded.shape[1] == 1:
        y_encoded = np.hstack([1 - y_encoded, y_encoded])
    pls_step.fit(X, y_encoded)

    pls_da = Pipeline([
        ('pls', pls_step),
        ('clf', LogisticRegression())
    ])
    X_pls = pls_step.transform(X)
    pls_da.named_steps['clf'].fit(X_pls, y)

    # Calculate VIP scores
    vip_scores = calculate_vip_da(pls_da, X, y)

    # Interpret VIP
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    interpretation = interpret_vip(vip_scores, feature_names)

    print(f"\nNumber of highly important features (VIP > 1.0): {len(interpretation['highly_important'])}")
    print("Top 10 discriminating features:")
    for name, score in interpretation['top_10']:
        print(f"  {name}: VIP = {score:.3f}")

    # Check discriminating features
    discriminating_features = [10, 20, 30]
    print("\nVIP scores for known discriminating features:")
    for idx in discriminating_features:
        print(f"  Feature {idx}: VIP = {vip_scores[idx]:.3f}")

    # Classification accuracy
    from sklearn.metrics import accuracy_score
    y_pred = pls_da.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\nClassification accuracy: {accuracy:.1%}")


def example_pipeline_vip():
    """Example: VIP with sklearn Pipeline (preprocessing + PLS)."""
    print("\n" + "="*60)
    print("Example 3: VIP with Preprocessing Pipeline")
    print("="*60)

    from sklearn.preprocessing import StandardScaler

    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = 2 * X[:, 0] + 3 * X[:, 5] - X[:, 10] + np.random.randn(100) * 0.5

    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pls', PLSRegression(n_components=5))
    ])
    pipeline.fit(X, y)

    # Calculate VIP (works with pipeline)
    vip_scores = calculate_vip(pipeline, X, y)

    # Interpret
    feature_names = [f"Var_{i}" for i in range(20)]
    interpretation = interpret_vip(vip_scores, feature_names)

    print("\nTop 5 important variables:")
    for name, score in interpretation['top_10'][:5]:
        print(f"  {name}: VIP = {score:.3f}")

    # Our ground truth important features are 0, 5, 10
    print("\nVIP scores for true important features:")
    for idx in [0, 5, 10]:
        print(f"  Variable {idx}: VIP = {vip_scores[idx]:.3f}")


def example_interpretation_categories():
    """Example: Categorizing features by VIP scores."""
    print("\n" + "="*60)
    print("Example 4: VIP Score Interpretation")
    print("="*60)

    # Simulate VIP scores
    vip_scores = np.array([1.8, 1.3, 0.9, 0.7, 1.5, 0.3, 1.1, 0.5, 2.2, 0.8])
    feature_names = [f"Feature_{i}" for i in range(10)]

    # Interpret
    result = interpret_vip(vip_scores, feature_names)

    print("\nHighly Important Features (VIP > 1.0):")
    for name, score in result['highly_important']:
        print(f"  {name}: VIP = {score:.3f}")

    print("\nModerately Important Features (0.8 < VIP ≤ 1.0):")
    for name, score in result['moderately_important']:
        print(f"  {name}: VIP = {score:.3f}")

    print("\nLow Importance Features (VIP ≤ 0.8):")
    for name, score in result['low_importance']:
        print(f"  {name}: VIP = {score:.3f}")

    print("\nInterpretation Guidelines:")
    print("  - VIP > 1.0: Highly influential, crucial for model")
    print("  - 0.8 < VIP ≤ 1.0: Moderately important")
    print("  - VIP ≤ 0.8: Low importance, may be noise or redundant")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Variable Importance in Projection (VIP) Examples")
    print("="*60)

    example_regression_vip()
    example_classification_vip()
    example_pipeline_vip()
    example_interpretation_categories()

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. VIP identifies important features/wavenumbers in PLS models")
    print("  2. VIP > 1.0 indicates high importance")
    print("  3. Works with both regression (PLS) and classification (PLS-DA)")
    print("  4. Compatible with sklearn Pipelines")
    print("  5. Use interpret_vip() for easy interpretation")
    print()
