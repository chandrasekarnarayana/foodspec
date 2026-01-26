"""
Comprehensive example: Professional Chemometrics Core pipeline.

Demonstrates all Phase 1 features:
- Spectral alignment (cross-correlation, DTW)
- NNLS spectral unmixing
- PLSR with VIP scores
- Bootstrap stability analysis
- Bland-Altman agreement analysis
- Drift monitoring with EWMA
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from foodspec.features.alignment import align_spectra
from foodspec.features.unmixing import NNLSUnmixer
from foodspec.modeling.chemometrics import PLSRegression, VIPCalculator
from foodspec.modeling.chemometrics import NNLSRegression
from foodspec.validation.stability import BootstrapStability, StabilityIndex
from foodspec.validation.agreement import BlandAltmanAnalysis, DemingRegression
from foodspec.qc.drift_ewma import DriftDetector


def demo_spectral_alignment():
    """Demonstrate spectral alignment."""
    print("\n" + "=" * 60)
    print("1. SPECTRAL ALIGNMENT DEMO")
    print("=" * 60)
    
    # Synthetic spectra with drift
    n_samples, n_wavenumbers = 20, 100
    X_baseline = np.exp(-((np.arange(n_wavenumbers) - 50) ** 2) / 200) + np.random.randn(n_wavenumbers) * 0.05
    
    X = np.tile(X_baseline, (n_samples, 1))
    for i in range(n_samples):
        X[i] = np.roll(X[i], np.random.randint(-5, 6))  # Add random shifts
    
    # Align using both methods
    X_aligned_xcorr = align_spectra(X, method="xcorr", max_shift=10)
    X_aligned_dtw = align_spectra(X, method="dtw", window=10)
    
    print(f"Input spectra shape: {X.shape}")
    print(f"Aligned (xcorr) shape: {X_aligned_xcorr.shape}")
    print(f"Aligned (DTW) shape: {X_aligned_dtw.shape}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(X.T, alpha=0.3, color='blue')
    axes[0].set_title("Original Spectra (Misaligned)")
    axes[0].set_ylabel("Intensity")
    
    axes[1].plot(X_aligned_xcorr.T, alpha=0.3, color='green')
    axes[1].set_title("Cross-Correlation Aligned")
    
    axes[2].plot(X_aligned_dtw.T, alpha=0.3, color='red')
    axes[2].set_title("DTW Aligned")
    
    for ax in axes:
        ax.set_xlabel("Wavenumber")
    
    plt.tight_layout()
    plt.savefig("/tmp/alignment_demo.png", dpi=100)
    print("✓ Saved alignment_demo.png")


def demo_nnls_unmixing():
    """Demonstrate NNLS spectral unmixing."""
    print("\n" + "=" * 60)
    print("2. NNLS SPECTRAL UNMIXING DEMO")
    print("=" * 60)
    
    # Create synthetic pure component library
    n_wavenumbers = 100
    library = np.array([
        np.exp(-((np.arange(n_wavenumbers) - 30) ** 2) / 100),
        np.exp(-((np.arange(n_wavenumbers) - 50) ** 2) / 100),
        np.exp(-((np.arange(n_wavenumbers) - 70) ** 2) / 100),
    ])
    
    # Create synthetic mixtures
    n_samples = 10
    true_coeffs = np.random.dirichlet([1, 1, 1], size=n_samples)  # Sum to 1
    mixtures = (true_coeffs @ library) + np.random.randn(n_samples, n_wavenumbers) * 0.02
    
    # Unmix
    unmixer = NNLSUnmixer()
    unmixer.fit(library)
    estimated_coeffs = unmixer.transform(mixtures)
    
    print(f"Library shape (n_components, n_wavenumbers): {library.shape}")
    print(f"Mixtures shape: {mixtures.shape}")
    print(f"Estimated coefficients shape: {estimated_coeffs.shape}")
    print(f"\nExample true coefficients: {true_coeffs[0]}")
    print(f"Example estimated coefficients: {estimated_coeffs[0]}")
    
    # Reconstruct
    reconstructed = estimated_coeffs @ library
    reconstruction_error = np.mean((mixtures - reconstructed) ** 2)
    print(f"Mean reconstruction error (MSE): {reconstruction_error:.6f}")


def demo_plsr_vip():
    """Demonstrate PLSR with VIP scores."""
    print("\n" + "=" * 60)
    print("3. PLSR WITH VIP SCORES DEMO")
    print("=" * 60)
    
    # Synthetic data
    X, y = make_regression(n_samples=100, n_features=30, noise=10, random_state=42)
    
    # PLSR
    pls = PLSRegression(n_components=5, scale=True)
    pls.fit(X, y)
    y_pred = pls.predict(X)
    
    # VIP scores
    vip = VIPCalculator.calculate_vip(X, y, n_components=5)
    
    # Performance
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_pred)
    
    print(f"Training R² score: {r2:.4f}")
    print(f"Number of components: {pls.n_components}")
    print(f"VIP scores shape: {vip.shape}")
    print(f"Top 5 important features: {np.argsort(vip)[-5:]}")
    print(f"Top 5 VIP scores: {vip[np.argsort(vip)[-5:]]}")
    
    # Plot VIP
    fig, ax = plt.subplots(figsize=(10, 6))
    VIPCalculator.plot_vip(vip, feature_names=[f"X{i}" for i in range(30)], ax=ax)
    plt.tight_layout()
    plt.savefig("/tmp/vip_scores.png", dpi=100)
    print("✓ Saved vip_scores.png")


def demo_bootstrap_stability():
    """Demonstrate bootstrap stability analysis."""
    print("\n" + "=" * 60)
    print("4. BOOTSTRAP STABILITY ANALYSIS DEMO")
    print("=" * 60)
    
    from sklearn.linear_model import Ridge
    
    # Data
    X, y = make_regression(n_samples=50, n_features=10, noise=5, random_state=42)
    
    # Bootstrap stability
    bs = BootstrapStability(n_bootstrap=50, confidence=0.95, random_state=42)
    param_mean, param_std, param_ci = bs.assess_parameter_stability(
        X, y,
        fit_func=lambda x, yy: Ridge(alpha=1.0).fit(x, yy),
        param_func=lambda m: m.coef_
    )
    
    print(f"Parameter means (first 5): {param_mean[:5]}")
    print(f"Parameter stds (first 5): {param_std[:5]}")
    print(f"Parameter CI ranges (first 2):")
    print(f"  X0: [{param_ci[0, 0]:.4f}, {param_ci[0, 1]:.4f}]")
    print(f"  X1: [{param_ci[1, 0]:.4f}, {param_ci[1, 1]:.4f}]")
    
    # Stability ratios
    stability_ratio = StabilityIndex.parameter_stability_ratio(param_std, param_mean)
    print(f"\nStability ratios (first 5): {stability_ratio[:5]}")
    print(f"Mean stability: {np.mean(stability_ratio):.4f}")


def demo_agreement_analysis():
    """Demonstrate agreement analysis (Bland-Altman, Deming)."""
    print("\n" + "=" * 60)
    print("5. AGREEMENT ANALYSIS DEMO")
    print("=" * 60)
    
    # Two measurement methods with slight agreement
    method1 = np.linspace(1, 10, 30) + np.random.randn(30) * 0.5
    method2 = method1 + np.random.randn(30) * 0.3  # Systematic + random error
    
    # Bland-Altman
    ba = BlandAltmanAnalysis(confidence=0.95)
    mean_diff, std_diff, ll, ul, corr = ba.calculate(method1, method2)
    
    print("Bland-Altman Analysis:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std dev of differences: {std_diff:.4f}")
    print(f"  Lower LOA: {ll:.4f}")
    print(f"  Upper LOA: {ul:.4f}")
    print(f"  Correlation: {corr:.4f}")
    
    # Deming Regression
    deming = DemingRegression(variance_ratio=1.0)
    deming.fit(method1, method2)
    
    ccc = deming.get_concordance_correlation(method1, method2)
    print(f"\nDeming Regression:")
    print(f"  Slope: {deming.slope_:.4f}")
    print(f"  Intercept: {deming.intercept_:.4f}")
    print(f"  Concordance Correlation: {ccc:.4f}")
    
    # Plot Bland-Altman
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ba.plot(ax=axes[0])
    deming.plot(ax=axes[1])
    plt.tight_layout()
    plt.savefig("/tmp/agreement_analysis.png", dpi=100)
    print("✓ Saved agreement_analysis.png")


def demo_drift_monitoring():
    """Demonstrate EWMA drift monitoring."""
    print("\n" + "=" * 60)
    print("6. EWMA DRIFT MONITORING DEMO")
    print("=" * 60)
    
    # Reference data (normal operation)
    X_ref = np.random.randn(50, 3)
    
    # Stream data with drift in middle
    X_normal_1 = np.random.randn(20, 3)
    X_drift = np.random.randn(15, 3) + np.array([2, 1, 0.5])  # Drifted
    X_normal_2 = np.random.randn(15, 3)
    X_stream = np.vstack([X_normal_1, X_drift, X_normal_2])
    
    # Drift detection
    dd = DriftDetector(lambda_=0.2, min_samples=10)
    dd.initialize(X_ref)
    results = dd.process_stream(X_stream)
    
    summary = dd.get_drift_summary()
    print(f"Stream length: {summary['n_observations']}")
    print(f"EWMA alarms: {summary['n_alarms']}")
    print(f"Outliers: {summary['n_outliers']}")
    print(f"Alarm rate: {summary['alarm_rate']:.1%}")
    
    # Plot
    fig = dd.plot_drift_report(figsize=(14, 8))
    plt.tight_layout()
    plt.savefig("/tmp/drift_monitoring.png", dpi=100)
    print("✓ Saved drift_monitoring.png")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("FOODSPEC PROFESSIONAL CHEMOMETRICS CORE - FULL DEMO")
    print("=" * 60)
    
    try:
        demo_spectral_alignment()
    except Exception as e:
        print(f"✗ Alignment demo failed: {e}")
    
    try:
        demo_nnls_unmixing()
    except Exception as e:
        print(f"✗ Unmixing demo failed: {e}")
    
    try:
        demo_plsr_vip()
    except Exception as e:
        print(f"✗ PLSR demo failed: {e}")
    
    try:
        demo_bootstrap_stability()
    except Exception as e:
        print(f"✗ Bootstrap stability demo failed: {e}")
    
    try:
        demo_agreement_analysis()
    except Exception as e:
        print(f"✗ Agreement analysis demo failed: {e}")
    
    try:
        demo_drift_monitoring()
    except Exception as e:
        print(f"✗ Drift monitoring demo failed: {e}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("Generated plots:")
    print("  - /tmp/alignment_demo.png")
    print("  - /tmp/vip_scores.png")
    print("  - /tmp/agreement_analysis.png")
    print("  - /tmp/drift_monitoring.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
