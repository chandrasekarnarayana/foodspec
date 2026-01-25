"""Baseline correction transformers."""
from __future__ import annotations


from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["ALSBaseline", "RubberbandBaseline", "PolynomialBaseline"]


class ALSBaseline(BaseEstimator, TransformerMixin):
    """Asymmetric Least Squares (ALS) baseline correction for spectroscopy.

    Fits a smooth baseline using weighted least-squares with asymmetric penalties,
    exploiting the fact that spectral peaks are positive deviations. Iteratively
    reweights residuals: positive residuals (peaks) get low weight, negative
    residuals (baseline) get high weight.

    **Algorithm (Eilers & Boelens, 2005):**

    Minimize: ||w * (y - baseline)||² + λ * ||D²(baseline)||²

    where:
    - w = asymmetric weights (updated each iteration)
    - λ = smoothness parameter (larger = smoother)
    - D² = second derivative operator (penalizes curvature)

    **Theory:**
    ALS is superior to polynomial fitting for spectroscopy because:
    - Preserves sharp peaks (adaptive weighting)
    - Handles broad fluorescence/scattering backgrounds
    - No assumption about baseline shape (polynomial assumes specific form)
    - Computationally efficient (sparse linear algebra)

    **Parameter Significance:**

    | Parameter | Typical Range | Effect | Red Flags |
    |-----------|---------------|--------|----------|
    | λ (lambda_) | 10² – 10⁶ | Higher = smoother baseline | λ > 10⁶: removes real broad peaks |
    | p | 0.001 – 0.1 | Lower = more asymmetric | p < 0.001: baseline undershoot artifacts |
    | max_iter | 5 – 20 | More iterations = convergence | >20: diminishing returns |

    **Common Settings:**
    - **Raman (fluorescence):** λ = 10⁵, p = 0.001 (very asymmetric)
    - **FTIR (scattering):** λ = 10⁴, p = 0.01 (moderate asymmetry)
    - **NIR (baseline drift):** λ = 10³, p = 0.05 (less asymmetric)

    **Quality Metrics:**
    - Baseline residual RMSE should be < 1% of signal range
    - Edge artifacts: |mean(first 20 points after correction)| < 0.1
    - SNR improvement: 3–10 dB typical

    **When to Use:**
    - Broad, slowly-varying baseline (fluorescence, scattering)
    - Preserving sharp peaks is critical
    - Raman, FTIR, NIR spectroscopy

    **When NOT to Use:**
    - Baseline has sharp features (use rubberband)
    - Negative peaks expected (use polynomial)
    - Extreme noise (smooth first with Savitzky-Golay)

    Parameters:
        lambda_ (float): Smoothness parameter. Default 1e5. Higher = smoother.
        p (float): Asymmetry parameter (0 < p < 1). Default 0.001. Lower = more asymmetric.
        max_iter (int): Maximum iterations. Default 10.

    Attributes:
        lambda_ (float): Stored smoothness parameter
        p (float): Stored asymmetry parameter
        max_iter (int): Stored max iterations

    Examples:
        >>> from foodspec.preprocess import ALSBaseline
        >>> import numpy as np
        >>> # Simulate spectrum with baseline
        >>> x = np.linspace(0, 1, 100)
        >>> baseline = 5 * np.exp(-x)  # Exponential background
        >>> peaks = np.exp(-100 * (x - 0.5)**2)  # Gaussian peak
        >>> spectrum = (baseline + peaks).reshape(1, -1)
        >>>
        >>> # Apply ALS correction
        >>> als = ALSBaseline(lambda_=1e4, p=0.01)
        >>> corrected = als.fit_transform(spectrum)
        >>>
        >>> # Check baseline removal
        >>> assert corrected.mean() < spectrum.mean()  # Baseline removed
        >>> assert corrected.max() > 0.5  # Peak preserved

    References:
        Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with
        asymmetric least squares smoothing. Leiden University Medical Centre
        Report, 1(1), 5.

    See Also:
        - Theory: docs/preprocessing/baseline_correction.md
        - Tutorial: docs/02-tutorials/beginner/oil_discrimination_basic.md
        - API: docs/api/preprocessing.md
    """

    def __init__(self, lambda_: float = 1e5, p: float = 0.001, max_iter: int = 10):
        self.lambda_ = lambda_
        self.p = p
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ALSBaseline":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        n_samples, n_wavenumbers = X.shape
        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")
        if not (0 < self.p < 1):
            raise ValueError("p must be in (0, 1).")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        D = _second_derivative_matrix(n_wavenumbers)
        baselines = np.zeros_like(X)
        for i, y in enumerate(X):
            w = np.ones(n_wavenumbers)
            for _ in range(self.max_iter):
                W = diags(w, 0, shape=(n_wavenumbers, n_wavenumbers))
                Z = W + self.lambda_ * (D.T @ D)
                z = spsolve(Z, w * y)
                w = self.p * (y > z) + (1 - self.p) * (y < z)
            baseline = z
            corrected_candidate = y - baseline
            edge_mean = corrected_candidate[: min(20, n_wavenumbers)].mean()
            if not np.isfinite(baseline).all() or abs(edge_mean) > 1.0:
                baseline = _poly_baseline(y, degree=2)
            baselines[i, :] = baseline
        return X - baselines


class RubberbandBaseline(BaseEstimator, TransformerMixin):
    """Convex hull (rubberband) baseline correction for spectroscopy.

    Constructs baseline by fitting a 'rubberband' stretched beneath the spectrum
    using the lower convex hull. Ideal for spectra with baseline composed of
    discrete features (not smooth curves).

    **Algorithm:**
    1. Compute lower convex hull of (wavenumber, intensity) points
    2. Interpolate hull points to create baseline
    3. Subtract baseline from spectrum

    **Theory:**
    Convex hull baseline assumes:
    - All peaks are positive deviations
    - Baseline connects lowest points in spectrum
    - No smoothness constraint (unlike ALS)
    - Geometrically intuitive (physical 'rubberband')

    **When to Use:**
    - Baseline has sharp features or discontinuities
    - ALS over-smooths (creates artifacts)
    - Simple, parameter-free method needed
    - LIBS, XRF, or emission spectra with sharp baseline features

    **When NOT to Use:**
    - Noisy spectra (hull follows noise peaks)
    - Smooth baseline expected (use ALS instead)
    - Negative peaks present (convex hull assumption violated)

    **Advantages:**
    - No parameters to tune (automatic)
    - Fast computation (O(n log n))
    - Handles non-smooth baselines

    **Disadvantages:**
    - Sensitive to noise (outliers become hull points)
    - No smoothness guarantee
    - May underestimate baseline if no valley points exist

    Examples:
        >>> from foodspec.preprocess import RubberbandBaseline
        >>> import numpy as np
        >>> # Simulate spectrum with step baseline
        >>> x = np.arange(100)
        >>> baseline = np.where(x < 50, 2.0, 3.0)  # Step function
        >>> peaks = np.exp(-0.1 * (x - 25)**2) + np.exp(-0.1 * (x - 75)**2)
        >>> spectrum = (baseline + peaks).reshape(1, -1)
        >>>
        >>> # Apply rubberband correction
        >>> rb = RubberbandBaseline()
        >>> corrected = rb.fit_transform(spectrum)
        >>>
        >>> # Check baseline removal
        >>> assert corrected.mean() < spectrum.mean()
        >>> assert corrected.min() >= -0.5  # No severe undershoot

    See Also:
        - ALSBaseline: Smooth baseline with asymmetric weighting
        - PolynomialBaseline: Parametric polynomial baseline
        - Theory: docs/preprocessing/baseline_correction.md
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RubberbandBaseline":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")

        n_samples, n_wavenumbers = X.shape
        wavenumbers = np.arange(n_wavenumbers)
        corrected = np.zeros_like(X)

        for i, y in enumerate(X):
            lower = _lower_hull_indices(wavenumbers, y)
            baseline = np.interp(wavenumbers, wavenumbers[lower], y[lower])
            corrected[i, :] = y - baseline

        return corrected


class PolynomialBaseline(BaseEstimator, TransformerMixin):
    """Polynomial baseline correction by least-squares fitting.

    Fits a polynomial of specified degree to the spectrum and subtracts it.
    Suitable for smooth, slowly-varying baselines when the functional form
    is approximately polynomial.

    **Algorithm:**
    1. Fit polynomial P(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ to spectrum
    2. Subtract polynomial baseline from spectrum

    **Theory:**
    Polynomial baseline assumes baseline follows a low-order polynomial.
    Common in:
    - Linear drift (degree 1)
    - Quadratic curvature (degree 2)
    - Cubic smooth baselines (degree 3)

    **Degree Selection Guide:**

    | Degree | Baseline Shape | Use Case | Red Flags |
    |--------|----------------|----------|----------|
    | 1 | Linear drift | Instrument drift, temperature effects | Removes broad peaks |
    | 2 | Parabolic | Scattering, lens effects | Overfits to noise if noisy |
    | 3 | Cubic smooth | General smooth baseline | Removes broad features |
    | 4+ | Complex curves | Rarely justified | Overfitting, Runge phenomenon |

    **When to Use:**
    - Baseline is smooth and slowly-varying
    - Physical model suggests polynomial (e.g., temperature drift = linear)
    - Simple, interpretable method needed

    **When NOT to Use:**
    - Baseline has sharp features (use rubberband)
    - Fluorescence or scattering (use ALS)
    - High degree needed (indicates overfitting)

    **Quality Metrics:**
    - Residual baseline RMSE < 1% of signal range
    - Polynomial degree justified by F-test (nested model comparison)
    - No Runge phenomenon (oscillations at edges)

    Parameters:
        degree (int): Polynomial degree. Default 3. Higher = more flexible.

    Attributes:
        degree (int): Stored polynomial degree

    Examples:
        >>> from foodspec.preprocess import PolynomialBaseline
        >>> import numpy as np
        >>> # Simulate spectrum with quadratic baseline
        >>> x = np.linspace(0, 1, 100)
        >>> baseline = 2 + 3*x + 1*x**2  # Quadratic
        >>> peaks = np.exp(-100 * (x - 0.5)**2)  # Gaussian peak
        >>> spectrum = (baseline + peaks).reshape(1, -1)
        >>>
        >>> # Apply polynomial correction
        >>> poly = PolynomialBaseline(degree=2)
        >>> corrected = poly.fit_transform(spectrum)
        >>>
        >>> # Check baseline removal
        >>> assert abs(corrected.mean()) < 0.5  # Baseline removed
        >>> assert corrected.max() > 0.5  # Peak preserved

    See Also:
        - ALSBaseline: For fluorescence/scattering
        - RubberbandBaseline: For non-smooth baselines
        - Theory: docs/preprocessing/baseline_correction.md
    """

    def __init__(self, degree: int = 3):
        self.degree = degree

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PolynomialBaseline":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_wavenumbers).")
        if self.degree < 0:
            raise ValueError("degree must be non-negative.")

        n_samples, n_wavenumbers = X.shape
        x_axis = np.linspace(0, 1, n_wavenumbers)
        corrected = np.zeros_like(X)
        for i, y in enumerate(X):
            coefs = np.polyfit(x_axis, y, deg=self.degree)
            baseline = np.polyval(coefs, x_axis)
            corrected[i, :] = y - baseline
        return corrected


def _second_derivative_matrix(n: int) -> csc_matrix:
    diagonals = [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)]
    offsets = [0, 1, 2]
    return diags(diagonals, offsets, shape=(n - 2, n))


def _poly_baseline(y: np.ndarray, degree: int = 2) -> np.ndarray:
    x_axis = np.linspace(0, 1, y.shape[0])
    coefs = np.polyfit(x_axis, y, deg=degree)
    return np.polyval(coefs, x_axis)


def _lower_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute lower hull indices using monotone chain."""

    order = np.argsort(x)
    lower: list[int] = []

    def cross(o: int, a: int, b: int) -> float:
        return (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o])

    for idx in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], idx) <= 0:
            lower.pop()
        lower.append(idx)

    return np.array(lower, dtype=int)
