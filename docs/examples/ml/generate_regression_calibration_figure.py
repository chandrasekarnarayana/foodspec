"""
Generate a simple regression calibration plot (predicted vs true) for foodspec docs.

Run from repository root:
    python docs/examples/ml/generate_regression_calibration_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from foodspec.chemometrics.models import make_pls_regression
from foodspec.chemometrics.validation import compute_regression_metrics


def main() -> None:
    rng = np.random.default_rng(42)
    n_samples = 80
    n_features = 15

    # Synthetic spectral features with a linear relationship to target + noise
    X = rng.normal(0, 1, size=(n_samples, n_features))
    true_coefs = rng.normal(0.5, 0.2, size=n_features)
    y = X @ true_coefs + rng.normal(0, 0.5, size=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = make_pls_regression(n_components=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).ravel()

    metrics = compute_regression_metrics(y_test, y_pred)
    print("Calibration metrics:")
    print(metrics)

    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_path = assets_dir / "regression_calibration.png"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred, alpha=0.7, label="Samples")
    lims = [
        min(y_test.min(), y_pred.min()) - 0.5,
        max(y_test.max(), y_pred.max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", label="Ideal (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Regression calibration: predicted vs true")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved calibration plot to {fig_path}")


if __name__ == "__main__":
    main()
