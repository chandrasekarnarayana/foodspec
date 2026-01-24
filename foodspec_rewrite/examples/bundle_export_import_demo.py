"""Example demonstrating bundle export and import for model deployment."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from foodspec.deploy import load_bundle, save_bundle


def main() -> None:
    """Demonstrate saving and loading a deployment bundle."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create a trained model
        print("Training model...")
        X_train = np.random.randn(100, 50)  # 100 samples, 50 features
        y_train = np.random.randint(0, 3, 100)  # 3 classes

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)

        # Save model to file
        model_path = tmp_path / "trained_model.joblib"
        joblib.dump(model, model_path)

        # 2. Create bundle with all components
        print("Creating deployment bundle...")

        protocol = {
            "version": "2.0.0",
            "data": {"modality": "raman", "wavenumber_range": [400, 4000]},
            "preprocessing": {
                "steps": [
                    {"method": "snv", "params": {}},
                    {"method": "savgol", "params": {"window_length": 11, "polyorder": 2}},
                ]
            },
        }

        preprocess_pipeline = [
            {"method": "snv", "params": {}},
            {"method": "savgol", "params": {"window_length": 11, "polyorder": 2}},
        ]

        label_encoder = {0: "authentic", 1: "adulterated_oil_a", 2: "adulterated_oil_b"}

        x_grid = np.linspace(400, 4000, 50)  # Wavenumber grid

        metadata_schema = {
            "sample_id": "str",
            "batch": "str",
            "temperature": "float",
            "acquisition_date": "datetime",
        }

        bundle_dir = tmp_path / "deployment_bundle"
        bundle_path = save_bundle(
            run_dir=tmp_path / "run",
            bundle_dir=bundle_dir,
            protocol=protocol,
            preprocess_pipeline=preprocess_pipeline,
            model_path=model_path,
            label_encoder=label_encoder,
            x_grid=x_grid,
            metadata_schema=metadata_schema,
        )

        print(f"Bundle saved to: {bundle_path}")
        print(f"Bundle contents:")
        for file in sorted(bundle_path.iterdir()):
            print(f"  - {file.name}")

        # 3. Load bundle for deployment
        print("\nLoading bundle for deployment...")
        bundle = load_bundle(bundle_path)

        print(f"Protocol version: {bundle['protocol']['version']}")
        print(f"Model type: {type(bundle['model']).__name__}")
        print(f"Label encoder: {bundle['label_encoder']}")
        print(f"X grid shape: {bundle['x_grid'].shape}")
        print(f"Python version: {bundle['package_versions']['python']}")
        print(f"NumPy version: {bundle['package_versions']['numpy']}")
        print(f"scikit-learn version: {bundle['package_versions']['scikit-learn']}")

        # 4. Use loaded model for prediction
        print("\nMaking predictions with loaded model...")
        X_test = np.random.randn(5, 50)
        predictions = bundle["model"].predict(X_test)
        probabilities = bundle["model"].predict_proba(X_test)

        print(f"Predictions: {predictions}")
        print(f"Predicted classes: {[bundle['label_encoder'][p] for p in predictions]}")
        print(f"Probabilities shape: {probabilities.shape}")

        # 5. Demonstrate preprocessing pipeline reconstruction
        print("\nPreprocessing pipeline:")
        for i, step in enumerate(bundle["preprocess_pipeline"], 1):
            print(f"  {i}. {step['method']}: {step['params']}")


if __name__ == "__main__":
    main()
