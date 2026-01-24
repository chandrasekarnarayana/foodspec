"""Example demonstrating inference-only prediction from a deployment bundle."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from foodspec.deploy import load_bundle, predict_from_bundle, predict_from_bundle_path, save_bundle


def main() -> None:
    """Demonstrate making predictions on new data using a deployment bundle."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create and save a trained model bundle
        print("=" * 70)
        print("STEP 1: Creating deployment bundle with trained model")
        print("=" * 70)

        # Simulate training data
        n_features = 100
        n_samples_train = 200
        X_train = np.random.randn(n_samples_train, n_features) + np.linspace(0, 5, n_features)
        y_train = np.random.randint(0, 3, n_samples_train)

        # Train model
        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)
        print(f"✓ Trained model on {n_samples_train} samples with {n_features} features")

        # Save model
        model_path = tmp_path / "trained_model.joblib"
        joblib.dump(model, model_path)

        # Create bundle with preprocessing pipeline
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

        label_encoder = {0: "authentic", 1: "adulterated_palm_oil", 2: "adulterated_sunflower_oil"}

        x_grid = np.linspace(400, 4000, n_features)

        bundle_dir = tmp_path / "deployment_bundle"
        bundle_path = save_bundle(
            run_dir=tmp_path / "run",
            bundle_dir=bundle_dir,
            protocol=protocol,
            preprocess_pipeline=preprocess_pipeline,
            model_path=model_path,
            label_encoder=label_encoder,
            x_grid=x_grid,
            metadata_schema={"sample_id": "str", "batch": "str", "acquisition_date": "datetime"},
        )

        print(f"✓ Bundle saved to: {bundle_path}")
        print(f"  Contents: {len(list(bundle_path.iterdir()))} files")

        # 2. Create new test data in CSV format
        print("\n" + "=" * 70)
        print("STEP 2: Creating new test data CSV")
        print("=" * 70)

        input_csv = tmp_path / "new_samples.csv"
        n_test_samples = 10

        # Generate test data in long format (sample_id, wavenumber, intensity)
        test_data = []
        for sample_idx in range(n_test_samples):
            sample_id = f"TEST_SAMPLE_{sample_idx:03d}"
            # Generate spectra with some variation
            spectrum = np.random.randn(n_features) + np.linspace(0, 5, n_features) + sample_idx * 0.1

            for feature_idx, (wavenumber, intensity) in enumerate(zip(x_grid, spectrum)):
                test_data.append(
                    {
                        "sample_id": sample_id,
                        "wavenumber": wavenumber,
                        "intensity": intensity,
                    }
                )

        test_df = pd.DataFrame(test_data)
        test_df.to_csv(input_csv, index=False)

        print(f"✓ Created test data CSV with {n_test_samples} samples")
        print(f"  File: {input_csv}")
        print(f"  Total rows: {len(test_df)}")
        print(f"  Unique samples: {test_df['sample_id'].nunique()}")

        # 3. Load bundle and make predictions
        print("\n" + "=" * 70)
        print("STEP 3: Loading bundle and making predictions")
        print("=" * 70)

        bundle = load_bundle(bundle_path)
        print(f"✓ Bundle loaded successfully")
        print(f"  Model: {type(bundle['model']).__name__}")
        print(f"  Label encoder: {bundle['label_encoder']}")
        print(f"  Preprocessing steps: {len(bundle['preprocess_pipeline'])}")

        output_dir = tmp_path / "predictions"
        predictions_df = predict_from_bundle(
            bundle=bundle,
            input_csv=input_csv,
            output_dir=output_dir,
            save_probabilities=True,
        )

        print(f"✓ Predictions completed")
        print(f"  Output directory: {output_dir}")

        # 4. Examine predictions
        print("\n" + "=" * 70)
        print("STEP 4: Examining predictions")
        print("=" * 70)

        print("\nPredictions DataFrame:")
        print(predictions_df.to_string(index=False))

        print(f"\nPrediction distribution:")
        for label, count in predictions_df["predicted_label"].value_counts().items():
            print(f"  {label}: {count} samples")

        # Load and display probabilities
        prob_df = pd.read_csv(output_dir / "probabilities.csv")
        print(f"\nProbabilities DataFrame (first 5 samples):")
        print(prob_df.head().to_string(index=False))

        # 5. Demonstrate convenience function
        print("\n" + "=" * 70)
        print("STEP 5: Using convenience function (single call)")
        print("=" * 70)

        output_dir_2 = tmp_path / "predictions_v2"
        predictions_df_2 = predict_from_bundle_path(
            bundle_dir=bundle_dir,
            input_csv=input_csv,
            output_dir=output_dir_2,
        )

        print(f"✓ Predictions completed using convenience function")
        print(f"  Output directory: {output_dir_2}")
        print(f"  Number of predictions: {len(predictions_df_2)}")

        # 6. Demonstrate batch prediction statistics
        print("\n" + "=" * 70)
        print("STEP 6: Batch prediction statistics")
        print("=" * 70)

        print(f"Total samples processed: {len(predictions_df)}")
        print(f"Prediction files generated:")
        print(f"  - predictions.csv ({(output_dir / 'predictions.csv').stat().st_size} bytes)")
        print(f"  - probabilities.csv ({(output_dir / 'probabilities.csv').stat().st_size} bytes)")

        # Check confidence statistics
        prob_df = pd.read_csv(output_dir / "probabilities.csv")
        prob_cols = [col for col in prob_df.columns if col.startswith("prob_")]
        max_probs = prob_df[prob_cols].max(axis=1)

        print(f"\nConfidence statistics:")
        print(f"  Mean confidence: {max_probs.mean():.3f}")
        print(f"  Min confidence: {max_probs.min():.3f}")
        print(f"  Max confidence: {max_probs.max():.3f}")
        print(f"  Std confidence: {max_probs.std():.3f}")

        # Identify high-confidence predictions
        high_conf_threshold = 0.7
        high_conf_count = (max_probs > high_conf_threshold).sum()
        print(f"\nHigh-confidence predictions (>{high_conf_threshold}): {high_conf_count}/{len(predictions_df)}")

        print("\n" + "=" * 70)
        print("✓ Prediction demo completed successfully!")
        print("=" * 70)


if __name__ == "__main__":
    main()
