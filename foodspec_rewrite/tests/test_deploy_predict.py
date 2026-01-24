"""Tests for inference-only prediction from bundles."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from foodspec.deploy import load_bundle, predict_from_bundle, predict_from_bundle_path, save_bundle


def test_predict_from_bundle_basic(tmp_path: Path) -> None:
    """Test basic prediction from bundle."""
    # Create and save bundle
    X_train = np.random.randn(50, 10)
    y_train = np.random.randint(0, 2, 50)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    protocol = {"version": "2.0.0"}
    label_encoder = {0: "class_a", 1: "class_b"}

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol=protocol,
        model_path=model_path,
        label_encoder=label_encoder,
    )

    # Create test input CSV
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(5):
        for feature_idx in range(10):
            test_data.append(
                {
                    "sample_id": f"sample_{sample_idx}",
                    "wavenumber": feature_idx,
                    "intensity": np.random.randn(),
                }
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    # Make predictions
    bundle = load_bundle(bundle_dir)
    output_dir = tmp_path / "predictions"
    predictions_df = predict_from_bundle(bundle, input_csv, output_dir)

    # Check predictions
    assert len(predictions_df) == 5
    assert "sample_id" in predictions_df.columns
    assert "predicted_class" in predictions_df.columns
    assert "predicted_label" in predictions_df.columns
    assert all(predictions_df["predicted_class"].isin([0, 1]))
    assert all(predictions_df["predicted_label"].isin(["class_a", "class_b"]))

    # Check output files
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "probabilities.csv").exists()

    # Check probabilities file
    prob_df = pd.read_csv(output_dir / "probabilities.csv")
    assert len(prob_df) == 5
    assert "sample_id" in prob_df.columns
    assert "prob_class_a" in prob_df.columns
    assert "prob_class_b" in prob_df.columns


def test_predict_from_bundle_with_preprocessing(tmp_path: Path) -> None:
    """Test prediction with preprocessing pipeline."""
    # Create and save bundle with preprocessing
    X_train = np.random.randn(50, 10) + 10  # Offset for SNV to have effect
    y_train = np.random.randint(0, 3, 50)
    model = LogisticRegression(random_state=42, max_iter=500)  # Increased to avoid warning
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    protocol = {"version": "2.0.0"}
    preprocess_pipeline = [{"method": "snv", "params": {}}]
    label_encoder = {0: "authentic", 1: "adulterated_a", 2: "adulterated_b"}

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol=protocol,
        preprocess_pipeline=preprocess_pipeline,
        model_path=model_path,
        label_encoder=label_encoder,
    )

    # Create test input
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(3):
        for feature_idx in range(10):
            test_data.append(
                {
                    "sample_id": f"S{sample_idx}",
                    "wavenumber": feature_idx,
                    "intensity": np.random.randn() + 10,
                }
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    # Make predictions
    bundle = load_bundle(bundle_dir)
    output_dir = tmp_path / "predictions"
    predictions_df = predict_from_bundle(bundle, input_csv, output_dir)

    # Check predictions
    assert len(predictions_df) == 3
    assert all(predictions_df["predicted_class"].isin([0, 1, 2]))
    assert all(
        predictions_df["predicted_label"].isin(["authentic", "adulterated_a", "adulterated_b"])
    )


def test_predict_from_bundle_no_probabilities(tmp_path: Path) -> None:
    """Test prediction without saving probabilities."""
    # Create simple bundle
    X_train = np.random.randn(20, 5)
    y_train = np.random.randint(0, 2, 20)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
    )

    # Create input
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(2):
        for feature_idx in range(5):
            test_data.append(
                {"sample_id": f"S{sample_idx}", "wavenumber": feature_idx, "intensity": 1.0}
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    # Predict without probabilities
    bundle = load_bundle(bundle_dir)
    output_dir = tmp_path / "predictions"
    predict_from_bundle(bundle, input_csv, output_dir, save_probabilities=False)

    # Check only predictions file exists
    assert (output_dir / "predictions.csv").exists()
    assert not (output_dir / "probabilities.csv").exists()


def test_predict_from_bundle_missing_model(tmp_path: Path) -> None:
    """Test error when bundle has no model."""
    # Create bundle without model
    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
    )

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"sample_id": ["S1"], "wavenumber": [0], "intensity": [1.0]}).to_csv(
        input_csv, index=False
    )

    bundle = load_bundle(bundle_dir)
    output_dir = tmp_path / "predictions"

    with pytest.raises(ValueError, match="Bundle does not contain a trained model"):
        predict_from_bundle(bundle, input_csv, output_dir)


def test_predict_from_bundle_missing_input_file(tmp_path: Path) -> None:
    """Test error when input CSV is missing."""
    # Create bundle
    X_train = np.random.randn(20, 5)
    y_train = np.random.randint(0, 2, 20)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
    )

    bundle = load_bundle(bundle_dir)
    input_csv = tmp_path / "nonexistent.csv"
    output_dir = tmp_path / "predictions"

    with pytest.raises(FileNotFoundError, match="Input CSV not found"):
        predict_from_bundle(bundle, input_csv, output_dir)


def test_predict_from_bundle_missing_columns(tmp_path: Path) -> None:
    """Test error when input CSV is missing required columns."""
    # Create bundle
    X_train = np.random.randn(20, 5)
    y_train = np.random.randint(0, 2, 20)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
    )

    # Create input with missing columns
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"sample_id": ["S1"], "wavenumber": [0]}).to_csv(input_csv, index=False)

    bundle = load_bundle(bundle_dir)
    output_dir = tmp_path / "predictions"

    with pytest.raises(ValueError, match="missing required columns"):
        predict_from_bundle(bundle, input_csv, output_dir)


def test_predict_from_bundle_path_convenience(tmp_path: Path) -> None:
    """Test convenience function that loads bundle and predicts."""
    # Create bundle
    X_train = np.random.randn(30, 8)
    y_train = np.random.randint(0, 2, 30)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
        label_encoder={0: "neg", 1: "pos"},
    )

    # Create input
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(4):
        for feature_idx in range(8):
            test_data.append(
                {"sample_id": f"T{sample_idx}", "wavenumber": feature_idx, "intensity": 0.5}
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    # Use convenience function
    output_dir = tmp_path / "predictions"
    predictions_df = predict_from_bundle_path(bundle_dir, input_csv, output_dir)

    # Check results
    assert len(predictions_df) == 4
    assert all(predictions_df["predicted_label"].isin(["neg", "pos"]))
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "probabilities.csv").exists()


def test_predict_from_bundle_custom_column_names(tmp_path: Path) -> None:
    """Test prediction with custom column names."""
    # Create bundle
    X_train = np.random.randn(25, 6)
    y_train = np.random.randint(0, 2, 25)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
    )

    # Create input with custom column names
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(2):
        for feature_idx in range(6):
            test_data.append(
                {"id": f"ID{sample_idx}", "freq": feature_idx, "value": 1.5}
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    # Predict with custom column names
    bundle = load_bundle(bundle_dir)
    output_dir = tmp_path / "predictions"
    predictions_df = predict_from_bundle(
        bundle,
        input_csv,
        output_dir,
        sample_id_col="id",
        wavenumber_col="freq",
        intensity_col="value",
    )

    # Check results
    assert len(predictions_df) == 2
    assert all(predictions_df["sample_id"].str.startswith("ID"))


def test_predict_from_bundle_multiclass(tmp_path: Path) -> None:
    """Test prediction with multiclass classification."""
    # Create bundle with 4 classes
    X_train = np.random.randn(100, 12)
    y_train = np.random.randint(0, 4, 100)
    model = LogisticRegression(random_state=42, max_iter=300)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    label_encoder = {0: "type_a", 1: "type_b", 2: "type_c", 3: "type_d"}

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
        label_encoder=label_encoder,
    )

    # Create input
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(6):
        for feature_idx in range(12):
            test_data.append(
                {"sample_id": f"M{sample_idx}", "wavenumber": feature_idx, "intensity": 2.0}
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    # Predict
    bundle = load_bundle(bundle_dir)
    output_dir = tmp_path / "predictions"
    predictions_df = predict_from_bundle(bundle, input_csv, output_dir)

    # Check results
    assert len(predictions_df) == 6
    assert all(predictions_df["predicted_class"].isin([0, 1, 2, 3]))
    assert all(predictions_df["predicted_label"].isin(label_encoder.values()))

    # Check probabilities
    prob_df = pd.read_csv(output_dir / "probabilities.csv")
    assert prob_df.shape == (6, 5)  # 6 samples, 1 ID + 4 probability columns
    assert "prob_type_a" in prob_df.columns
    assert "prob_type_b" in prob_df.columns
    assert "prob_type_c" in prob_df.columns
    assert "prob_type_d" in prob_df.columns


def test_predict_from_bundle_unsorted_wavenumbers(tmp_path: Path) -> None:
    """Test that wavenumbers are correctly sorted before prediction."""
    # Create bundle
    X_train = np.random.randn(40, 7)
    y_train = np.random.randint(0, 2, 40)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        run_dir=tmp_path,
        bundle_dir=bundle_dir,
        protocol={"version": "2.0.0"},
        model_path=model_path,
    )

    # Create input with unsorted wavenumbers
    input_csv = tmp_path / "input.csv"
    test_data = []
    for sample_idx in range(3):
        # Add wavenumbers in reverse order
        for feature_idx in range(6, -1, -1):
            test_data.append(
                {"sample_id": f"U{sample_idx}", "wavenumber": feature_idx, "intensity": 0.8}
            )
    pd.DataFrame(test_data).to_csv(input_csv, index=False)

    # Predict
    bundle = load_bundle(bundle_dir)
    output_dir = tmp_path / "predictions"
    predictions_df = predict_from_bundle(bundle, input_csv, output_dir)

    # Should not raise and should produce valid predictions
    assert len(predictions_df) == 3
    assert all(predictions_df["predicted_class"].isin([0, 1]))
