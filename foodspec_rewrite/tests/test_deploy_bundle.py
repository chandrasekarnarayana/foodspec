"""Tests for deployment bundle export and import."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from foodspec.deploy import load_bundle, save_bundle


def test_save_load_minimal_bundle(tmp_path: Path) -> None:
    """Test saving and loading a minimal bundle with only protocol."""
    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "bundle"

    protocol = {"version": "2.0.0", "data": {"modality": "raman"}}

    # Save bundle
    bundle_path = save_bundle(run_dir=run_dir, bundle_dir=bundle_dir, protocol=protocol)

    # Check bundle exists
    assert bundle_path.exists()
    assert (bundle_path / "bundle_manifest.json").exists()
    assert (bundle_path / "protocol.json").exists()
    assert (bundle_path / "package_versions.json").exists()

    # Load bundle
    bundle = load_bundle(bundle_path)

    assert bundle["protocol"] == protocol
    assert bundle["preprocess_pipeline"] is None
    assert bundle["model"] is None
    assert bundle["label_encoder"] is None
    assert bundle["x_grid"] is None
    assert bundle["metadata_schema"] is None
    assert "python" in bundle["package_versions"]
    assert "numpy" in bundle["package_versions"]


def test_save_load_full_bundle(tmp_path: Path) -> None:
    """Test saving and loading a full bundle with all components."""
    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "bundle"
    model_file = tmp_path / "model.joblib"

    # Create dummy model
    model = LogisticRegression(random_state=42)
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, 10)
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)

    protocol = {"version": "2.0.0", "data": {"modality": "raman"}}
    preprocess_pipeline = [
        {"method": "snv", "params": {}},
        {"method": "savgol", "params": {"window_length": 11, "polyorder": 2}},
    ]
    label_encoder = {0: "authentic", 1: "adulterated"}
    x_grid = np.array([400.0, 500.0, 600.0])
    metadata_schema = {"sample_id": "str", "temperature": "float"}

    # Save bundle
    bundle_path = save_bundle(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        protocol=protocol,
        preprocess_pipeline=preprocess_pipeline,
        model_path=model_file,
        label_encoder=label_encoder,
        x_grid=x_grid,
        metadata_schema=metadata_schema,
    )

    # Check all components exist
    assert (bundle_path / "protocol.json").exists()
    assert (bundle_path / "preprocess_pipeline.json").exists()
    assert (bundle_path / "model.joblib").exists()
    assert (bundle_path / "label_encoder.json").exists()
    assert (bundle_path / "x_grid.npy").exists()
    assert (bundle_path / "metadata_schema.json").exists()

    # Load bundle
    bundle = load_bundle(bundle_path)

    assert bundle["protocol"] == protocol
    assert bundle["preprocess_pipeline"] == preprocess_pipeline
    assert isinstance(bundle["model"], LogisticRegression)
    assert bundle["label_encoder"] == label_encoder
    np.testing.assert_array_equal(bundle["x_grid"], x_grid)
    assert bundle["metadata_schema"] == metadata_schema
    assert bundle["package_versions"]["python"].startswith("3.")


def test_load_missing_manifest(tmp_path: Path) -> None:
    """Test error handling when manifest is missing."""
    bundle_dir = tmp_path / "empty_bundle"
    bundle_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Bundle manifest not found"):
        load_bundle(bundle_dir)


def test_load_unsupported_format(tmp_path: Path) -> None:
    """Test error handling for unsupported bundle format."""
    bundle_dir = tmp_path / "bad_bundle"
    bundle_dir.mkdir()

    # Create manifest with unsupported version
    manifest = {"bundle_format_version": "99.0.0", "contents": {}}
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsupported bundle format"):
        load_bundle(bundle_dir)


def test_save_bundle_creates_directories(tmp_path: Path) -> None:
    """Test that save_bundle creates directories if they don't exist."""
    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "nested" / "bundle" / "dir"

    protocol = {"version": "2.0.0"}

    # Should not raise
    bundle_path = save_bundle(run_dir=run_dir, bundle_dir=bundle_dir, protocol=protocol)

    assert bundle_path.exists()
    assert (bundle_path / "protocol.json").exists()


def test_bundle_roundtrip_with_model(tmp_path: Path) -> None:
    """Test full roundtrip with model predictions."""
    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "bundle"
    model_file = tmp_path / "model.joblib"

    # Train model
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)

    protocol = {"version": "2.0.0"}
    label_encoder = {0: "class_a", 1: "class_b"}

    # Save
    bundle_path = save_bundle(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        protocol=protocol,
        model_path=model_file,
        label_encoder=label_encoder,
    )

    # Load
    bundle = load_bundle(bundle_path)

    # Test predictions
    X_test = np.array([[2, 3], [6, 7]])
    predictions = bundle["model"].predict(X_test)

    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)
    assert bundle["label_encoder"][predictions[0]] in ["class_a", "class_b"]


def test_label_encoder_int_keys_preserved(tmp_path: Path) -> None:
    """Test that label encoder integer keys are preserved after save/load."""
    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "bundle"

    protocol = {"version": "2.0.0"}
    label_encoder = {0: "zero", 1: "one", 2: "two"}

    bundle_path = save_bundle(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        protocol=protocol,
        label_encoder=label_encoder,
    )

    bundle = load_bundle(bundle_path)

    # Check keys are integers, not strings
    assert all(isinstance(k, int) for k in bundle["label_encoder"].keys())
    assert bundle["label_encoder"] == label_encoder


def test_x_grid_array_preserved(tmp_path: Path) -> None:
    """Test that x_grid numpy array is correctly saved and loaded."""
    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "bundle"

    protocol = {"version": "2.0.0"}
    x_grid = np.linspace(400, 4000, 1000)

    bundle_path = save_bundle(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        protocol=protocol,
        x_grid=x_grid,
    )

    bundle = load_bundle(bundle_path)

    np.testing.assert_array_almost_equal(bundle["x_grid"], x_grid)
    assert bundle["x_grid"].shape == x_grid.shape


def test_package_versions_recorded(tmp_path: Path) -> None:
    """Test that package versions are captured."""
    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "bundle"

    protocol = {"version": "2.0.0"}

    bundle_path = save_bundle(run_dir=run_dir, bundle_dir=bundle_dir, protocol=protocol)
    bundle = load_bundle(bundle_path)

    versions = bundle["package_versions"]
    assert "python" in versions
    assert "numpy" in versions
    assert "pandas" in versions
    assert "scikit-learn" in versions

    # Check that versions are strings
    assert isinstance(versions["python"], str)
    assert isinstance(versions["numpy"], str)
