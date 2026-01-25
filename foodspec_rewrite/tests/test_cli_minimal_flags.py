"""Tests for CLI run command with minimal flags."""

import pytest
from typer.testing import CliRunner

from foodspec.cli.commands.run import generate_minimal_protocol


def test_generate_minimal_protocol():
    """Test auto-generation of protocol from minimal parameters."""
    protocol = generate_minimal_protocol(
        input_path="data.csv",
        task_name="classify",
        modality="raman",
        label="class",
    )
    
    # Check DataSpec
    assert protocol.data.input == "data.csv"
    assert protocol.data.modality == "raman"
    assert protocol.data.label == "class"
    assert "sample_id" in protocol.data.metadata_map
    assert "modality" in protocol.data.metadata_map
    assert "label" in protocol.data.metadata_map
    
    # Check TaskSpec
    assert protocol.task.name == "classify"
    assert protocol.task.objective == "classification"
    
    # Check defaults are applied
    assert protocol.preprocess.recipe == "basic"
    assert protocol.model.family == "sklearn"
    assert protocol.model.estimator == "logreg"
    assert protocol.validation.scheme == "train_test_split"
    assert "accuracy" in protocol.validation.metrics


def test_generate_minimal_protocol_default_label():
    """Test protocol generation with default label."""
    protocol = generate_minimal_protocol(
        input_path="data.csv",
        task_name="test",
        modality="nir",
    )
    
    assert protocol.data.label == "label"


def test_minimal_flags_missing_all():
    """Test error when neither protocol nor minimal flags provided."""
    from foodspec.cli.main import app
    
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--outdir", "test_output"])
    
    assert result.exit_code == 2
    # Error message goes to stderr


def test_minimal_flags_missing_some():
    """Test error when only some minimal flags provided."""
    from foodspec.cli.main import app
    
    runner = CliRunner()
    
    # Missing --modality
    result = runner.invoke(app, [
        "run",
        "--input", "data.csv",
        "--task", "classify",
        "--outdir", "test_output",
    ])
    
    assert result.exit_code == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
