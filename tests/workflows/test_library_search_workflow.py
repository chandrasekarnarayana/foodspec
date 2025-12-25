"""Comprehensive tests for LibrarySearchWorkflow class."""

import numpy as np
import pandas as pd

from foodspec.workflows.library_search import LibrarySearchWorkflow


class TestLibrarySearchWorkflow:
    """Test suite for LibrarySearchWorkflow."""

    def test_workflow_initialization_default(self):
        """Test default initialization."""
        workflow = LibrarySearchWorkflow()
        assert workflow.metric == "cosine"
        assert workflow.top_k == 5

    def test_workflow_initialization_custom(self):
        """Test initialization with custom parameters."""
        workflow = LibrarySearchWorkflow(metric="sid", top_k=3)
        assert workflow.metric == "sid"
        assert workflow.top_k == 3

    def test_run_basic(self):
        """Test run method returns correct structure."""
        workflow = LibrarySearchWorkflow(metric="cosine", top_k=2)
        library_df = pd.DataFrame({
            "name": ["A", "B", "C"],
            "col1": [1.0, 2.0, 3.0],
            "col2": [4.0, 5.0, 6.0],
        })
        query = np.array([1.0, 2.0, 3.0])

        result = workflow.run(library_df, query)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # top_k = 2
        assert list(result.columns) == ["index", "label", "score", "confidence", "metric"]
        assert all(result["metric"] == "cosine")
        assert all(result["label"].astype(str).isin(["0", "1", "2"]))

    def test_run_with_wavenumbers(self):
        """Test run method with wavenumbers parameter."""
        workflow = LibrarySearchWorkflow()
        library_df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]})
        query = np.array([1.5, 3.5])
        wavenumbers = np.array([800, 1200])

        result = workflow.run(library_df, query, wavenumbers=wavenumbers)

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5  # default top_k

    def test_run_respects_top_k(self):
        """Test that run respects top_k parameter."""
        for k in [1, 2, 5, 10]:
            workflow = LibrarySearchWorkflow(top_k=k)
            library_df = pd.DataFrame({f"s{i}": np.random.rand(20) for i in range(5)})
            query = np.random.rand(5)

            result = workflow.run(library_df, query)

            assert len(result) <= k

    def test_validate_valid_metric(self):
        """Test validation with valid metrics."""
        for metric in ["cosine", "sid", "sam"]:
            workflow = LibrarySearchWorkflow(metric=metric, top_k=5)
            validation = workflow.validate()

            assert validation["ok"] is True
            assert len(validation["issues"]) == 0

    def test_validate_invalid_metric(self):
        """Test validation with invalid metric."""
        workflow = LibrarySearchWorkflow(metric="invalid_metric")
        validation = workflow.validate()

        assert validation["ok"] is False
        assert len(validation["issues"]) == 1
        assert "unknown metric" in validation["issues"][0]

    def test_validate_invalid_top_k(self):
        """Test validation with invalid top_k."""
        for invalid_k in [0, -1, -5]:
            workflow = LibrarySearchWorkflow(top_k=invalid_k)
            validation = workflow.validate()

            assert validation["ok"] is False
            assert len(validation["issues"]) == 1
            assert "top_k must be > 0" in validation["issues"][0]

    def test_validate_multiple_issues(self):
        """Test validation with multiple issues."""
        workflow = LibrarySearchWorkflow(metric="bad_metric", top_k=-1)
        validation = workflow.validate()

        assert validation["ok"] is False
        assert len(validation["issues"]) == 2

    def test_to_dict(self):
        """Test to_dict method."""
        workflow = LibrarySearchWorkflow(metric="sam", top_k=7)
        config = workflow.to_dict()

        assert config["metric"] == "sam"
        assert config["top_k"] == 7
        assert isinstance(config["top_k"], int)

    def test_to_dict_preserves_values(self):
        """Test that to_dict preserves all values."""
        for metric, k in [("cosine", 3), ("sid", 10), ("sam", 1)]:
            workflow = LibrarySearchWorkflow(metric=metric, top_k=k)
            config = workflow.to_dict()

            assert config["metric"] == metric
            assert config["top_k"] == k

    def test_hash_consistency(self):
        """Test that __hash__ returns consistent values."""
        workflow1 = LibrarySearchWorkflow(metric="cosine", top_k=5)
        workflow2 = LibrarySearchWorkflow(metric="cosine", top_k=5)

        assert hash(workflow1) == hash(workflow2)

    def test_hash_difference(self):
        """Test that different workflows have different hashes."""
        workflow1 = LibrarySearchWorkflow(metric="cosine", top_k=5)
        workflow2 = LibrarySearchWorkflow(metric="sid", top_k=5)
        workflow3 = LibrarySearchWorkflow(metric="cosine", top_k=3)

        # Different metrics or top_k should (likely) produce different hashes
        assert hash(workflow1) != hash(workflow2) or hash(workflow1) != hash(workflow3)

    def test_hash_is_integer(self):
        """Test that __hash__ returns an integer."""
        workflow = LibrarySearchWorkflow()
        h = hash(workflow)

        assert isinstance(h, int)

    def test_run_single_library_entry(self):
        """Test run with single library entry."""
        workflow = LibrarySearchWorkflow(top_k=5)
        library_df = pd.DataFrame({"s1": [1.0], "s2": [2.0]})
        query = np.array([1.0, 2.0])

        result = workflow.run(library_df, query)

        assert len(result) == 1
        assert result.iloc[0]["index"] == 0

    def test_run_with_different_metrics(self):
        """Test run with different metrics."""
        library_df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]})
        query = np.array([1.5, 3.5])

        for metric in ["cosine", "sid", "sam"]:
            workflow = LibrarySearchWorkflow(metric=metric)
            result = workflow.run(library_df, query)

            assert all(result["metric"] == metric)

    def test_run_large_library(self):
        """Test run with large library."""
        workflow = LibrarySearchWorkflow(top_k=10)
        # Create library with 100 entries
        library_df = pd.DataFrame(np.random.rand(100, 20))
        query = np.random.rand(20)

        result = workflow.run(library_df, query)

        assert len(result) == 10
        assert result["index"].min() >= 0
        assert result["index"].max() < 100

    def test_run_output_indices_are_valid(self):
        """Test that output indices are valid library indices."""
        workflow = LibrarySearchWorkflow(top_k=3)
        library_df = pd.DataFrame(np.random.rand(15, 5))
        query = np.random.rand(5)

        result = workflow.run(library_df, query)

        # All indices should be valid
        assert all((result["index"] >= 0) & (result["index"] < 15))
        # Indices should be unique
        assert len(result["index"]) == len(result["index"].unique())
