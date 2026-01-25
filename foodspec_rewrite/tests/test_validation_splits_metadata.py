"""
Unit tests for metadata-aware validation splitters.

Tests cover:
- LeaveOneGroupOutSplitter: deterministic ordering, fold_info structure, metadata validation
- LeaveOneBatchOutSplitter & LeaveOneStageOutSplitter: convenience wrappers
- Error handling: missing metadata keys, mismatched array lengths
- Fold stability: consistent results across runs
"""

import numpy as np
import pandas as pd
import pytest

from foodspec.validation.splits import (
    LeaveOneGroupOutSplitter,
    LeaveOneBatchOutSplitter,
    LeaveOneStageOutSplitter,
    StratifiedKFoldOrGroupKFold,
)


@pytest.fixture
def sample_data():
    """Create sample data with metadata."""
    n_samples = 12
    X = np.random.randn(n_samples, 5)
    y = np.array([0, 1] * (n_samples // 2))
    
    meta = pd.DataFrame({
        "sample_id": [f"s{i:03d}" for i in range(n_samples)],
        "group": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "batch": ["A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "B", "B"],
        "stage": ["discovery"] * 6 + ["validation"] * 6,
    })
    
    return X, y, meta


class TestLeaveOneGroupOutSplitter:
    """Test LeaveOneGroupOutSplitter with metadata."""

    def test_basic_splitting(self, sample_data):
        """Test basic LOGO splitting with metadata."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        # Should have 4 folds (4 unique groups)
        assert len(folds) == 4
        
        # Each fold should have 3 elements (train_idx, test_idx, fold_info)
        for fold in folds:
            assert len(fold) == 3
            train_idx, test_idx, fold_info = fold
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert isinstance(fold_info, dict)

    def test_fold_info_structure(self, sample_data):
        """Test fold_info contains required keys."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        for train_idx, test_idx, fold_info in folds:
            assert "fold_id" in fold_info
            assert "held_out_group" in fold_info
            assert "n_train" in fold_info
            assert "n_test" in fold_info
            assert fold_info["n_train"] + fold_info["n_test"] == len(meta)

    def test_deterministic_ordering(self, sample_data):
        """Test folds are ordered deterministically by group value."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        held_out_groups = [fold[2]["held_out_group"] for fold in folds]
        
        # Should be in sorted order: 1, 2, 3, 4
        assert held_out_groups == [1, 2, 3, 4]

    def test_correct_held_out_group(self, sample_data):
        """Test that correct group is held out in each fold."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        for train_idx, test_idx, fold_info in folds:
            held_out_group = fold_info["held_out_group"]
            
            # Test set should contain only held_out_group
            test_groups = meta.iloc[test_idx]["group"].unique()
            assert len(test_groups) == 1
            assert test_groups[0] == held_out_group
            
            # Train set should not contain held_out_group
            train_groups = meta.iloc[train_idx]["group"].unique()
            assert held_out_group not in train_groups

    def test_no_leakage(self, sample_data):
        """Test there's no overlap between train and test indices."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        for train_idx, test_idx, _ in folds:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, "Train and test sets should not overlap"

    def test_complete_coverage(self, sample_data):
        """Test all samples are covered across folds."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        all_indices = set()
        for train_idx, test_idx, _ in folds:
            all_indices.update(train_idx)
            all_indices.update(test_idx)
        
        assert len(all_indices) == len(meta), "All samples should be covered"

    def test_fold_sizes(self, sample_data):
        """Test fold sizes match metadata structure."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        # Each group has 3 samples
        for train_idx, test_idx, fold_info in folds:
            assert fold_info["n_test"] == 3, "Each held-out group has 3 samples"
            assert fold_info["n_train"] == 9, "Remaining samples are 9"

    def test_missing_group_key(self, sample_data):
        """Test error when group_key not in metadata."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="nonexistent_key")
        
        with pytest.raises(ValueError, match="not found in metadata"):
            list(splitter.split(X, y, meta))

    def test_error_message_lists_available_keys(self, sample_data):
        """Test error message lists available metadata keys."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="bad_key")
        
        with pytest.raises(ValueError) as exc_info:
            list(splitter.split(X, y, meta))
        
        error_msg = str(exc_info.value)
        assert "Available keys:" in error_msg
        assert "group" in error_msg or "batch" in error_msg

    def test_mismatched_lengths(self, sample_data):
        """Test error when X, y, meta have different lengths."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        
        # X too short
        with pytest.raises(ValueError):
            list(splitter.split(X[:-1], y, meta))
        
        # y too short
        with pytest.raises(ValueError):
            list(splitter.split(X, y[:-1], meta))

    def test_string_groups(self):
        """Test splitting with string group identifiers."""
        X = np.random.randn(9, 3)
        y = np.array([0, 1, 0] * 3)
        meta = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        })
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        assert len(folds) == 3
        held_out_groups = [fold[2]["held_out_group"] for fold in folds]
        assert held_out_groups == ["A", "B", "C"]

    def test_numeric_groups(self):
        """Test splitting with numeric group identifiers."""
        X = np.random.randn(9, 3)
        y = np.array([0, 1, 0] * 3)
        meta = pd.DataFrame({
            "group": [10, 10, 10, 20, 20, 20, 30, 30, 30],
        })
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        assert len(folds) == 3
        held_out_groups = [fold[2]["held_out_group"] for fold in folds]
        assert held_out_groups == [10, 20, 30]

    def test_fold_id_sequence(self, sample_data):
        """Test fold_id is sequential starting from 0."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        folds = list(splitter.split(X, y, meta))
        
        fold_ids = [fold[2]["fold_id"] for fold in folds]
        assert fold_ids == list(range(len(folds)))

    def test_reproducibility(self, sample_data):
        """Test results are reproducible across multiple runs."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        
        folds1 = list(splitter.split(X, y, meta))
        folds2 = list(splitter.split(X, y, meta))
        
        assert len(folds1) == len(folds2)
        for (train1, test1, info1), (train2, test2, info2) in zip(folds1, folds2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)
            assert info1 == info2


class TestLeaveOneBatchOutSplitter:
    """Test LeaveOneBatchOutSplitter convenience wrapper."""

    def test_batch_splitting(self, sample_data):
        """Test batch splitting uses 'batch' key."""
        X, y, meta = sample_data
        
        splitter = LeaveOneBatchOutSplitter()
        folds = list(splitter.split(X, y, meta))
        
        # 2 unique batches: A, B
        assert len(folds) == 2
        
        held_out_batches = [fold[2]["held_out_group"] for fold in folds]
        assert held_out_batches == ["A", "B"]

    def test_batch_key_missing_error(self):
        """Test error when 'batch' key missing."""
        X = np.random.randn(6, 2)
        y = np.array([0, 1, 0, 1, 0, 1])
        meta = pd.DataFrame({"group": [1, 1, 2, 2, 3, 3]})
        
        splitter = LeaveOneBatchOutSplitter()
        
        with pytest.raises(ValueError, match="batch"):
            list(splitter.split(X, y, meta))

    def test_equivalent_to_group_splitter(self, sample_data):
        """Test LeaveOneBatchOutSplitter is equivalent to using group_key='batch'."""
        X, y, meta = sample_data
        
        batch_splitter = LeaveOneBatchOutSplitter()
        group_splitter = LeaveOneGroupOutSplitter(group_key="batch")
        
        batch_folds = list(batch_splitter.split(X, y, meta))
        group_folds = list(group_splitter.split(X, y, meta))
        
        assert len(batch_folds) == len(group_folds)
        for (train1, test1, info1), (train2, test2, info2) in zip(batch_folds, group_folds):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)
            assert info1 == info2


class TestLeaveOneStageOutSplitter:
    """Test LeaveOneStageOutSplitter convenience wrapper."""

    def test_stage_splitting(self, sample_data):
        """Test stage splitting uses 'stage' key."""
        X, y, meta = sample_data
        
        splitter = LeaveOneStageOutSplitter()
        folds = list(splitter.split(X, y, meta))
        
        # 2 unique stages: discovery, validation
        assert len(folds) == 2
        
        held_out_stages = [fold[2]["held_out_group"] for fold in folds]
        assert held_out_stages == ["discovery", "validation"]

    def test_stage_key_missing_error(self):
        """Test error when 'stage' key missing."""
        X = np.random.randn(6, 2)
        y = np.array([0, 1, 0, 1, 0, 1])
        meta = pd.DataFrame({"group": [1, 1, 2, 2, 3, 3]})
        
        splitter = LeaveOneStageOutSplitter()
        
        with pytest.raises(ValueError, match="stage"):
            list(splitter.split(X, y, meta))

    def test_equivalent_to_group_splitter(self, sample_data):
        """Test LeaveOneStageOutSplitter is equivalent to using group_key='stage'."""
        X, y, meta = sample_data
        
        stage_splitter = LeaveOneStageOutSplitter()
        group_splitter = LeaveOneGroupOutSplitter(group_key="stage")
        
        stage_folds = list(stage_splitter.split(X, y, meta))
        group_folds = list(group_splitter.split(X, y, meta))
        
        assert len(stage_folds) == len(group_folds)
        for (train1, test1, info1), (train2, test2, info2) in zip(stage_folds, group_folds):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)
            assert info1 == info2


class TestSplitterIntegration:
    """Integration tests with evaluation workflows."""

    def test_splitter_with_cross_validation(self, sample_data):
        """Test splitter works in CV loop."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="group")
        
        fold_results = []
        for train_idx, test_idx, fold_info in splitter.split(X, y, meta):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Simulate training and evaluation
            fold_result = {
                "fold_id": fold_info["fold_id"],
                "train_size": len(X_train),
                "test_size": len(X_test),
                "held_out_group": fold_info["held_out_group"],
            }
            fold_results.append(fold_result)
        
        assert len(fold_results) == 4
        total_test_samples = sum(r["test_size"] for r in fold_results)
        assert total_test_samples == len(meta)

    def test_splitter_with_feature_extraction(self, sample_data):
        """Test splitter with feature extraction (no leakage)."""
        X, y, meta = sample_data
        
        splitter = LeaveOneGroupOutSplitter(group_key="batch")
        
        for train_idx, test_idx, fold_info in splitter.split(X, y, meta):
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Fit scaler on train only (no leakage)
            mean_train = np.mean(X_train, axis=0)
            std_train = np.std(X_train, axis=0) + 1e-8
            
            # Apply to test
            X_train_scaled = (X_train - mean_train) / std_train
            X_test_scaled = (X_test - mean_train) / std_train
            
            assert X_train_scaled.shape == X_train.shape
            assert X_test_scaled.shape == X_test.shape


__all__ = [
    "TestLeaveOneGroupOutSplitter",
    "TestLeaveOneBatchOutSplitter",
    "TestLeaveOneStageOutSplitter",
    "TestSplitterIntegration",
]
