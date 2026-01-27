"""Integration tests for preprocessing engine."""

import numpy as np
import pytest

from foodspec.preprocess import (
    PreprocessCache,
    PreprocessManifest,
    build_pipeline_from_recipe,
    compute_cache_key,
    compute_data_hash,
    compute_recipe_hash,
    load_preset_yaml,
    load_recipe,
)
from foodspec.preprocess.spectroscopy_operators import (
    DespikeOperator,
    FluorescenceRemovalOperator,
    MSCOperator,
)


class TestFullRamanPipeline:
    """Test complete Raman preprocessing pipeline."""

    def test_raman_preset_loads(self):
        """Test that raman preset loads correctly."""
        preset = load_preset_yaml("raman")
        assert preset["modality"] == "raman"
        assert "steps" in preset
        assert len(preset["steps"]) > 0

    def test_raman_pipeline_from_preset(self, synthetic_raman_data):
        """Test building pipeline from raman preset."""
        preset = load_preset_yaml("raman")
        pipeline = build_pipeline_from_recipe(preset)

        # Run pipeline
        result, metrics = pipeline.transform(synthetic_raman_data)

        assert result.x.shape == synthetic_raman_data.x.shape
        assert np.all(np.isfinite(result.x))

    def test_raman_operators_individually(self, synthetic_raman_data):
        """Test Raman-specific operators."""
        # Despike
        despike = DespikeOperator(window=5, threshold=5.0)
        despiked = despike.transform(synthetic_raman_data)
        assert despiked.x.shape == synthetic_raman_data.x.shape

        # Fluorescence removal
        fluor = FluorescenceRemovalOperator(method="poly", poly_order=2)
        corrected = fluor.transform(synthetic_raman_data)
        assert corrected.x.shape == synthetic_raman_data.x.shape


class TestFullFTIRPipeline:
    """Test complete FTIR preprocessing pipeline."""

    def test_ftir_preset_loads(self):
        """Test that ftir preset loads correctly."""
        preset = load_preset_yaml("ftir")
        assert preset["modality"] == "ftir"
        assert "steps" in preset

    def test_ftir_pipeline_from_preset(self, synthetic_ftir_data):
        """Test building pipeline from ftir preset."""
        preset = load_preset_yaml("ftir")
        pipeline = build_pipeline_from_recipe(preset)

        # Run pipeline
        result, metrics = pipeline.transform(synthetic_ftir_data)

        assert result.x.shape == synthetic_ftir_data.x.shape
        assert np.all(np.isfinite(result.x))

    def test_msc_operator(self, synthetic_ftir_data):
        """Test MSC operator."""
        msc = MSCOperator()
        msc.fit(synthetic_ftir_data)
        corrected = msc.transform(synthetic_ftir_data)
        assert corrected.x.shape == synthetic_ftir_data.x.shape


class TestRecipeLoading:
    """Test recipe loading and merging."""

    def test_load_recipe_with_preset(self):
        """Test loading recipe with preset."""
        pipeline = load_recipe(preset="default")
        assert len(pipeline.steps) > 0

    def test_load_recipe_with_overrides(self):
        """Test loading recipe with CLI overrides."""
        cli_overrides = {
            "override_steps": [
                {"op": "baseline", "lam": 1e6}  # Override lam parameter
            ]
        }
        pipeline = load_recipe(preset="default", cli_overrides=cli_overrides)

        # Find baseline step and check parameter
        baseline_step = next((s for s in pipeline.steps if s.name == "baseline"), None)
        assert baseline_step is not None
        assert baseline_step.config.get("lam") == 1e6

    def test_protocol_integration(self):
        """Test loading recipe from protocol config."""
        protocol_config = {
            "preprocess": {
                "modality": "raman",
                "steps": [
                    {"op": "baseline", "method": "als"},
                    {"op": "normalization", "method": "snv"},
                ],
            }
        }

        from foodspec.preprocess.loaders import load_recipe
        pipeline = load_recipe(protocol_config=protocol_config)
        assert len(pipeline.steps) == 2


class TestCachingSystem:
    """Test caching and provenance."""

    def test_data_hash_deterministic(self, synthetic_raman_data):
        """Test that data hash is deterministic."""
        hash1 = compute_data_hash(synthetic_raman_data.x, synthetic_raman_data.wavenumbers)
        hash2 = compute_data_hash(synthetic_raman_data.x, synthetic_raman_data.wavenumbers)
        assert hash1 == hash2

    def test_recipe_hash_deterministic(self, sample_recipe_dict):
        """Test that recipe hash is deterministic."""
        hash1 = compute_recipe_hash(sample_recipe_dict)
        hash2 = compute_recipe_hash(sample_recipe_dict)
        assert hash1 == hash2

    def test_cache_key_generation(self):
        """Test cache key generation."""
        data_hash = "abc123"
        recipe_hash = "def456"
        cache_key = compute_cache_key(data_hash, recipe_hash, seed=42)
        assert isinstance(cache_key, str)
        assert len(cache_key) == 16

    def test_cache_put_get(self, temp_cache_dir, synthetic_raman_data):
        """Test caching put/get operations."""
        cache = PreprocessCache(temp_cache_dir)

        # Put data
        cache_key = "test_key_123"
        cache.put(cache_key, synthetic_raman_data.x, synthetic_raman_data.wavenumbers)

        # Get data
        result = cache.get(cache_key)
        assert result is not None
        np.testing.assert_array_equal(result["X"], synthetic_raman_data.x)

    def test_cache_miss(self, temp_cache_dir):
        """Test cache miss returns None."""
        cache = PreprocessCache(temp_cache_dir)
        result = cache.get("nonexistent_key")
        assert result is None


class TestManifestGeneration:
    """Test manifest generation."""

    def test_manifest_creation(self, sample_recipe_dict):
        """Test manifest creation and finalization."""
        manifest = PreprocessManifest(
            run_id="test_run_001",
            recipe=sample_recipe_dict,
            cache_key="abc123",
            seed=42,
        )

        # Record operators
        manifest.record_operator("despike", 12.3, spikes_removed=5)
        manifest.record_operator("baseline", 45.6)

        # Add warning
        manifest.add_warning("Sample 10 had NaN values")

        # Finalize
        manifest.finalize(
            n_samples_input=100,
            n_samples_output=98,
            n_features=512,
            rejected_spectra=2,
        )

        manifest_dict = manifest.to_dict()
        assert manifest_dict["run_id"] == "test_run_001"
        assert len(manifest_dict["operators_applied"]) == 2
        assert len(manifest_dict["warnings"]) == 1
        assert manifest_dict["statistics"]["n_samples_input"] == 100

    def test_manifest_save_load(self, temp_cache_dir, sample_recipe_dict):
        """Test manifest save/load."""
        manifest = PreprocessManifest(
            run_id="test_run_002",
            recipe=sample_recipe_dict,
            cache_key="xyz789",
        )
        manifest.finalize(50, 50, 256)

        # Save
        manifest_path = temp_cache_dir / "manifest.json"
        manifest.save(manifest_path)

        # Load
        loaded = PreprocessManifest.load(manifest_path)
        assert loaded.run_id == "test_run_002"
        assert loaded.cache_key == "xyz789"


class TestReproducibility:
    """Test deterministic behavior."""

    def test_same_seed_same_output(self, synthetic_raman_data):
        """Test that same seed produces same output."""
        np.random.seed(42)
        pipeline1 = load_recipe(preset="raman")
        result1, _ = pipeline1.transform(synthetic_raman_data)

        np.random.seed(42)
        pipeline2 = load_recipe(preset="raman")
        result2, _ = pipeline2.transform(synthetic_raman_data)

        np.testing.assert_array_almost_equal(result1.x, result2.x)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unknown_preset_raises_error(self):
        """Test that unknown preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset_yaml("nonexistent_preset")

    def test_food_specific_presets_load(self):
        """Test that food-specific presets load correctly."""
        for name in ["dairy", "meat", "fruit", "grain"]:
            preset = load_preset_yaml(name)
            assert "steps" in preset
            assert len(preset["steps"]) > 0

    def test_unknown_operator_skips_gracefully(self, synthetic_raman_data):
        """Test that unknown operator is skipped."""
        recipe = {
            "modality": "raman",
            "steps": [
                {"op": "unknown_op", "param": 123},
                {"op": "baseline", "method": "als"},
            ],
        }
        pipeline = build_pipeline_from_recipe(recipe)
        # Should only have baseline step (unknown_op skipped)
        assert len(pipeline.steps) == 1

    def test_empty_recipe(self, synthetic_raman_data):
        """Test empty recipe returns data unchanged."""
        recipe = {"modality": "raman", "steps": []}
        pipeline = build_pipeline_from_recipe(recipe)
        result, _ = pipeline.transform(synthetic_raman_data)
        np.testing.assert_array_equal(result.x, synthetic_raman_data.x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
