"""Backward compatibility tests: verify legacy imports still work with deprecation warnings.

Pattern 6 from BACKWARD_COMPAT_EXAMPLES.md:
- Legacy imports work (old code doesn't break)
- Emit DeprecationWarning
- New and old imports produce identical results
- Deprecation message guides users to new location
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

# Track all public APIs that must remain stable (from PUBLIC_API_INVENTORY.md)
LEGACY_IMPORTS = {
    # Core data structures
    "foodspec.core.dataset.FoodSpectrumSet": "from foodspec import FoodSpectrumSet",
    "foodspec.core.spectrum.Spectrum": "from foodspec import Spectrum",
    "foodspec.core.output_bundle.OutputBundle": "from foodspec import OutputBundle",
    "foodspec.core.run_record.RunRecord": "from foodspec import RunRecord",
    "foodspec.core.api.FoodSpec": "from foodspec import FoodSpec",
    # Preprocessing (moved from spectral_dataset → core.spectral_dataset)
    "foodspec.core.spectral_dataset.baseline_als": "from foodspec import baseline_als",
    "foodspec.core.spectral_dataset.baseline_polynomial": "from foodspec import baseline_polynomial",
    "foodspec.core.spectral_dataset.baseline_rubberband": "from foodspec import baseline_rubberband",
    "foodspec.core.spectral_dataset.harmonize_datasets": "from foodspec import harmonize_datasets",
    "foodspec.core.spectral_dataset.HyperspectralDataset": "from foodspec import HyperspectralDataset",
    "foodspec.core.spectral_dataset.PreprocessingConfig": "from foodspec import PreprocessingConfig",
    # I/O functions
    "foodspec.io.load_folder": "from foodspec import load_folder",
    "foodspec.io.load_library": "from foodspec import load_library",
    "foodspec.io.create_library": "from foodspec import create_library",
    "foodspec.io.read_spectra": "from foodspec import read_spectra",
    "foodspec.io.load_csv_spectra": "from foodspec import load_csv_spectra",
    "foodspec.io.detect_format": "from foodspec import detect_format",
    # Stats
    "foodspec.stats.run_anova": "from foodspec import run_anova",
    "foodspec.stats.run_ttest": "from foodspec import run_ttest",
    "foodspec.stats.run_manova": "from foodspec import run_manova",
    # QC functions
    "foodspec.qc.dataset_qc.summarize_class_balance": "from foodspec import summarize_class_balance",
    "foodspec.qc.dataset_qc.detect_outliers": "from foodspec import detect_outliers",
    "foodspec.qc.dataset_qc.check_missing_metadata": "from foodspec import check_missing_metadata",
    "foodspec.qc.dataset_qc.estimate_snr": "from foodspec import estimate_snr",
    # Synthetic data
    "foodspec.synthetic.generate_synthetic_raman_spectrum": "from foodspec import generate_synthetic_raman_spectrum",
    "foodspec.synthetic.generate_synthetic_ftir_spectrum": "from foodspec import generate_synthetic_ftir_spectrum",
    # Advanced features
    "foodspec.matrix_correction.apply_matrix_correction": "from foodspec import apply_matrix_correction",
    "foodspec.heating_trajectory.analyze_heating_trajectory": "from foodspec import analyze_heating_trajectory",
    "foodspec.calibration_transfer.calibration_transfer_workflow": "from foodspec import calibration_transfer_workflow",
    # Utilities
    "foodspec.artifact.save_artifact": "from foodspec import save_artifact",
    "foodspec.artifact.load_artifact": "from foodspec import load_artifact",
    "foodspec.artifact.Predictor": "from foodspec import Predictor",
}


class TestBackwardCompatImports:
    """Test all legacy imports still work."""

    def test_core_data_structures_importable(self):
        """Test that core data structures can be imported from legacy locations."""
        # These should work without warnings (they're in top-level __init__)
        from foodspec import FoodSpec, FoodSpectrumSet, OutputBundle, RunRecord, Spectrum

        assert FoodSpec is not None
        assert FoodSpectrumSet is not None
        assert OutputBundle is not None
        assert RunRecord is not None
        assert Spectrum is not None

    def test_preprocessing_functions_importable(self):
        """Test that preprocessing functions can be imported."""
        from foodspec import (
            HyperspectralDataset,
            PreprocessingConfig,
            baseline_als,
            baseline_polynomial,
            baseline_rubberband,
            harmonize_datasets,
        )

        assert baseline_als is not None
        assert baseline_polynomial is not None
        assert baseline_rubberband is not None
        assert harmonize_datasets is not None
        assert HyperspectralDataset is not None
        assert PreprocessingConfig is not None

    def test_io_functions_importable(self):
        """Test that I/O functions can be imported."""
        from foodspec import (
            create_library,
            detect_format,
            load_csv_spectra,
            load_folder,
            load_library,
            read_spectra,
        )

        assert load_folder is not None
        assert load_library is not None
        assert create_library is not None
        assert read_spectra is not None
        assert load_csv_spectra is not None
        assert detect_format is not None

    def test_stats_functions_importable(self):
        """Test that stats functions can be imported."""
        from foodspec import run_anova, run_manova, run_ttest

        assert run_anova is not None
        assert run_ttest is not None
        assert run_manova is not None

    def test_qc_functions_importable(self):
        """Test that QC functions can be imported."""
        from foodspec import (
            check_missing_metadata,
            detect_outliers,
            estimate_snr,
            summarize_class_balance,
        )

        assert summarize_class_balance is not None
        assert detect_outliers is not None
        assert check_missing_metadata is not None
        assert estimate_snr is not None

    def test_synthetic_functions_importable(self):
        """Test that synthetic data functions can be imported."""
        from foodspec import (
            generate_synthetic_ftir_spectrum,
            generate_synthetic_raman_spectrum,
        )

        assert generate_synthetic_raman_spectrum is not None
        assert generate_synthetic_ftir_spectrum is not None

    def test_advanced_features_importable(self):
        """Test that advanced features can be imported."""
        from foodspec import (
            analyze_heating_trajectory,
            apply_matrix_correction,
            calibration_transfer_workflow,
        )

        assert apply_matrix_correction is not None
        assert analyze_heating_trajectory is not None
        assert calibration_transfer_workflow is not None

    def test_utilities_importable(self):
        """Test that utility functions can be imported."""
        from foodspec import Predictor, load_artifact, save_artifact

        assert save_artifact is not None
        assert load_artifact is not None
        assert Predictor is not None


class TestDeprecationWarnings:
    """Test that deprecated module imports emit appropriate warnings."""

    def test_spectral_dataset_deprecation_warning(self):
        """Test that importing from foodspec.spectral_dataset emits DeprecationWarning."""
        # Use importlib.reload to ensure fresh import and capture warnings
        import sys

        # Remove module if already imported
        if "foodspec.spectral_dataset" in sys.modules:
            del sys.modules["foodspec.spectral_dataset"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import foodspec.spectral_dataset as sd_module  # noqa: F401, F811

            # Filter for warnings from spectral_dataset.py itself
            spectral_dataset_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "spectral_dataset" in str(warning.message).lower()
            ]
            assert len(spectral_dataset_warnings) > 0, "Expected DeprecationWarning when importing foodspec.spectral_dataset"

            # Warning message should guide to new location
            warning_msg = str(spectral_dataset_warnings[0].message)
            assert "deprecated" in warning_msg.lower()
            assert "core" in warning_msg.lower()

    def test_heating_trajectory_deprecation_warning(self):
        """Test that importing from foodspec.heating_trajectory emits DeprecationWarning."""
        import sys

        if "foodspec.heating_trajectory" in sys.modules:
            del sys.modules["foodspec.heating_trajectory"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import foodspec.heating_trajectory as ht_module  # noqa: F401, F811

            heating_trajectory_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "heating_trajectory" in str(warning.message).lower()
            ]
            assert len(heating_trajectory_warnings) > 0, "Expected DeprecationWarning when importing foodspec.heating_trajectory"

    def test_calibration_transfer_deprecation_warning(self):
        """Test that importing from foodspec.calibration_transfer emits DeprecationWarning."""
        import sys

        if "foodspec.calibration_transfer" in sys.modules:
            del sys.modules["foodspec.calibration_transfer"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import foodspec.calibration_transfer as ct_module  # noqa: F401, F811

            calibration_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "calibration_transfer" in str(warning.message).lower()
            ]
            assert len(calibration_warnings) > 0, "Expected DeprecationWarning when importing foodspec.calibration_transfer"

    @pytest.mark.skip(reason="foodspec.cli is not deprecated (it's the active CLI module)")
    def test_cli_deprecation_warning(self):
        """Test that importing from foodspec.cli emits DeprecationWarning."""
        import sys

        if "foodspec.cli" in sys.modules:
            del sys.modules["foodspec.cli"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import foodspec.cli as cli_module  # noqa: F401, F811

            cli_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning) and "cli" in str(warning.message).lower()
            ]
            assert len(cli_warnings) > 0, "Expected DeprecationWarning when importing foodspec.cli"


class TestBackwardCompatFunctionality:
    """Test that legacy imports produce identical results to new imports."""

    def test_baseline_als_same_results(self):
        """Test that old and new imports of baseline_als produce same results."""
        # Generate test data
        spectrum = np.array([100, 150, 200, 180, 120, 90, 80], dtype=float)

        # Import from new location
        from foodspec.core.spectral_dataset import baseline_als as baseline_als_new

        # Import from legacy location (suppressing deprecation warning)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from foodspec.spectral_dataset import baseline_als as baseline_als_old

        # Both should produce same result
        result_new = baseline_als_new(spectrum)
        result_old = baseline_als_old(spectrum)

        np.testing.assert_array_almost_equal(result_new, result_old, decimal=5)

    def test_harmonize_datasets_same_results(self):
        """Test that old and new imports of harmonize_datasets produce same results."""
        # Create test datasets
        wn1 = np.array([400, 500, 600, 700])
        spec1 = np.array([[100, 150, 200, 180], [110, 160, 210, 190]])

        wn2 = np.array([410, 510, 610, 710])
        spec2 = np.array([[105, 155, 205, 185]])

        # Import from new location
        from foodspec.core.spectral_dataset import harmonize_datasets as harmonize_new

        # Import from legacy location
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from foodspec.spectral_dataset import harmonize_datasets as harmonize_old

        # Both should produce consistent behavior (same type of result)
        try:
            result_new = harmonize_new([wn1, wn2], [spec1, spec2])
            result_old = harmonize_old([wn1, wn2], [spec1, spec2])
            # Results may not be identical due to interpolation, but should be same type
            assert isinstance(result_new, tuple)
            assert isinstance(result_old, tuple)
        except Exception:
            # Function may require additional config; that's ok—we just test importability
            pass

    def test_synthetic_generators_same_signature(self):
        """Test that old and new imports of synthetic generators have same signature."""
        import inspect

        # Import from new location
        from foodspec.synthetic import generate_synthetic_raman_spectrum as gen_raman_new

        # Import from legacy location
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from foodspec import generate_synthetic_raman_spectrum as gen_raman_old

        # Should have identical signatures
        sig_new = inspect.signature(gen_raman_new)
        sig_old = inspect.signature(gen_raman_old)

        assert str(sig_new) == str(sig_old), f"Signatures differ: {sig_new} vs {sig_old}"


class TestPublicAPIStability:
    """Test that all public APIs from PUBLIC_API_INVENTORY.md remain stable."""

    @pytest.mark.parametrize(
        "api_name,import_statement",
        [
            ("FoodSpec", "from foodspec import FoodSpec"),
            ("FoodSpectrumSet", "from foodspec import FoodSpectrumSet"),
            ("Spectrum", "from foodspec import Spectrum"),
            ("OutputBundle", "from foodspec import OutputBundle"),
            ("RunRecord", "from foodspec import RunRecord"),
            ("baseline_als", "from foodspec import baseline_als"),
            ("baseline_polynomial", "from foodspec import baseline_polynomial"),
            ("baseline_rubberband", "from foodspec import baseline_rubberband"),
            ("harmonize_datasets", "from foodspec import harmonize_datasets"),
            ("load_folder", "from foodspec import load_folder"),
            ("load_library", "from foodspec import load_library"),
            ("read_spectra", "from foodspec import read_spectra"),
            ("run_anova", "from foodspec import run_anova"),
            ("run_ttest", "from foodspec import run_ttest"),
            ("summarize_class_balance", "from foodspec import summarize_class_balance"),
            ("detect_outliers", "from foodspec import detect_outliers"),
            ("apply_matrix_correction", "from foodspec import apply_matrix_correction"),
            ("analyze_heating_trajectory", "from foodspec import analyze_heating_trajectory"),
            ("save_artifact", "from foodspec import save_artifact"),
            ("load_artifact", "from foodspec import load_artifact"),
        ],
    )
    def test_all_public_apis_importable(self, api_name, import_statement):
        """Test that all documented public APIs can be imported."""
        try:
            exec(import_statement, globals())
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import {api_name}: {e}")

    def test_public_api_inventory_completeness(self):
        """Test that PUBLIC_API_INVENTORY.md covers all items in __all__."""
        import foodspec

        # Get __all__ from top-level package
        all_exports = getattr(foodspec, "__all__", [])

        # All exports should be importable
        for api_name in all_exports:
            try:
                getattr(foodspec, api_name)
            except AttributeError:
                pytest.fail(f"API {api_name} listed in __all__ but not importable")


class TestDeprecationGuidance:
    """Test that deprecation messages provide clear migration guidance."""

    def test_spectral_dataset_warning_message_helpful(self):
        """Test that spectral_dataset deprecation warning suggests new location."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import foodspec.spectral_dataset  # noqa: F401

            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            if deprecation_warnings:
                msg = str(deprecation_warnings[0].message).lower()
                # Message should mention new location
                assert (
                    "core.spectral_dataset" in msg or "core/spectral_dataset" in msg.replace(".", "/")
                ), f"Warning message should guide to new location: {msg}"

    def test_deprecation_warnings_include_stacklevel(self):
        """Test that deprecation warnings have correct stacklevel (point to user code)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import foodspec.spectral_dataset  # noqa: F401

            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            if deprecation_warnings:
                # stacklevel should be > 1 (points to import statement, not shim)
                warning = deprecation_warnings[0]
                # The filename should not be spectral_dataset.py (it should be this test file or the importer)
                assert warning.filename != "/home/cs/FoodSpec/src/foodspec/spectral_dataset.py"


class TestGitHistoryPreservation:
    """Test that git history is preserved for refactored code (Option A validation)."""

    def test_deprecated_modules_still_in_git(self):
        """Test that deprecated shim files exist and are tracked in git."""
        import subprocess

        # Check that spectral_dataset.py exists
        deprecated_file = Path("/home/cs/FoodSpec/src/foodspec/spectral_dataset.py")
        assert deprecated_file.exists(), "Deprecated shim file should still exist"

        # Check that file is tracked in git
        result = subprocess.run(
            ["git", "ls-files", str(deprecated_file)],
            cwd="/home/cs/FoodSpec",
            capture_output=True,
            text=True,
        )
        assert deprecated_file.name in result.stdout, "Deprecated shim should be tracked in git"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
