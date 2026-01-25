"""Tests for paper-ready figure presets."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from foodspec.viz.paper import (
    FigurePreset,
    apply_figure_preset,
    figure_context,
    get_figure_preset_config,
    list_presets,
    save_figure,
)


class TestFigurePresetEnum:
    """Test FigurePreset enum."""

    def test_preset_enum_values(self) -> None:
        """Test that all presets have correct string values."""
        assert FigurePreset.JOSS.value == "joss"
        assert FigurePreset.IEEE.value == "ieee"
        assert FigurePreset.ELSEVIER.value == "elsevier"
        assert FigurePreset.NATURE.value == "nature"

    def test_preset_enum_count(self) -> None:
        """Test that there are exactly 4 presets."""
        assert len(FigurePreset) == 4


class TestApplyFigurePreset:
    """Test apply_figure_preset() function."""

    def setup_method(self) -> None:
        """Save original rcParams before each test."""
        self.original_params = copy.deepcopy(plt.rcParams)

    def teardown_method(self) -> None:
        """Restore original rcParams after each test."""
        plt.rcParams.update(self.original_params)

    def test_apply_preset_joss_enum(self) -> None:
        """Test applying JOSS preset via enum."""
        apply_figure_preset(FigurePreset.JOSS)
        assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.8])
        assert plt.rcParams["font.size"] == 9
        assert plt.rcParams["lines.linewidth"] == 1.5

    def test_apply_preset_joss_string(self) -> None:
        """Test applying JOSS preset via string."""
        apply_figure_preset("joss")
        assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.8])

    def test_apply_preset_ieee_enum(self) -> None:
        """Test applying IEEE preset via enum."""
        apply_figure_preset(FigurePreset.IEEE)
        assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.5])
        assert plt.rcParams["font.size"] == 9
        assert plt.rcParams["axes.linewidth"] == 1.0

    def test_apply_preset_ieee_string(self) -> None:
        """Test applying IEEE preset via string."""
        apply_figure_preset("ieee")
        assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.5])

    def test_apply_preset_elsevier(self) -> None:
        """Test applying Elsevier preset."""
        apply_figure_preset(FigurePreset.ELSEVIER)
        assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.6])
        assert plt.rcParams["font.size"] == 10
        assert plt.rcParams["axes.labelsize"] == 11

    def test_apply_preset_nature(self) -> None:
        """Test applying Nature preset."""
        apply_figure_preset(FigurePreset.NATURE)
        assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.8])
        assert plt.rcParams["font.size"] == 8
        assert plt.rcParams["lines.linewidth"] == 1.0

    def test_apply_preset_invalid_string(self) -> None:
        """Test that invalid preset string raises error."""
        with pytest.raises(ValueError, match="Invalid preset"):
            apply_figure_preset("invalid_preset")

    def test_apply_preset_case_insensitive(self) -> None:
        """Test that preset strings are case-insensitive."""
        apply_figure_preset("JOSS")
        assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.8])

        apply_figure_preset("IeEe")
        assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.5])

    def test_apply_preset_modifies_rcparams(self) -> None:
        """Test that applying preset modifies multiple rcParams."""
        original_figsize = plt.rcParams["figure.figsize"]
        original_fontsize = plt.rcParams["font.size"]

        apply_figure_preset(FigurePreset.NATURE)

        assert plt.rcParams["figure.figsize"] != original_figsize
        assert plt.rcParams["font.size"] != original_fontsize

    def test_apply_preset_sets_spines(self) -> None:
        """Test that presets set spine visibility correctly."""
        apply_figure_preset(FigurePreset.JOSS)
        assert plt.rcParams["axes.spines.left"] is True
        assert plt.rcParams["axes.spines.bottom"] is True
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False


class TestGetFigurePresetConfig:
    """Test get_figure_preset_config() function."""

    def test_get_config_returns_dict(self) -> None:
        """Test that function returns a dict."""
        config = get_figure_preset_config(FigurePreset.JOSS)
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_get_config_via_enum(self) -> None:
        """Test getting config via enum."""
        config = get_figure_preset_config(FigurePreset.IEEE)
        assert config["figure.figsize"] == (3.5, 2.5)

    def test_get_config_via_string(self) -> None:
        """Test getting config via string."""
        config = get_figure_preset_config("nature")
        assert config["figure.figsize"] == (3.5, 2.8)
        assert config["font.size"] == 8

    def test_get_config_does_not_modify_rcparams(self) -> None:
        """Test that getting config doesn't modify global rcParams."""
        original = copy.deepcopy(plt.rcParams)
        _ = get_figure_preset_config(FigurePreset.JOSS)
        assert plt.rcParams == original

    def test_get_config_returns_copy(self) -> None:
        """Test that config is a copy, not a reference."""
        config = get_figure_preset_config(FigurePreset.JOSS)
        config["figure.figsize"] = (999, 999)
        config2 = get_figure_preset_config(FigurePreset.JOSS)
        assert config2["figure.figsize"] != (999, 999)


class TestFigureContext:
    """Test figure_context() context manager."""

    def setup_method(self) -> None:
        """Save original rcParams before each test."""
        self.original_params = copy.deepcopy(plt.rcParams)

    def teardown_method(self) -> None:
        """Restore original rcParams after each test."""
        plt.rcParams.update(self.original_params)

    def test_context_applies_preset(self) -> None:
        """Test that context manager applies preset inside block."""
        original_figsize = plt.rcParams["figure.figsize"]

        with figure_context(FigurePreset.JOSS):
            assert plt.rcParams["figure.figsize"] == pytest.approx([3.5, 2.8])
            assert plt.rcParams["figure.figsize"] != pytest.approx(original_figsize)

    def test_context_restores_rcparams(self) -> None:
        """Test that rcParams are restored after exiting context."""
        original_params = copy.deepcopy(plt.rcParams)

        with figure_context(FigurePreset.IEEE):
            pass

        assert plt.rcParams == original_params

    def test_context_with_enum(self) -> None:
        """Test context manager with enum preset."""
        with figure_context(FigurePreset.NATURE):
            assert plt.rcParams["font.size"] == 8

    def test_context_with_string(self) -> None:
        """Test context manager with string preset."""
        with figure_context("elsevier"):
            assert plt.rcParams["font.size"] == 10

    def test_context_nested(self) -> None:
        """Test nested context managers."""
        original_params = copy.deepcopy(plt.rcParams)

        with figure_context(FigurePreset.JOSS):
            assert plt.rcParams["font.size"] == 9

            with figure_context(FigurePreset.NATURE):
                assert plt.rcParams["font.size"] == 8

            # Back to JOSS
            assert plt.rcParams["font.size"] == 9

        # Back to original
        assert plt.rcParams == original_params

    def test_context_exception_still_restores(self) -> None:
        """Test that rcParams are restored even if exception occurs."""
        original_params = copy.deepcopy(plt.rcParams)

        try:
            with figure_context(FigurePreset.JOSS):
                assert plt.rcParams["font.size"] == 9
                raise ValueError("Test exception")
        except ValueError:
            pass

        # rcParams should be restored despite exception
        assert plt.rcParams == original_params

    def test_context_figures_use_preset(self) -> None:
        """Test that figures created in context use preset styling."""
        with figure_context(FigurePreset.IEEE):
            fig, ax = plt.subplots()
            # Figure created with IEEE preset figsize
            assert fig.get_figwidth() == pytest.approx(3.5, abs=0.01)
            assert fig.get_figheight() == pytest.approx(2.5, abs=0.01)
            plt.close(fig)


class TestSaveFigure:
    """Test save_figure() function."""

    def test_save_figure_creates_file(self, tmp_path: Path) -> None:
        """Test that save_figure creates a file."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "test_plot.png"
        result_path = save_figure(fig, output_path)

        assert output_path.exists()
        assert result_path == output_path.resolve()
        plt.close(fig)

    def test_save_figure_returns_path(self, tmp_path: Path) -> None:
        """Test that save_figure returns Path object."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "test.png"
        result = save_figure(fig, output_path)

        assert isinstance(result, Path)
        assert result.is_absolute()
        plt.close(fig)

    def test_save_figure_png(self, tmp_path: Path) -> None:
        """Test saving as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "plot.png"
        save_figure(fig, output_path, dpi=150)

        assert output_path.exists()
        assert output_path.suffix == ".png"
        plt.close(fig)

    def test_save_figure_pdf(self, tmp_path: Path) -> None:
        """Test saving as PDF."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "plot.pdf"
        save_figure(fig, output_path)

        assert output_path.exists()
        assert output_path.suffix == ".pdf"
        plt.close(fig)

    def test_save_figure_svg(self, tmp_path: Path) -> None:
        """Test saving as SVG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "plot.svg"
        save_figure(fig, output_path)

        assert output_path.exists()
        assert output_path.suffix == ".svg"
        plt.close(fig)

    def test_save_figure_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that save_figure creates parent directories."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "nested" / "dir" / "structure" / "plot.png"
        save_figure(fig, output_path)

        assert output_path.exists()
        plt.close(fig)

    def test_save_figure_with_dpi(self, tmp_path: Path) -> None:
        """Test saving with custom DPI."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_300 = tmp_path / "plot_300dpi.png"
        output_150 = tmp_path / "plot_150dpi.png"

        save_figure(fig, output_300, dpi=300)
        save_figure(fig, output_150, dpi=150)

        # Higher DPI file should be larger (more pixels)
        assert output_300.stat().st_size > output_150.stat().st_size
        plt.close(fig)

    def test_save_figure_string_path(self, tmp_path: Path) -> None:
        """Test that save_figure accepts string paths."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_str = str(tmp_path / "plot.png")
        result = save_figure(fig, output_str)

        assert Path(output_str).exists()
        assert isinstance(result, Path)
        plt.close(fig)

    def test_save_figure_transparent(self, tmp_path: Path) -> None:
        """Test saving with transparent background."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "transparent.png"
        save_figure(fig, output_path, transparent=True)

        assert output_path.exists()
        plt.close(fig)

    def test_save_figure_tight_bbox(self, tmp_path: Path) -> None:
        """Test saving with tight bounding box."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        ax.set_title("Test Title")

        output_path = tmp_path / "tight.png"
        save_figure(fig, output_path, bbox_inches="tight", pad_inches=0.05)

        assert output_path.exists()
        plt.close(fig)


class TestListPresets:
    """Test list_presets() function."""

    def test_list_presets_returns_dict(self) -> None:
        """Test that list_presets returns a dict."""
        presets = list_presets()
        assert isinstance(presets, dict)

    def test_list_presets_has_all_presets(self) -> None:
        """Test that all presets are listed."""
        presets = list_presets()
        assert "joss" in presets
        assert "ieee" in presets
        assert "elsevier" in presets
        assert "nature" in presets

    def test_list_presets_has_descriptions(self) -> None:
        """Test that presets have non-empty descriptions."""
        presets = list_presets()
        for name, desc in presets.items():
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestIntegration:
    """Integration tests combining multiple features."""

    def setup_method(self) -> None:
        """Save original rcParams before each test."""
        self.original_params = copy.deepcopy(plt.rcParams)

    def teardown_method(self) -> None:
        """Restore original rcParams after each test."""
        plt.rcParams.update(self.original_params)
        plt.close("all")

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: apply preset, create figure, save."""
        apply_figure_preset(FigurePreset.JOSS)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_title("Test Figure")

        output_path = tmp_path / "test.png"
        result = save_figure(fig, output_path, dpi=300)

        assert result.exists()
        assert plt.rcParams["font.size"] == 9
        plt.close(fig)

    def test_context_manager_with_save(self, tmp_path: Path) -> None:
        """Test context manager with figure saving."""
        original_params = copy.deepcopy(plt.rcParams)

        with figure_context(FigurePreset.NATURE):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            output_path = tmp_path / "nature_fig.png"
            save_figure(fig, output_path)
            plt.close(fig)

        assert (tmp_path / "nature_fig.png").exists()
        assert plt.rcParams == original_params

    def test_multiple_formats(self, tmp_path: Path) -> None:
        """Test saving same figure in multiple formats."""
        with figure_context(FigurePreset.IEEE):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            for ext in ["png", "pdf", "svg"]:
                path = tmp_path / f"plot.{ext}"
                save_figure(fig, path)
                assert path.exists()

            plt.close(fig)
