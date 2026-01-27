"""Demonstration of paper-ready figure presets for different journals.

This example shows how to use FoodSpec's figure presets to style matplotlib
figures for publication in different scientific journals (JOSS, IEEE, Elsevier, Nature).

The key features:
- apply_figure_preset(): Set global matplotlib rcParams for a journal style
- figure_context(): Temporarily apply a preset within a code block
- save_figure(): Save publication-ready figures with proper DPI and formatting
"""

from pathlib import Path

import matplotlib.pyplot as plt

from foodspec.viz.paper import (
    FigurePreset,
    apply_figure_preset,
    figure_context,
    list_presets,
    save_figure,
)


def main() -> None:
    """Run paper presets demo."""
    # Print available presets
    print("Available figure presets:")
    print("-" * 50)
    presets = list_presets()
    for name, description in presets.items():
        print(f"  {name:10s}: {description}")
    print()

    # Create output directory
    output_dir = Path("outputs/paper_presets_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example 1: Global preset application
    print("Example 1: Applying JOSS preset globally")
    print("-" * 50)
    apply_figure_preset(FigurePreset.JOSS)

    fig, ax = plt.subplots()
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 2, 3, 5]
    ax.plot(x, y, marker="o", label="Data")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title("JOSS Preset Example")
    ax.legend()

    save_figure(fig, output_dir / "joss_example.png", dpi=300)
    print(f"  ✓ Saved JOSS-styled figure to {output_dir / 'joss_example.png'}")
    plt.close(fig)
    print()

    # Example 2: Context manager for temporary preset
    print("Example 2: Using context manager for temporary IEEE preset")
    print("-" * 50)
    with figure_context(FigurePreset.IEEE):
        fig, ax = plt.subplots()
        ax.plot(x, y, marker="s", label="Data")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_title("IEEE Preset Example")
        ax.legend()

        save_figure(fig, output_dir / "ieee_example.png", dpi=300)
        print(f"  ✓ Saved IEEE-styled figure to {output_dir / 'ieee_example.png'}")
        plt.close(fig)

    print("  ✓ rcParams restored after context exit")
    print()

    # Example 3: Create figures for all presets
    print("Example 3: Creating figures for all presets")
    print("-" * 50)
    for preset in FigurePreset:
        with figure_context(preset):
            fig, ax = plt.subplots()
            ax.plot(x, y, marker="^")
            ax.fill_between(x, y, alpha=0.3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"{preset.value.upper()} Preset")

            filename = f"{preset.value}_preset.png"
            save_figure(fig, output_dir / filename, dpi=300)
            print(f"  ✓ Saved {preset.value:10s} figure to {output_dir / filename}")
            plt.close(fig)

    print()
    print("=" * 50)
    print(f"All examples completed! Figures saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
