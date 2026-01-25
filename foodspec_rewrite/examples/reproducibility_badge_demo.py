#!/usr/bin/env python3
"""
Reproducibility Badge Demo
===========================

Demonstrates the reproducibility badge generator for FoodSpec workflows.

The badge visually indicates reproducibility status:
- Green: Fully reproducible (all components present)
- Yellow: Partially reproducible (missing only environment hash)
- Red: Not reproducible (missing critical components)

Components tracked:
1. Random seed
2. Protocol hash
3. Data hash
4. Environment hash
"""

from pathlib import Path
from unittest.mock import MagicMock

from foodspec.viz import plot_reproducibility_badge, get_reproducibility_status


def create_demo_output_dir():
    """Create directory for demo outputs."""
    output_dir = Path("outputs/badge_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def demo_fully_reproducible():
    """Demo: Fully reproducible manifest (green badge)."""
    print("\n" + "=" * 70)
    print("DEMO 1: Fully Reproducible Workflow")
    print("=" * 70)
    
    # Create manifest with all reproducibility components
    manifest = MagicMock()
    manifest.seed = 42
    manifest.protocol_hash = "abc123def456789"
    manifest.data_hash = "xyz789uvw012345"
    manifest.env_hash = "env456hash789012"
    
    # Get status
    status = get_reproducibility_status(manifest)
    
    print("\nReproducibility Status:")
    print(f"  Level: {status['level']}")
    print(f"  Status: {status['status']}")
    print(f"  Components Present: {status['components_present']}/{status['total_components']}")
    print(f"  Fully Reproducible: {status['is_fully_reproducible']}")
    print(f"  Missing: {status['missing_components'] if status['missing_components'] else 'None'}")
    
    print("\nComponent Details:")
    for name, value in status["components"].items():
        present = "✓" if value is not None else "✗"
        print(f"  {present} {name}: {value}")
    
    # Generate badge
    output_dir = create_demo_output_dir()
    badge_dir = output_dir / "green_badge"
    fig = plot_reproducibility_badge(manifest, save_path=badge_dir)
    
    badge_path = badge_dir / "reproducibility_badge.png"
    print(f"\n✓ Green badge saved: {badge_path}")
    
    return fig


def demo_partially_reproducible():
    """Demo: Partially reproducible manifest (yellow badge)."""
    print("\n" + "=" * 70)
    print("DEMO 2: Partially Reproducible Workflow (Missing Environment)")
    print("=" * 70)
    
    # Create manifest missing environment hash
    manifest = MagicMock()
    manifest.seed = 123
    manifest.protocol_hash = "proto_abc123"
    manifest.data_hash = "data_xyz789"
    manifest.env_hash = None  # Missing environment hash
    
    # Get status
    status = get_reproducibility_status(manifest)
    
    print("\nReproducibility Status:")
    print(f"  Level: {status['level']}")
    print(f"  Status: {status['status']}")
    print(f"  Components Present: {status['components_present']}/{status['total_components']}")
    print(f"  Partially Reproducible: {status['is_partially_reproducible']}")
    print(f"  Missing: {status['missing_components']}")
    
    print("\nComponent Details:")
    for name, value in status["components"].items():
        present = "✓" if value is not None else "✗"
        print(f"  {present} {name}: {value if value else 'MISSING'}")
    
    # Generate badge
    output_dir = create_demo_output_dir()
    badge_dir = output_dir / "yellow_badge"
    fig = plot_reproducibility_badge(manifest, save_path=badge_dir)
    
    badge_path = badge_dir / "reproducibility_badge.png"
    print(f"\n✓ Yellow badge saved: {badge_path}")
    
    return fig


def demo_not_reproducible():
    """Demo: Non-reproducible manifest (red badge)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Not Reproducible Workflow (Missing Critical Items)")
    print("=" * 70)
    
    # Create manifest missing critical components
    manifest = MagicMock()
    manifest.seed = None  # Missing seed
    manifest.protocol_hash = "proto_def456"
    manifest.data_hash = None  # Missing data hash
    manifest.env_hash = None
    
    # Get status
    status = get_reproducibility_status(manifest)
    
    print("\nReproducibility Status:")
    print(f"  Level: {status['level']}")
    print(f"  Status: {status['status']}")
    print(f"  Components Present: {status['components_present']}/{status['total_components']}")
    print(f"  Fully Reproducible: {status['is_fully_reproducible']}")
    print(f"  Missing: {status['missing_components']}")
    
    print("\nComponent Details:")
    for name, value in status["components"].items():
        present = "✓" if value is not None else "✗"
        print(f"  {present} {name}: {value if value else 'MISSING'}")
    
    # Generate badge
    output_dir = create_demo_output_dir()
    badge_dir = output_dir / "red_badge"
    fig = plot_reproducibility_badge(manifest, save_path=badge_dir)
    
    badge_path = badge_dir / "reproducibility_badge.png"
    print(f"\n✓ Red badge saved: {badge_path}")
    
    return fig


def demo_nested_attributes():
    """Demo: Manifest with nested attribute structure."""
    print("\n" + "=" * 70)
    print("DEMO 4: Nested Attribute Structure")
    print("=" * 70)
    
    # Create manifest with nested attributes
    manifest = MagicMock()
    manifest.config = MagicMock()
    manifest.config.seed = 999
    manifest.hashes = MagicMock()
    manifest.hashes.protocol = "nested_proto_hash"
    manifest.hashes.data = "nested_data_hash"
    manifest.environment = MagicMock()
    manifest.environment.hash = "nested_env_hash"
    
    # Get status
    status = get_reproducibility_status(manifest)
    
    print("\nReproducibility Status:")
    print(f"  Level: {status['level']}")
    print(f"  Status: {status['status']}")
    print(f"  Components Present: {status['components_present']}/{status['total_components']}")
    
    print("\nComponent Details (from nested attributes):")
    for name, value in status["components"].items():
        present = "✓" if value is not None else "✗"
        print(f"  {present} {name}: {value}")
    
    # Generate badge
    output_dir = create_demo_output_dir()
    badge_dir = output_dir / "nested_badge"
    fig = plot_reproducibility_badge(manifest, save_path=badge_dir)
    
    badge_path = badge_dir / "reproducibility_badge.png"
    print(f"\n✓ Badge saved (nested): {badge_path}")
    
    return fig


def demo_custom_styling():
    """Demo: Custom badge styling."""
    print("\n" + "=" * 70)
    print("DEMO 5: Custom Badge Styling")
    print("=" * 70)
    
    manifest = MagicMock()
    manifest.seed = 777
    manifest.protocol_hash = "custom_proto"
    manifest.data_hash = "custom_data"
    manifest.env_hash = "custom_env"
    
    output_dir = create_demo_output_dir()
    
    # Generate badges with different sizes
    print("\nGenerating badges with custom sizes:")
    
    sizes = [
        ((4, 2), "default"),
        ((6, 3), "large"),
        ((3, 1.5), "small"),
    ]
    
    for (width, height), size_name in sizes:
        badge_dir = output_dir / f"custom_{size_name}"
        fig = plot_reproducibility_badge(
            manifest,
            save_path=badge_dir,
            figure_size=(width, height),
            dpi=200
        )
        badge_path = badge_dir / "reproducibility_badge.png"
        print(f"  ✓ {size_name} badge ({width}x{height}): {badge_path}")
    
    return fig


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("FoodSpec Reproducibility Badge Demo")
    print("=" * 70)
    print("\nThis demo showcases reproducibility badge generation.")
    print("Badges provide at-a-glance status of workflow reproducibility.")
    
    # Run demos
    demo_fully_reproducible()
    demo_partially_reproducible()
    demo_not_reproducible()
    demo_nested_attributes()
    demo_custom_styling()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nGenerated 8 reproducibility badges:")
    print("  • outputs/badge_demo/green_badge/ (fully reproducible)")
    print("  • outputs/badge_demo/yellow_badge/ (partially reproducible)")
    print("  • outputs/badge_demo/red_badge/ (not reproducible)")
    print("  • outputs/badge_demo/nested_badge/ (nested attributes)")
    print("  • outputs/badge_demo/custom_default/ (default size)")
    print("  • outputs/badge_demo/custom_large/ (large size)")
    print("  • outputs/badge_demo/custom_small/ (small size)")
    
    print("\nBadge Color System:")
    print("  🟢 Green: All components present (seed, protocol, data, env)")
    print("  🟡 Yellow: Missing only environment hash")
    print("  🔴 Red: Missing critical components (seed, protocol, or data)")
    
    print("\nUse badges in:")
    print("  • Research papers and reports")
    print("  • Documentation and README files")
    print("  • Quality control dashboards")
    print("  • Reproducibility audits")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
